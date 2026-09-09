"""Continuous News-to-Shock Monitoring — watch topics, detect events, map to shocks, alert.

Fixes from Sprint 5.5:
- Configs and alerts persisted to Postgres (survive restarts)
- Immediate first poll on config save (no 10-min wait)
- Broader Brave search (past week, not just past day)
- Fallback classification when LLM fails
- Status logging so users can see what happened
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from .audit import log_action
from .database import get_db, set_tenant_context, tenant_session
from .middleware import require_role
from .models import Project, Scenario

router = APIRouter(tags=["monitoring"])


# ── In-memory task tracking (tasks can't be persisted, but configs/alerts are in DB now) ──

_monitor_tasks: Dict[str, asyncio.Task] = {}


# ── Schemas ──────────────────────────────────────────────────────────

class WatchConfig(BaseModel):
    topics: List[str] = []
    sectors: List[str] = []
    severity_threshold: float = 0.6
    poll_interval_minutes: int = 10
    auto_create_scenario: bool = False
    enabled: bool = True


class AlertResponse(BaseModel):
    id: str
    project_id: str
    title: str
    source: Optional[str]
    severity: float
    category: str
    affected_sectors: List[str]
    suggested_shock: Optional[dict]
    status: str
    detected_at: str


# ── Brave News Search (improved) ────────────────────────────────────

async def search_news(query: str, count: int = 5) -> List[dict]:
    """Search Brave News API — uses past week for better coverage."""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        logger.warning("[NewsMonitor] No BRAVE_API_KEY — search disabled")
        return []

    url = "https://api.search.brave.com/res/v1/news/search"
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key}
    params = {"q": query, "count": count, "freshness": "pw"}  # past week (not past day)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning(f"[NewsMonitor] Brave search returned {resp.status} for '{query}'")
                    return []
                data = await resp.json()
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "source": r.get("meta_url", {}).get("hostname", ""),
                        "age": r.get("age", ""),
                    }
                    for r in data.get("results", [])[:count]
                ]
                logger.info(f"[NewsMonitor] Brave search '{query}': {len(results)} results")
                return results
    except Exception as e:
        logger.warning(f"[NewsMonitor] Brave search failed: {e}")
        return []


# ── LLM Event Classification (with fallback) ────────────────────────

CLASSIFY_PROMPT = """Analyze this news event for a policy simulation platform.

Event: {title}
Description: {description}
Context: {context}

Return VALID JSON only:
{{"severity": 0.0-1.0, "category": "CONFLICT|SANCTIONS|ECONOMIC|DIPLOMATIC|TECHNOLOGY|CLIMATE|REGULATORY", "affected_sectors": ["sector1"], "suggested_shock": {{"sector": "most affected", "magnitude": 0.01-0.5, "duration_days": 30-365, "reasoning": "one sentence"}}, "relevance": 0.0-1.0}}"""


async def classify_event(title: str, description: str, context: str) -> Optional[dict]:
    """LLM classification with keyword fallback."""
    from policy_engine.connectors.kimi import KimiClient

    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        # Keyword-based fallback classification
        return _keyword_classify(title, description)

    client = KimiClient(api_key=api_key)
    prompt = CLASSIFY_PROMPT.format(title=title, description=description[:300], context=context)

    try:
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "Geopolitical risk analyst. Valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = response.get("content", "") or response.get("reasoning", "")
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        json_str = re.sub(r',\s*}', '}', json_str.strip())
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"[NewsMonitor] LLM classification failed: {e}, using keyword fallback")
        return _keyword_classify(title, description)


def _keyword_classify(title: str, description: str) -> dict:
    """Fast keyword-based fallback when LLM is unavailable."""
    text = f"{title} {description}".lower()

    category = "ECONOMIC"
    severity = 0.5
    sector = "Government"
    magnitude = 0.05

    if any(w in text for w in ["war", "conflict", "military", "attack", "strike"]):
        category, severity, sector, magnitude = "CONFLICT", 0.8, "Government", 0.15
    elif any(w in text for w in ["sanction", "embargo", "ban", "restrict", "tariff"]):
        category, severity, sector, magnitude = "SANCTIONS", 0.7, "Technology", 0.12
    elif any(w in text for w in ["ai", "chip", "semiconductor", "tech", "digital"]):
        category, severity, sector, magnitude = "TECHNOLOGY", 0.6, "Technology", 0.08
    elif any(w in text for w in ["oil", "energy", "opec", "gas", "price"]):
        category, severity, sector, magnitude = "ECONOMIC", 0.65, "Energy", 0.10
    elif any(w in text for w in ["regulation", "law", "compliance", "policy", "govern"]):
        category, severity, sector, magnitude = "REGULATORY", 0.5, "Government", 0.05
    elif any(w in text for w in ["climate", "carbon", "green", "renewable"]):
        category, severity, sector, magnitude = "CLIMATE", 0.5, "Energy", 0.06
    elif any(w in text for w in ["diplomat", "treaty", "agreement", "summit"]):
        category, severity, sector, magnitude = "DIPLOMATIC", 0.4, "Government", 0.03

    return {
        "severity": severity,
        "category": category,
        "affected_sectors": [sector],
        "suggested_shock": {"sector": sector, "magnitude": magnitude, "duration_days": 180, "reasoning": f"Keyword-classified from: {title[:60]}"},
        "relevance": 0.6,
    }


# ── Persistence helpers (Postgres via project settings JSONB) ────────

async def _load_config(tenant_id: str, project_id: str) -> dict:
    """Load monitoring config from project settings."""
    async with tenant_session(tenant_id) as db:
        result = await db.execute(
            select(Project.settings).where(Project.id == uuid.UUID(project_id))
        )
        row = result.scalar_one_or_none()
        if row and isinstance(row, dict):
            return row.get("monitoring_config", {})
    return {}


async def _save_config(tenant_id: str, project_id: str, config: dict):
    """Save monitoring config to project settings."""
    async with tenant_session(tenant_id) as db:
        result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
        project = result.scalar_one_or_none()
        if project:
            from sqlalchemy.orm.attributes import flag_modified
            settings = dict(project.settings or {})
            settings["monitoring_config"] = config
            project.settings = settings
            flag_modified(project, "settings")
            await db.flush()


async def _load_alerts(tenant_id: str, project_id: str) -> List[dict]:
    """Load alerts from project settings."""
    async with tenant_session(tenant_id) as db:
        result = await db.execute(
            select(Project.settings).where(Project.id == uuid.UUID(project_id))
        )
        row = result.scalar_one_or_none()
        if row and isinstance(row, dict):
            return row.get("monitoring_alerts", [])
    return []


async def _save_alerts(tenant_id: str, project_id: str, alerts: List[dict]):
    """Save alerts to project settings (keep last 100). Uses direct SQL UPDATE to avoid JSONB mutation detection issues."""
    import json as _json
    async with tenant_session(tenant_id) as db:
        result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
        project = result.scalar_one_or_none()
        if project:
            settings = dict(project.settings or {})  # Create NEW dict (not in-place mutation)
            settings["monitoring_alerts"] = alerts[-100:]
            # Force SQLAlchemy to detect the change by assigning a new object
            from sqlalchemy.orm.attributes import flag_modified
            project.settings = settings
            flag_modified(project, "settings")
            await db.flush()
            logger.debug(f"[NewsMonitor] Saved {len(alerts)} alerts for {project_id[:8]}")


# ── Background Monitor Task ──────────────────────────────────────────

async def _monitor_loop(project_id: str, tenant_id: str, config: dict):
    """Background polling loop with persistence."""
    topics = config.get("topics", [])
    sectors = config.get("sectors", [])
    threshold = config.get("severity_threshold", 0.6)
    interval = config.get("poll_interval_minutes", 10) * 60
    auto_scenario = config.get("auto_create_scenario", False)
    context = f"Topics: {', '.join(topics)}. Sectors: {', '.join(sectors)}."

    logger.info(f"[NewsMonitor] Started for {project_id[:8]}: topics={topics}, threshold={threshold}")

    # Run immediately on first tick (don't wait 10 min)
    first_run = True

    while True:
        try:
            if not first_run:
                await asyncio.sleep(interval)
            first_run = False

            # Load existing alerts from DB
            alerts = await _load_alerts(tenant_id, project_id)
            existing_titles = {a["title"] for a in alerts}
            new_count = 0

            for topic in topics:
                articles = await search_news(topic, count=5)

                for article in articles:
                    if article["title"] in existing_titles:
                        continue

                    classification = await classify_event(
                        article["title"], article.get("description", ""), context
                    )
                    if not classification:
                        continue

                    severity = classification.get("severity", 0)
                    relevance = classification.get("relevance", 0)

                    if severity >= threshold and relevance >= 0.3:
                        alert = {
                            "id": str(uuid.uuid4()),
                            "project_id": project_id,
                            "title": article["title"],
                            "source": article.get("source", ""),
                            "url": article.get("url", ""),
                            "severity": severity,
                            "category": classification.get("category", "UNKNOWN"),
                            "affected_sectors": classification.get("affected_sectors", []),
                            "suggested_shock": classification.get("suggested_shock"),
                            "status": "new",
                            "detected_at": datetime.now(timezone.utc).isoformat(),
                        }
                        alerts.append(alert)
                        existing_titles.add(article["title"])
                        new_count += 1
                        logger.info(f"[NewsMonitor] Alert: {article['title'][:60]} (severity={severity:.2f})")

                        if auto_scenario and classification.get("suggested_shock"):
                            shock = classification["suggested_shock"]
                            try:
                                async with tenant_session(tenant_id) as db:
                                    scenario = Scenario(
                                        id=uuid.uuid4(),
                                        project_id=uuid.UUID(project_id),
                                        tenant_id=uuid.UUID(tenant_id),
                                        name=f"Auto: {article['title'][:60]}",
                                        description=f"Auto-generated: {article.get('url', '')}",
                                        shocks=[{
                                            "name": article["title"][:80],
                                            "type": "sector",
                                            "sector": shock.get("sector", ""),
                                            "magnitude": shock.get("magnitude", 0.1),
                                            "duration_days": shock.get("duration_days", 180),
                                        }],
                                    )
                                    db.add(scenario)
                                alert["status"] = "scenario_created"
                            except Exception as e:
                                logger.warning(f"[NewsMonitor] Auto-scenario failed: {e}")

            # Persist alerts to DB
            await _save_alerts(tenant_id, project_id, alerts)
            if new_count > 0:
                logger.info(f"[NewsMonitor] {project_id[:8]}: {new_count} new alerts (total: {len(alerts)})")
            else:
                logger.debug(f"[NewsMonitor] {project_id[:8]}: poll complete, no new alerts")

        except asyncio.CancelledError:
            logger.info(f"[NewsMonitor] Stopped for {project_id[:8]}")
            return
        except Exception as e:
            logger.error(f"[NewsMonitor] Poll error for {project_id[:8]}: {e}")
            await asyncio.sleep(60)  # Wait 1 min on error, then retry


# ── Routes ───────────────────────────────────────────────────────────

async def restore_monitors_from_db():
    """Called on server startup — restarts all enabled monitors from persisted configs."""
    from .database import async_session_factory
    from .models import Project, Membership

    try:
        async with async_session_factory() as db:
            # Find all projects with monitoring enabled
            result = await db.execute(select(Project).where(Project.settings != text("'{}'::jsonb")))
            projects = result.scalars().all()

            restored = 0
            for project in projects:
                settings = project.settings or {}
                config = settings.get("monitoring_config", {})
                if config.get("enabled") and config.get("topics"):
                    pid = str(project.id)
                    tid = str(project.tenant_id)
                    if pid not in _monitor_tasks:
                        task = asyncio.create_task(_monitor_loop(pid, tid, config))
                        _monitor_tasks[pid] = task
                        restored += 1

            if restored > 0:
                logger.info(f"[NewsMonitor] Restored {restored} monitors from database on startup")
    except Exception as e:
        logger.warning(f"[NewsMonitor] Failed to restore monitors: {e}")


@router.post("/v1/projects/{pid}/monitoring/poll-now")
async def poll_now(
    pid: str,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger an immediate poll (useful for testing)."""
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(select(Project).where(
        Project.id == uuid.UUID(pid), Project.tenant_id == uuid.UUID(user["tenant_id"])
    ))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    config = (project.settings or {}).get("monitoring_config", {})
    if not config.get("topics"):
        raise HTTPException(status_code=400, detail="No watch topics configured")

    # Run one poll cycle synchronously
    topics = config.get("topics", [])
    threshold = config.get("severity_threshold", 0.6)
    context = f"Topics: {', '.join(topics)}. Sectors: {', '.join(config.get('sectors', []))}."

    alerts = await _load_alerts(user["tenant_id"], pid)
    existing_titles = {a["title"] for a in alerts}
    new_alerts = []

    for topic in topics:
        articles = await search_news(topic, count=5)
        for article in articles:
            if article["title"] in existing_titles:
                continue
            classification = await classify_event(article["title"], article.get("description", ""), context)
            if not classification:
                continue
            severity = classification.get("severity", 0)
            relevance = classification.get("relevance", 0)
            if severity >= threshold and relevance >= 0.3:
                alert = {
                    "id": str(uuid.uuid4()), "project_id": pid, "title": article["title"],
                    "source": article.get("source", ""), "url": article.get("url", ""),
                    "severity": severity, "category": classification.get("category", "UNKNOWN"),
                    "affected_sectors": classification.get("affected_sectors", []),
                    "suggested_shock": classification.get("suggested_shock"),
                    "status": "new", "detected_at": datetime.now(timezone.utc).isoformat(),
                }
                alerts.append(alert)
                new_alerts.append(alert)
                existing_titles.add(article["title"])

    await _save_alerts(user["tenant_id"], pid, alerts)

    # Also ensure background task is running
    if pid not in _monitor_tasks or _monitor_tasks[pid].done():
        if config.get("enabled") and config.get("topics"):
            task = asyncio.create_task(_monitor_loop(pid, user["tenant_id"], config))
            _monitor_tasks[pid] = task

    return {"new_alerts": len(new_alerts), "total_alerts": len(alerts), "articles_searched": sum(1 for _ in topics)}


@router.get("/v1/projects/{pid}/monitoring")
async def get_monitoring_config(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    """Get monitoring config (from DB, survives restarts)."""
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(select(Project.settings).where(
        Project.id == uuid.UUID(pid), Project.tenant_id == uuid.UUID(user["tenant_id"])
    ))
    settings = result.scalar_one_or_none() or {}
    config = settings.get("monitoring_config", {}) if isinstance(settings, dict) else {}
    is_active = pid in _monitor_tasks and not _monitor_tasks[pid].done()
    return {"config": config, "active": is_active, "project_id": pid}


@router.put("/v1/projects/{pid}/monitoring")
async def update_monitoring_config(
    pid: str,
    req: WatchConfig,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    """Save config to DB and start monitor. Runs first poll immediately."""
    await set_tenant_context(db, user["tenant_id"])

    result = await db.execute(
        select(Project).where(Project.id == uuid.UUID(pid), Project.tenant_id == uuid.UUID(user["tenant_id"]))
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    config = req.model_dump()

    # Persist to DB
    settings = project.settings or {}
    settings["monitoring_config"] = config
    project.settings = settings
    await db.flush()

    # Stop existing monitor
    if pid in _monitor_tasks:
        _monitor_tasks[pid].cancel()
        del _monitor_tasks[pid]

    # Start new monitor if enabled
    if req.enabled and req.topics:
        task = asyncio.create_task(_monitor_loop(pid, user["tenant_id"], config))
        _monitor_tasks[pid] = task

    return {"config": config, "active": req.enabled and bool(req.topics), "project_id": pid}


@router.delete("/v1/projects/{pid}/monitoring")
async def stop_monitoring(
    pid: str,
    user: dict = Depends(require_role("analyst")),
):
    if pid in _monitor_tasks:
        _monitor_tasks[pid].cancel()
        del _monitor_tasks[pid]
    return {"active": False, "project_id": pid}


@router.get("/v1/projects/{pid}/alerts")
async def get_alerts(
    pid: str,
    status: Optional[str] = Query(None),
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    """Get alerts from DB (persisted, survive restarts)."""
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(select(Project.settings).where(
        Project.id == uuid.UUID(pid), Project.tenant_id == uuid.UUID(user["tenant_id"])
    ))
    settings = result.scalar_one_or_none() or {}
    alerts = settings.get("monitoring_alerts", []) if isinstance(settings, dict) else []
    if status:
        alerts = [a for a in alerts if a.get("status") == status]
    return {"alerts": alerts, "total": len(alerts), "project_id": pid}


@router.post("/v1/projects/{pid}/alerts/{aid}/acknowledge")
async def acknowledge_alert(
    pid: str, aid: str,
    user: dict = Depends(require_role("analyst")),
):
    """Update alert status in DB."""
    return await _update_alert_status(pid, aid, "acknowledged", user["tenant_id"])


@router.post("/v1/projects/{pid}/alerts/{aid}/dismiss")
async def dismiss_alert(
    pid: str, aid: str,
    user: dict = Depends(require_role("analyst")),
):
    return await _update_alert_status(pid, aid, "dismissed", user["tenant_id"])


async def _update_alert_status(project_id: str, alert_id: str, new_status: str, tenant_id: str) -> dict:
    """Update a specific alert's status in persisted alerts."""
    alerts = await _load_alerts(tenant_id, project_id)
    for alert in alerts:
        if alert["id"] == alert_id:
            alert["status"] = new_status
            await _save_alerts(tenant_id, project_id, alerts)
            return {"status": new_status, "alert_id": alert_id}
    raise HTTPException(status_code=404, detail="Alert not found")


@router.post("/v1/projects/{pid}/alerts/{aid}/create-scenario")
async def create_scenario_from_alert(
    pid: str, aid: str,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    alerts = await _load_alerts(user["tenant_id"], pid)

    alert = next((a for a in alerts if a["id"] == aid), None)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    shock = alert.get("suggested_shock")
    if not shock:
        raise HTTPException(status_code=400, detail="Alert has no suggested shock")

    scenario = Scenario(
        id=uuid.uuid4(),
        project_id=uuid.UUID(pid),
        tenant_id=uuid.UUID(user["tenant_id"]),
        name=f"Event: {alert['title'][:60]}",
        description=f"From monitored event. Source: {alert.get('url', '')}",
        shocks=[{
            "name": alert["title"][:80], "type": "sector",
            "sector": shock.get("sector", ""), "magnitude": shock.get("magnitude", 0.1),
            "duration_days": shock.get("duration_days", 180),
            "description": shock.get("reasoning", ""),
        }],
        created_by=uuid.UUID(user["sub"]),
    )
    db.add(scenario)
    await db.flush()

    alert["status"] = "scenario_created"
    await _save_alerts(user["tenant_id"], pid, alerts)

    return {"status": "scenario_created", "scenario_id": str(scenario.id), "alert_id": aid}
