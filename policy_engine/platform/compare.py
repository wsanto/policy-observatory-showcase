"""Scenario Comparison API — side-by-side run metrics with deltas."""

import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, set_tenant_context
from .middleware import require_role
from .models import Artifact, Run
from .storage import download_file

router = APIRouter(tags=["compare"])


def _safe_get(data: dict, *keys, default=None):
    """Nested dict access without KeyError."""
    for k in keys:
        if isinstance(data, dict):
            data = data.get(k, default)
        else:
            return default
    return data


def _extract_run_metrics(snapshot: dict) -> dict:
    """Extract comparable metrics from a run snapshot."""
    ticks = snapshot.get("ticks", [])
    breakthroughs = snapshot.get("breakthroughs", [])
    gdp_projections = snapshot.get("gdp_projections", [])
    impact = snapshot.get("impact_assessment", {})
    scorecards = impact.get("scorecards", {})
    empirical = impact.get("empirical", {})

    # GDP summary
    gdp_summary = {}
    for g in gdp_projections:
        gdp_summary[g.get("scenario_name", "unknown")] = {
            "baseline_gdp_b": g.get("baseline_gdp_b", 0),
            "projected_gdp_b": g.get("projected_gdp_b", 0),
            "delta_pct": g.get("delta_pct", 0),
            "p5": _safe_get(g, "confidence_intervals", "p5", default=0),
            "p95": _safe_get(g, "confidence_intervals", "p95", default=0),
        }

    # Scorecard summary
    scorecard_summary = {}
    for obj_id, sc in scorecards.items():
        scorecard_summary[obj_id] = {
            "title": sc.get("objective_title", obj_id),
            "policies": sc.get("total_policies", 0),
            "risk_adj_gdp_pct": sc.get("risk_adjusted_gdp_pct", 0),
            "avg_novelty": sc.get("avg_novelty", 0),
            "kpis_on_track": sc.get("kpis_on_track", 0),
            "kpis_total": sc.get("kpis_total", 0),
        }

    # Stress tests
    stress_summary = {}
    for name, st in (empirical.get("stress_tests", {}) or {}).items():
        stress_summary[name] = {
            "gdp_impact_pct": st.get("gdp_impact_pct", 0),
            "worst_sector": st.get("worst_sector", ""),
        }

    # Systemic importance (top 3)
    systemic = []
    for s in (empirical.get("systemic_importance", []) or [])[:3]:
        systemic.append({"sector": s.get("sector", ""), "debtrank": s.get("debtrank", 0)})

    return {
        "total_ticks": len(ticks),
        "total_discoveries": len([t for t in ticks if t.get("outcome") == "discovered"]),
        "total_breakthroughs": len(breakthroughs),
        "avg_novelty": sum(t.get("novelty_score", 0) for t in ticks) / max(len(ticks), 1),
        "gdp": gdp_summary,
        "scorecards": scorecard_summary,
        "stress_tests": stress_summary,
        "systemic_importance": systemic,
        "counterfactual": empirical.get("counterfactual", {}),
    }


def _compute_deltas(baseline: dict, comparison: dict) -> dict:
    """Compute deltas between two sets of metrics."""
    deltas = {}

    # GDP deltas
    deltas["gdp"] = {}
    for scenario in set(list(baseline.get("gdp", {}).keys()) + list(comparison.get("gdp", {}).keys())):
        b = baseline.get("gdp", {}).get(scenario, {})
        c = comparison.get("gdp", {}).get(scenario, {})
        if b and c:
            deltas["gdp"][scenario] = {
                "projected_delta_b": round(c.get("projected_gdp_b", 0) - b.get("projected_gdp_b", 0), 2),
                "delta_pct_change": round(c.get("delta_pct", 0) - b.get("delta_pct", 0), 2),
            }

    # Scorecard deltas
    deltas["scorecards"] = {}
    for obj_id in set(list(baseline.get("scorecards", {}).keys()) + list(comparison.get("scorecards", {}).keys())):
        b = baseline.get("scorecards", {}).get(obj_id, {})
        c = comparison.get("scorecards", {}).get(obj_id, {})
        if b and c:
            deltas["scorecards"][obj_id] = {
                "title": c.get("title", obj_id),
                "gdp_pct_delta": round(c.get("risk_adj_gdp_pct", 0) - b.get("risk_adj_gdp_pct", 0), 4),
                "policies_delta": c.get("policies", 0) - b.get("policies", 0),
                "novelty_delta": round(c.get("avg_novelty", 0) - b.get("avg_novelty", 0), 4),
            }

    # Aggregate deltas
    deltas["discoveries_delta"] = comparison.get("total_discoveries", 0) - baseline.get("total_discoveries", 0)
    deltas["novelty_delta"] = round(comparison.get("avg_novelty", 0) - baseline.get("avg_novelty", 0), 4)

    return deltas


@router.get("/v1/projects/{pid}/compare")
async def compare_runs(
    pid: str,
    run_ids: str = Query(..., description="Comma-separated run IDs to compare"),
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    """Compare 2-4 runs side-by-side with delta calculations."""
    await set_tenant_context(db, user["tenant_id"])

    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 run IDs to compare")
    if len(ids) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 runs can be compared")

    results = []
    for rid in ids:
        # Verify run belongs to tenant + project
        run_result = await db.execute(
            select(Run).where(
                Run.id == uuid.UUID(rid),
                Run.project_id == uuid.UUID(pid),
                Run.tenant_id == uuid.UUID(user["tenant_id"]),
            )
        )
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {rid} not found in this project")
        if run.status != "completed":
            raise HTTPException(status_code=400, detail=f"Run {rid} is '{run.status}', not completed")

        # Load snapshot artifact
        art_result = await db.execute(
            select(Artifact).where(
                Artifact.run_id == uuid.UUID(rid),
                Artifact.artifact_type == "snapshot",
            )
        )
        artifact = art_result.scalar_one_or_none()
        if not artifact:
            raise HTTPException(status_code=404, detail=f"No snapshot artifact for run {rid}")

        snapshot = json.loads(download_file(artifact.storage_key))
        metrics = _extract_run_metrics(snapshot)

        results.append({
            "run_id": rid,
            "name": run.name,
            "run_mode": run.run_mode,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "metrics": metrics,
        })

    # Compute deltas (first run = baseline)
    baseline_metrics = results[0]["metrics"]
    for i in range(1, len(results)):
        results[i]["deltas"] = _compute_deltas(baseline_metrics, results[i]["metrics"])

    return {
        "project_id": pid,
        "baseline": results[0],
        "comparisons": results[1:],
        "run_count": len(results),
    }
