"""Scenario builder + shock library routes."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .audit import log_action
from .database import get_db, set_tenant_context
from .middleware import require_role
from .models import Project, Scenario

router = APIRouter(tags=["scenarios"])


# ── Schemas ──────────────────────────────────────────────────────────

class ShockDef(BaseModel):
    name: str
    type: str = "sector"  # sector, price, supply, regulatory, geopolitical
    sector: Optional[str] = None
    magnitude: float = 0.1  # 0-1 scale
    duration_days: int = 180
    description: Optional[str] = None


class CreateScenarioRequest(BaseModel):
    name: str
    description: Optional[str] = None
    shocks: list[ShockDef] = []


class ScenarioResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str]
    shocks: list
    created_at: str


# ── Preset shock library ─────────────────────────────────────────────

SHOCK_LIBRARY = [
    {
        "id": "hormuz_blockade",
        "name": "Strait of Hormuz Blockade",
        "category": "geopolitical",
        "shocks": [
            {"name": "Energy supply disruption", "type": "sector", "sector": "Energy", "magnitude": 0.25, "duration_days": 120},
            {"name": "Transport disruption", "type": "sector", "sector": "Transportation", "magnitude": 0.15, "duration_days": 90},
            {"name": "Water infrastructure stress", "type": "sector", "sector": "Water", "magnitude": 0.05, "duration_days": 60},
        ],
    },
    {
        "id": "chip_embargo",
        "name": "Global AI Chip Embargo",
        "category": "technology",
        "shocks": [
            {"name": "Tech sector contraction", "type": "sector", "sector": "Technology", "magnitude": 0.20, "duration_days": 365},
            {"name": "Education pipeline disruption", "type": "sector", "sector": "Education", "magnitude": 0.10, "duration_days": 180},
            {"name": "Space program delays", "type": "sector", "sector": "Space", "magnitude": 0.08, "duration_days": 270},
        ],
    },
    {
        "id": "oil_collapse",
        "name": "Oil Price Collapse ($30/bbl)",
        "category": "economic",
        "shocks": [
            {"name": "Energy revenue collapse", "type": "sector", "sector": "Energy", "magnitude": 0.30, "duration_days": 365},
            {"name": "Government spending cuts", "type": "sector", "sector": "Government", "magnitude": 0.15, "duration_days": 270},
        ],
    },
    {
        "id": "talent_exodus",
        "name": "Talent Exodus (10% workforce)",
        "category": "social",
        "shocks": [
            {"name": "Education brain drain", "type": "sector", "sector": "Education", "magnitude": 0.15, "duration_days": 365},
            {"name": "Tech talent loss", "type": "sector", "sector": "Technology", "magnitude": 0.10, "duration_days": 365},
            {"name": "Healthcare workforce gap", "type": "sector", "sector": "Healthcare", "magnitude": 0.08, "duration_days": 180},
        ],
    },
    {
        "id": "cyber_attack",
        "name": "Critical Infrastructure Cyber Attack",
        "category": "security",
        "shocks": [
            {"name": "Water systems compromised", "type": "sector", "sector": "Water", "magnitude": 0.20, "duration_days": 60},
            {"name": "Energy grid disruption", "type": "sector", "sector": "Energy", "magnitude": 0.15, "duration_days": 45},
            {"name": "Tech infrastructure damage", "type": "sector", "sector": "Technology", "magnitude": 0.10, "duration_days": 90},
        ],
    },
    {
        "id": "regional_conflict",
        "name": "Regional Conflict Escalation",
        "category": "geopolitical",
        "shocks": [
            {"name": "Multi-sector disruption", "type": "sector", "sector": "Energy", "magnitude": 0.15, "duration_days": 180},
            {"name": "Transport disruption", "type": "sector", "sector": "Transportation", "magnitude": 0.12, "duration_days": 120},
            {"name": "Government emergency spending", "type": "sector", "sector": "Government", "magnitude": 0.10, "duration_days": 90},
            {"name": "Tech supply chain breaks", "type": "sector", "sector": "Technology", "magnitude": 0.08, "duration_days": 150},
        ],
    },
    {
        "id": "rate_hike",
        "name": "Interest Rate Shock (+300bps)",
        "category": "economic",
        "shocks": [
            {"name": "Real estate contraction", "type": "sector", "sector": "Real Estate", "magnitude": 0.20, "duration_days": 365},
            {"name": "Finance sector stress", "type": "sector", "sector": "Finance", "magnitude": 0.10, "duration_days": 270},
        ],
    },
    {
        "id": "pandemic",
        "name": "Pandemic Lockdown (COVID-like)",
        "category": "health",
        "shocks": [
            {"name": "Healthcare overwhelm", "type": "sector", "sector": "Healthcare", "magnitude": 0.25, "duration_days": 180},
            {"name": "Transport halt", "type": "sector", "sector": "Transportation", "magnitude": 0.30, "duration_days": 90},
            {"name": "Government emergency response", "type": "sector", "sector": "Government", "magnitude": 0.15, "duration_days": 180},
            {"name": "Education disruption", "type": "sector", "sector": "Education", "magnitude": 0.20, "duration_days": 120},
        ],
    },
]


# ── Routes ───────────────────────────────────────────────────────────

@router.get("/v1/scenarios/library")
async def get_shock_library(
    user: dict = Depends(require_role("viewer")),
):
    """Browse the preset shock library."""
    return {"library": SHOCK_LIBRARY, "total": len(SHOCK_LIBRARY)}


@router.post("/v1/projects/{pid}/scenarios", response_model=ScenarioResponse)
async def create_scenario(
    pid: str,
    req: CreateScenarioRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    # Verify project
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    scenario = Scenario(
        id=uuid.uuid4(),
        project_id=uuid.UUID(pid),
        tenant_id=uuid.UUID(user["tenant_id"]),
        name=req.name,
        description=req.description,
        shocks=[s.model_dump() for s in req.shocks],
        created_by=uuid.UUID(user["sub"]),
    )
    db.add(scenario)
    await db.flush()

    await log_action(
        db, action="scenario.create", tenant_id=user["tenant_id"],
        user_id=user["sub"], resource_type="scenario", resource_id=str(scenario.id),
    )

    return ScenarioResponse(
        id=str(scenario.id), project_id=pid, name=scenario.name,
        description=scenario.description, shocks=scenario.shocks,
        created_at=scenario.created_at.isoformat(),
    )


@router.get("/v1/projects/{pid}/scenarios")
async def list_scenarios(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Scenario).where(
            Scenario.project_id == uuid.UUID(pid),
            Scenario.tenant_id == uuid.UUID(user["tenant_id"]),
        ).order_by(Scenario.created_at.desc())
    )
    scenarios = result.scalars().all()
    return {
        "scenarios": [
            ScenarioResponse(
                id=str(s.id), project_id=pid, name=s.name,
                description=s.description, shocks=s.shocks,
                created_at=s.created_at.isoformat(),
            )
            for s in scenarios
        ],
        "total": len(scenarios),
    }


@router.delete("/v1/projects/{pid}/scenarios/{sid}")
async def delete_scenario(
    pid: str, sid: str,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Scenario).where(
            Scenario.id == uuid.UUID(sid),
            Scenario.project_id == uuid.UUID(pid),
            Scenario.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    scenario = result.scalar_one_or_none()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    await db.delete(scenario)
    await db.flush()
    return {"status": "deleted", "scenario_id": sid}
