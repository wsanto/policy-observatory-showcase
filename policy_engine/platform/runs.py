"""Simulation run lifecycle routes."""

import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .audit import log_action
from .database import get_db, set_tenant_context
from .middleware import get_current_user, require_role
from .models import Artifact, Project, Run
from .orchestrator import cancel_run, is_run_active, launch_run
from .storage import download_file, get_presigned_url

router = APIRouter(tags=["runs"])


# ── Schemas ──────────────────────────────────────────────────────────

class CreateRunRequest(BaseModel):
    name: Optional[str] = None
    run_mode: str = "smoke"  # smoke, standard, publication
    model_selections: list[str] = []
    config_overrides: Optional[dict] = None


class RunResponse(BaseModel):
    id: str
    project_id: str
    name: Optional[str]
    run_mode: str
    status: str
    progress: dict
    model_selections: list[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: str


class ArtifactResponse(BaseModel):
    id: str
    artifact_type: str
    storage_key: str
    version: int
    created_at: str


# ── Run CRUD ─────────────────────────────────────────────────────────

@router.post("/v1/projects/{pid}/runs", response_model=RunResponse)
async def create_run(
    pid: str,
    req: CreateRunRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    # Verify project exists
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if req.run_mode not in ("smoke", "standard", "publication"):
        raise HTTPException(status_code=400, detail="Invalid run_mode")

    # If no models selected, use all defaults
    model_selections = req.model_selections or []

    run = Run(
        id=uuid.uuid4(),
        project_id=uuid.UUID(pid),
        tenant_id=uuid.UUID(user["tenant_id"]),
        name=req.name or f"{project.name} — {req.run_mode}",
        run_mode=req.run_mode,
        status="queued",
        config_snapshot={
            "run_mode": req.run_mode,
            "model_selections": model_selections,
            "config_overrides": req.config_overrides or {},
        },
        model_selections=model_selections,
        progress={"phase": 0, "phase_name": "Queued", "percent": 0},
        created_by=uuid.UUID(user["sub"]),
    )
    db.add(run)
    await db.flush()

    await log_action(
        db, action="run.create", tenant_id=user["tenant_id"],
        user_id=user["sub"], resource_type="run", resource_id=str(run.id),
        metadata={"run_mode": req.run_mode, "project_id": pid},
    )

    # Launch background task
    launch_run(
        run_id=str(run.id),
        tenant_id=user["tenant_id"],
        run_mode=req.run_mode,
        model_selections=model_selections,
        config_overrides=req.config_overrides,
    )

    return RunResponse(
        id=str(run.id), project_id=pid, name=run.name,
        run_mode=run.run_mode, status=run.status,
        progress=run.progress or {}, model_selections=run.model_selections or [],
        started_at=None, completed_at=None,
        created_at=run.created_at.isoformat(),
    )


@router.get("/v1/projects/{pid}/runs")
async def list_runs(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Run).where(
            Run.project_id == uuid.UUID(pid),
            Run.tenant_id == uuid.UUID(user["tenant_id"]),
        ).order_by(Run.created_at.desc())
    )
    runs = result.scalars().all()
    return {
        "runs": [
            RunResponse(
                id=str(r.id), project_id=pid, name=r.name,
                run_mode=r.run_mode, status=r.status,
                progress=r.progress or {}, model_selections=r.model_selections or [],
                started_at=r.started_at.isoformat() if r.started_at else None,
                completed_at=r.completed_at.isoformat() if r.completed_at else None,
                created_at=r.created_at.isoformat(),
            )
            for r in runs
        ],
        "total": len(runs),
    }


@router.get("/v1/runs/{rid}", response_model=RunResponse)
async def get_run(
    rid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Run).where(
            Run.id == uuid.UUID(rid),
            Run.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunResponse(
        id=str(run.id), project_id=str(run.project_id), name=run.name,
        run_mode=run.run_mode, status=run.status,
        progress=run.progress or {}, model_selections=run.model_selections or [],
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        created_at=run.created_at.isoformat(),
    )


@router.post("/v1/runs/{rid}/cancel")
async def cancel_simulation(
    rid: str,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Run).where(
            Run.id == uuid.UUID(rid),
            Run.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in ("queued", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel run in '{run.status}' state")

    cancelled = cancel_run(rid)
    if not cancelled:
        run.status = "cancelled"
        await db.flush()

    await log_action(
        db, action="run.cancel", tenant_id=user["tenant_id"],
        user_id=user["sub"], resource_type="run", resource_id=rid,
    )

    return {"status": "cancelled", "run_id": rid}


@router.get("/v1/runs/{rid}/artifacts")
async def list_artifacts(
    rid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    # Verify run belongs to tenant
    run_result = await db.execute(
        select(Run).where(
            Run.id == uuid.UUID(rid),
            Run.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    if not run_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Run not found")

    result = await db.execute(
        select(Artifact).where(
            Artifact.run_id == uuid.UUID(rid),
            Artifact.tenant_id == uuid.UUID(user["tenant_id"]),
        ).order_by(Artifact.created_at)
    )
    artifacts = result.scalars().all()
    return {
        "artifacts": [
            ArtifactResponse(
                id=str(a.id), artifact_type=a.artifact_type,
                storage_key=a.storage_key, version=a.version,
                created_at=a.created_at.isoformat(),
            )
            for a in artifacts
        ],
        "total": len(artifacts),
    }


@router.get("/v1/runs/{rid}/artifacts/{artifact_type}")
async def get_artifact_data(
    rid: str,
    artifact_type: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    """Download artifact data as JSON."""
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Artifact).where(
            Artifact.run_id == uuid.UUID(rid),
            Artifact.tenant_id == uuid.UUID(user["tenant_id"]),
            Artifact.artifact_type == artifact_type,
        )
    )
    artifact = result.scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    data = download_file(artifact.storage_key)
    return json.loads(data)
