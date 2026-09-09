"""Background simulation orchestrator — wraps the policy study engine for project-scoped runs."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger
from sqlalchemy import select, update

from .database import tenant_session
from .models import Artifact, Run
from .storage import upload_bytes

# Run mode → StudyConfig parameter presets
RUN_PROFILES = {
    "smoke": {
        "calibration_ticks": 2,
        "primary_ticks": 5,
        "ablation_ticks": 0,
        "tick_interval_s": 3.0,
        "monte_carlo_iterations": 1000,
        "agent_count": 100,
        "simulation_steps": 20,
    },
    "standard": {
        "calibration_ticks": 5,
        "primary_ticks": 50,
        "ablation_ticks": 10,
        "tick_interval_s": 2.0,
        "monte_carlo_iterations": 5000,
        "agent_count": 500,
        "simulation_steps": 50,
    },
    "publication": {
        "calibration_ticks": 10,
        "primary_ticks": 925,
        "ablation_ticks": 506,
        "tick_interval_s": 2.0,
        "monte_carlo_iterations": 10000,
        "agent_count": 1500,
        "simulation_steps": 100,
    },
}

# Active runs tracked in memory for cancellation
_active_runs: Dict[str, asyncio.Task] = {}


async def _update_run_status(
    tenant_id: str,
    run_id: str,
    status: str,
    progress: Optional[dict] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    error: Optional[str] = None,
):
    """Update run status in Postgres."""
    async with tenant_session(tenant_id) as db:
        values: Dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if started_at is not None:
            values["started_at"] = started_at
        if completed_at is not None:
            values["completed_at"] = completed_at
        if error:
            values["progress"] = {**(progress or {}), "error": error}

        await db.execute(
            update(Run).where(Run.id == uuid.UUID(run_id)).values(**values)
        )


async def _save_artifact(
    tenant_id: str,
    run_id: str,
    artifact_type: str,
    data: Any,
) -> str:
    """Serialize artifact to JSON, upload to S3, create DB record."""
    json_bytes = json.dumps(data, indent=2, default=str).encode()

    # Find project_id for storage path
    async with tenant_session(tenant_id) as db:
        result = await db.execute(
            select(Run.project_id).where(Run.id == uuid.UUID(run_id))
        )
        project_id = str(result.scalar_one())

    storage_key = f"tenant/{tenant_id}/project/{project_id}/artifacts/{run_id}/{artifact_type}.json"
    upload_bytes(storage_key, json_bytes, content_type="application/json")

    async with tenant_session(tenant_id) as db:
        artifact = Artifact(
            id=uuid.uuid4(),
            run_id=uuid.UUID(run_id),
            tenant_id=uuid.UUID(tenant_id),
            artifact_type=artifact_type,
            storage_key=storage_key,
        )
        db.add(artifact)

    return storage_key


async def run_simulation(
    run_id: str,
    tenant_id: str,
    run_mode: str,
    model_selections: list[str],
    config_overrides: Optional[dict] = None,
):
    """Execute a simulation run as a background task."""
    try:
        await _update_run_status(
            tenant_id, run_id, "running",
            progress={"phase": 0, "phase_name": "Initializing", "percent": 0},
            started_at=datetime.now(timezone.utc),
        )

        # Build config from profile + overrides
        profile = RUN_PROFILES.get(run_mode, RUN_PROFILES["smoke"])
        config_params = {**profile, **(config_overrides or {})}

        # Load project policy model from database
        from policy_engine.platform.models import PolicyModel, Project
        async with tenant_session(tenant_id) as db:
            run_result = await db.execute(
                select(Run.project_id).where(Run.id == uuid.UUID(run_id))
            )
            project_id = str(run_result.scalar_one())

            proj_result = await db.execute(
                select(Project).where(Project.id == uuid.UUID(project_id))
            )
            project = proj_result.scalar_one_or_none()

            pm_result = await db.execute(
                select(PolicyModel).where(PolicyModel.project_id == uuid.UUID(project_id))
            )
            policy_model = pm_result.scalar_one_or_none()

        # Build project-specific config
        from policy_engine.platform.project_study import ProjectConfig, ProjectStudy

        objectives = (policy_model.objectives if policy_model else None) or [
            {"id": f"OBJ{i+1}", "title": f"Objective {i+1}", "short": f"OBJ{i+1}", "description": ""}
            for i in range(3)
        ]

        project_config = ProjectConfig(
            project_name=project.name if project else "Policy Simulation",
            geography=project.geography if project else "Global",
            objectives=objectives,
            kpis=(policy_model.kpis if policy_model else None) or [],
            sectors=(policy_model.sectors if policy_model else None) or [],
            **config_params,
        )

        # Phase progress callback
        async def on_event(event: dict):
            if event.get("subtype") == "phase_change":
                phase = event.get("data", {}).get("phase", 0)
                phase_name = event.get("data", {}).get("phase_name", "")
                pct = min(int((phase / 5) * 100), 100)
                await _update_run_status(
                    tenant_id, run_id, "running",
                    progress={"phase": phase, "phase_name": phase_name, "percent": pct},
                )

        study = ProjectStudy(config=project_config, event_callback=on_event)

        logger.info(f"[Orchestrator] Starting run {run_id} (mode={run_mode}, project={project_config.project_name}, geo={project_config.geography}, objectives={len(objectives)})")
        await study.run()

        # Persist artifacts (get_snapshot is sync, not async)
        snapshot = study.telemetry.get_snapshot()
        await _save_artifact(tenant_id, run_id, "snapshot", snapshot)
        await _save_artifact(tenant_id, run_id, "ticks", snapshot.get("ticks", []))
        await _save_artifact(tenant_id, run_id, "breakthroughs", snapshot.get("breakthroughs", []))
        await _save_artifact(tenant_id, run_id, "gdp_projections", snapshot.get("gdp_projections", []))

        if snapshot.get("impact_assessment"):
            await _save_artifact(tenant_id, run_id, "impact_assessment", snapshot["impact_assessment"])

        await _update_run_status(
            tenant_id, run_id, "completed",
            progress={"phase": 5, "phase_name": "Complete", "percent": 100},
            completed_at=datetime.now(timezone.utc),
        )
        logger.info(f"[Orchestrator] Run {run_id} completed successfully")

    except asyncio.CancelledError:
        await _update_run_status(tenant_id, run_id, "cancelled")
        logger.info(f"[Orchestrator] Run {run_id} cancelled")
    except Exception as e:
        logger.error(f"[Orchestrator] Run {run_id} failed: {e}")
        await _update_run_status(
            tenant_id, run_id, "failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc),
        )
    finally:
        _active_runs.pop(run_id, None)


def launch_run(run_id: str, tenant_id: str, run_mode: str, model_selections: list[str], config_overrides: Optional[dict] = None):
    """Launch a simulation as a background asyncio task. Returns immediately."""
    task = asyncio.create_task(
        run_simulation(run_id, tenant_id, run_mode, model_selections, config_overrides)
    )
    _active_runs[run_id] = task
    return task


def cancel_run(run_id: str) -> bool:
    """Cancel a running simulation. Returns True if found and cancelled."""
    task = _active_runs.get(run_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def is_run_active(run_id: str) -> bool:
    """Check if a run is currently executing."""
    task = _active_runs.get(run_id)
    return task is not None and not task.done()
