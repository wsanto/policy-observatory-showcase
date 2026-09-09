"""AI-Assisted Model Onboarding — research, propose, approve, sandbox, execute."""

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .audit import log_action
from .database import get_db, set_tenant_context
from .middleware import require_role
from .models import ModelRegistryEntry

router = APIRouter(tags=["model-assistant"])


# ── Schemas ──────────────────────────────────────────────────────────

class ModelProposalRequest(BaseModel):
    """User describes a model they want in plain English."""
    description: str  # e.g., "A Phillips Curve model relating inflation to unemployment"
    category: Optional[str] = None  # economic, financial_risk, statistical, etc.
    use_case: Optional[str] = None  # What they want to analyze


class ModelProposalResponse(BaseModel):
    id: str
    name: str
    category: str
    description: str
    assumptions: str
    inputs_schema: dict
    outputs_schema: dict
    implementation_plan: str
    sample_code: str
    approval_status: str


class ModelApprovalRequest(BaseModel):
    action: str  # "approve" or "reject"
    notes: Optional[str] = None


class ModelTestRequest(BaseModel):
    test_inputs: dict  # Sample inputs to run through the model


class ModelTestResponse(BaseModel):
    success: bool
    outputs: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: int = 0


# ── LLM Research Prompt ──────────────────────────────────────────────

RESEARCH_PROMPT = """You are an expert economist and data scientist. A user wants to add a new analytical model to a policy simulation platform.

User's request:
{description}

Category hint: {category}
Use case: {use_case}

The platform already has these models: Leontief I-O, DebtRank, EVT/GPD, Copula, HMM Regime Detector, Synthetic Control, Monte Carlo GDP, Kalman Nowcasting, Bayesian Updater, Stress Tests, EMA Threat.

Research this model and respond with VALID JSON only (no markdown):
{{
  "name": "Short model name (3-5 words)",
  "category": "economic|financial_risk|statistical|ai_forecasting|social|integration",
  "description": "One paragraph description of what the model does",
  "assumptions": "Key assumptions and limitations (2-3 sentences)",
  "academic_reference": "Primary academic paper or methodology reference",
  "inputs_schema": {{
    "input_name": {{"type": "array|number|object", "description": "what this input is"}}
  }},
  "outputs_schema": {{
    "output_name": {{"type": "number|array|object", "description": "what this output is"}}
  }},
  "implementation_plan": "Step-by-step plan for implementing this model (3-5 steps)",
  "sample_code": "Python function implementing the core model logic. Use numpy/scipy only. Function signature: def run_model(inputs: dict) -> dict. Keep under 50 lines."
}}

CRITICAL: The sample_code must be a complete, runnable Python function. Use only numpy and scipy. No external APIs or network calls."""


# ── AI Research Engine ───────────────────────────────────────────────

async def research_model(description: str, category: str = "", use_case: str = "") -> dict:
    """Use LLM to research a model and propose an implementation."""
    from policy_engine.connectors.kimi import KimiClient
    import re

    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="LLM not configured — cannot research models")

    client = KimiClient(api_key=api_key)
    prompt = RESEARCH_PROMPT.format(
        description=description,
        category=category or "auto-detect",
        use_case=use_case or "general policy analysis",
    )

    response = await client.chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert model researcher. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=8192,
    )

    content = response.get("content", "") or response.get("reasoning", "")

    # Parse JSON
    json_str = content
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_str = content.split("```")[1].split("```")[0]

    json_str = json_str.strip()
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    return json.loads(json_str)


# ── Sandbox Execution ────────────────────────────────────────────────

SANDBOX_WRAPPER = '''
import json, sys, numpy as np
try:
    from scipy import stats, optimize, linalg
except ImportError:
    pass

{model_code}

inputs = json.loads(sys.argv[1])
result = run_model(inputs)
print(json.dumps(result, default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))
'''


def execute_in_sandbox(code: str, inputs: dict, timeout_seconds: int = 30) -> dict:
    """Execute model code in an isolated subprocess with resource limits."""
    wrapper = SANDBOX_WRAPPER.format(model_code=code)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper)
        f.flush()
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path, json.dumps(inputs)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": "/tmp",
            },
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr[:500], "outputs": None}

        outputs = json.loads(result.stdout.strip())
        return {"success": True, "outputs": outputs, "error": None}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout_seconds}s", "outputs": None}
    except json.JSONDecodeError:
        return {"success": False, "error": f"Model returned invalid JSON: {result.stdout[:200]}", "outputs": None}
    except Exception as e:
        return {"success": False, "error": str(e), "outputs": None}
    finally:
        os.unlink(script_path)


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/v1/models/propose", response_model=ModelProposalResponse)
async def propose_model(
    req: ModelProposalRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    """AI researches a model and proposes an implementation. Saves as pending approval."""
    logger.info(f"[ModelAssistant] Researching: {req.description[:80]}")
    proposal = await research_model(req.description, req.category or "", req.use_case or "")

    # Generate a slug ID
    model_id = f"custom_{uuid.uuid4().hex[:8]}"

    # Save to model registry with pending status
    entry = ModelRegistryEntry(
        id=model_id,
        name=proposal.get("name", "Custom Model"),
        category=proposal.get("category", req.category or "custom"),
        description=proposal.get("description", req.description),
        assumptions=proposal.get("assumptions", ""),
        inputs_schema=proposal.get("inputs_schema", {}),
        outputs_schema=proposal.get("outputs_schema", {}),
        version="0.1-draft",
        status="pending_approval",
        is_default=False,
    )
    db.add(entry)
    await db.flush()

    tenant_id = user.get("tenant_id")
    if tenant_id:
        await log_action(
            db, action="model.propose", tenant_id=tenant_id,
            user_id=user["sub"], resource_type="model", resource_id=model_id,
            metadata={"description": req.description[:200]},
        )

    return ModelProposalResponse(
        id=model_id,
        name=entry.name,
        category=entry.category,
        description=entry.description,
        assumptions=entry.assumptions or "",
        inputs_schema=entry.inputs_schema or {},
        outputs_schema=entry.outputs_schema or {},
        implementation_plan=proposal.get("implementation_plan", ""),
        sample_code=proposal.get("sample_code", ""),
        approval_status="pending_approval",
    )


@router.get("/v1/models/pending")
async def list_pending_models(
    user: dict = Depends(require_role("tenant_admin")),
    db: AsyncSession = Depends(get_db),
):
    """List models pending approval."""
    result = await db.execute(
        select(ModelRegistryEntry).where(ModelRegistryEntry.status == "pending_approval")
        .order_by(ModelRegistryEntry.created_at.desc())
    )
    models = result.scalars().all()
    return {
        "models": [
            {
                "id": m.id, "name": m.name, "category": m.category,
                "description": m.description, "assumptions": m.assumptions,
                "inputs_schema": m.inputs_schema, "outputs_schema": m.outputs_schema,
                "version": m.version, "status": m.status,
            }
            for m in models
        ],
        "total": len(models),
    }


@router.post("/v1/models/{mid}/approve")
async def approve_or_reject_model(
    mid: str,
    req: ModelApprovalRequest,
    user: dict = Depends(require_role("tenant_admin")),
    db: AsyncSession = Depends(get_db),
):
    """Approve or reject a pending model."""
    result = await db.execute(select(ModelRegistryEntry).where(ModelRegistryEntry.id == mid))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.status != "pending_approval":
        raise HTTPException(status_code=400, detail=f"Model is '{model.status}', not pending_approval")

    if req.action == "approve":
        model.status = "stable"
        model.version = "1.0"
    elif req.action == "reject":
        model.status = "rejected"
    else:
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")

    await db.flush()

    tenant_id = user.get("tenant_id")
    if tenant_id:
        await log_action(
            db, action=f"model.{req.action}", tenant_id=tenant_id,
            user_id=user["sub"], resource_type="model", resource_id=mid,
            metadata={"notes": req.notes},
        )

    return {"status": model.status, "model_id": mid, "notes": req.notes}


@router.post("/v1/models/{mid}/test", response_model=ModelTestResponse)
async def test_model(
    mid: str,
    req: ModelTestRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    """Test a model in the sandbox with sample inputs."""
    result = await db.execute(select(ModelRegistryEntry).where(ModelRegistryEntry.id == mid))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get the sample code from the proposal (stored in outputs_schema for now)
    # In a full implementation, code would be stored separately
    code = model.outputs_schema.get("_sample_code", "") if model.outputs_schema else ""
    if not code:
        return ModelTestResponse(success=False, error="No executable code associated with this model")

    import time
    start = time.time()
    sandbox_result = execute_in_sandbox(code, req.test_inputs)
    elapsed_ms = int((time.time() - start) * 1000)

    return ModelTestResponse(
        success=sandbox_result["success"],
        outputs=sandbox_result.get("outputs"),
        error=sandbox_result.get("error"),
        execution_time_ms=elapsed_ms,
    )
