"""Model registry browse and detail routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .middleware import get_current_user
from .models import ModelRegistryEntry

router = APIRouter(tags=["models"])


class ModelResponse(BaseModel):
    id: str
    name: str
    category: str
    description: str
    assumptions: Optional[str]
    version: str
    status: str
    is_default: bool


class ModelDetailResponse(ModelResponse):
    inputs_schema: Optional[dict]
    outputs_schema: Optional[dict]


@router.get("/v1/models")
async def list_models(
    category: Optional[str] = Query(None, description="Filter by category: economic, financial_risk, statistical, ai_forecasting, social, integration"),
    status: Optional[str] = Query(None, description="Filter by status: stable, beta, deprecated"),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = select(ModelRegistryEntry)
    if category:
        q = q.where(ModelRegistryEntry.category == category)
    if status:
        q = q.where(ModelRegistryEntry.status == status)
    q = q.order_by(ModelRegistryEntry.category, ModelRegistryEntry.name)

    result = await db.execute(q)
    models = result.scalars().all()

    return {
        "models": [
            ModelResponse(
                id=m.id, name=m.name, category=m.category,
                description=m.description, assumptions=m.assumptions,
                version=m.version, status=m.status, is_default=m.is_default,
            )
            for m in models
        ],
        "total": len(models),
    }


@router.get("/v1/models/{mid}", response_model=ModelDetailResponse)
async def get_model(
    mid: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ModelRegistryEntry).where(ModelRegistryEntry.id == mid)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelDetailResponse(
        id=model.id, name=model.name, category=model.category,
        description=model.description, assumptions=model.assumptions,
        version=model.version, status=model.status, is_default=model.is_default,
        inputs_schema=model.inputs_schema, outputs_schema=model.outputs_schema,
    )
