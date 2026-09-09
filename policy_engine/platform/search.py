"""Full-text search across projects, objectives, and discoveries."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, set_tenant_context
from .middleware import require_role
from .models import Artifact, PolicyModel, Project, Run, Source

router = APIRouter(tags=["search"])


@router.get("/v1/search")
async def search(
    q: str = Query(..., min_length=2, description="Search query"),
    type: Optional[str] = Query(None, description="Filter: project, source, objective"),
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    """Search across projects, sources, objectives, and discoveries within the tenant."""
    await set_tenant_context(db, user["tenant_id"])
    tenant_id = uuid.UUID(user["tenant_id"])
    results = []
    query_lower = f"%{q.lower()}%"

    # Search projects
    if not type or type == "project":
        proj_result = await db.execute(
            select(Project).where(
                Project.tenant_id == tenant_id,
                or_(
                    Project.name.ilike(query_lower),
                    Project.description.ilike(query_lower),
                    Project.geography.ilike(query_lower),
                ),
            ).limit(limit)
        )
        for p in proj_result.scalars().all():
            results.append({
                "type": "project",
                "id": str(p.id),
                "title": p.name,
                "subtitle": p.description or p.geography or "",
                "url": f"/projects/{p.id}",
            })

    # Search sources
    if not type or type == "source":
        src_result = await db.execute(
            select(Source).where(
                Source.tenant_id == tenant_id,
                or_(
                    Source.title.ilike(query_lower),
                    Source.url.ilike(query_lower),
                ),
            ).limit(limit)
        )
        for s in src_result.scalars().all():
            results.append({
                "type": "source",
                "id": str(s.id),
                "title": s.title,
                "subtitle": s.url or s.source_type,
                "url": f"/projects/{s.project_id}",
            })

    # Search policy model objectives
    if not type or type == "objective":
        pm_result = await db.execute(
            select(PolicyModel).where(PolicyModel.tenant_id == tenant_id)
        )
        for pm in pm_result.scalars().all():
            for obj in (pm.objectives or []):
                title = obj.get("title", "")
                desc = obj.get("description", "")
                if q.lower() in title.lower() or q.lower() in desc.lower():
                    results.append({
                        "type": "objective",
                        "id": obj.get("id", ""),
                        "title": f"{obj.get('id', '')}: {title}",
                        "subtitle": desc[:100],
                        "url": f"/projects/{pm.project_id}",
                    })

    # Search run names
    if not type or type == "run":
        run_result = await db.execute(
            select(Run).where(
                Run.tenant_id == tenant_id,
                Run.name.ilike(query_lower),
            ).limit(limit)
        )
        for r in run_result.scalars().all():
            results.append({
                "type": "run",
                "id": str(r.id),
                "title": r.name,
                "subtitle": f"{r.run_mode} — {r.status}",
                "url": f"/runs/{r.id}",
            })

    return {"results": results[:limit], "total": len(results), "query": q}
