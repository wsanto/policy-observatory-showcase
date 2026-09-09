"""Project CRUD, source management, and policy model routes."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio

from .audit import get_client_ip, log_action
from .database import get_db, set_tenant_context
from .document_processor import process_source
from .middleware import get_current_user, require_role
from .models import PolicyModel, Project, Run, Source
from .storage import upload_file

router = APIRouter(tags=["projects"])


# ── Schemas ──────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    industry_tags: list[str] = []
    geography: Optional[str] = None


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    industry_tags: Optional[list[str]] = None
    geography: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    industry_tags: list[str]
    geography: Optional[str]
    status: str
    source_count: int = 0
    run_count: int = 0
    created_at: str


class AddLinkRequest(BaseModel):
    url: str
    title: str


class SourceResponse(BaseModel):
    id: str
    source_type: str
    title: str
    url: Optional[str]
    extraction_status: str
    created_at: str


class PolicyModelResponse(BaseModel):
    objectives: list
    kpis: list
    sectors: list
    stakeholder_config: dict
    version: int


class UpdatePolicyModelRequest(BaseModel):
    objectives: Optional[list] = None
    kpis: Optional[list] = None
    sectors: Optional[list] = None
    stakeholder_config: Optional[dict] = None


# ── Project CRUD ─────────────────────────────────────────────────────

@router.post("/v1/projects", response_model=ProjectResponse)
async def create_project(
    req: CreateProjectRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    project = Project(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(user["tenant_id"]),
        name=req.name,
        description=req.description,
        industry_tags=req.industry_tags,
        geography=req.geography,
        created_by=uuid.UUID(user["sub"]),
    )
    db.add(project)
    await db.flush()

    await log_action(
        db, action="project.create", tenant_id=user["tenant_id"],
        user_id=user["sub"], resource_type="project", resource_id=str(project.id),
    )

    return ProjectResponse(
        id=str(project.id), name=project.name, description=project.description,
        industry_tags=project.industry_tags or [], geography=project.geography,
        status=project.status, created_at=project.created_at.isoformat(),
    )


@router.get("/v1/projects")
async def list_projects(
    status: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    q = select(Project).where(Project.tenant_id == uuid.UUID(user["tenant_id"]))
    if status:
        q = q.where(Project.status == status)
    if tag:
        q = q.where(Project.industry_tags.any(tag))
    q = q.order_by(Project.updated_at.desc())
    result = await db.execute(q)
    projects = result.scalars().all()

    items = []
    for p in projects:
        src_count = await db.execute(
            select(func.count()).select_from(Source).where(Source.project_id == p.id)
        )
        run_count = await db.execute(
            select(func.count()).select_from(Run).where(Run.project_id == p.id)
        )
        items.append(ProjectResponse(
            id=str(p.id), name=p.name, description=p.description,
            industry_tags=p.industry_tags or [], geography=p.geography,
            status=p.status, source_count=src_count.scalar() or 0,
            run_count=run_count.scalar() or 0,
            created_at=p.created_at.isoformat(),
        ))
    return {"projects": items, "total": len(items)}


@router.get("/v1/projects/{pid}", response_model=ProjectResponse)
async def get_project(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    src_count = await db.execute(
        select(func.count()).select_from(Source).where(Source.project_id == project.id)
    )
    run_count = await db.execute(
        select(func.count()).select_from(Run).where(Run.project_id == project.id)
    )

    return ProjectResponse(
        id=str(project.id), name=project.name, description=project.description,
        industry_tags=project.industry_tags or [], geography=project.geography,
        status=project.status, source_count=src_count.scalar() or 0,
        run_count=run_count.scalar() or 0,
        created_at=project.created_at.isoformat(),
    )


@router.put("/v1/projects/{pid}", response_model=ProjectResponse)
async def update_project(
    pid: str,
    req: UpdateProjectRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if req.name is not None:
        project.name = req.name
    if req.description is not None:
        project.description = req.description
    if req.industry_tags is not None:
        project.industry_tags = req.industry_tags
    if req.geography is not None:
        project.geography = req.geography
    await db.flush()

    return ProjectResponse(
        id=str(project.id), name=project.name, description=project.description,
        industry_tags=project.industry_tags or [], geography=project.geography,
        status=project.status, created_at=project.created_at.isoformat(),
    )


@router.delete("/v1/projects/{pid}")
async def archive_project(
    pid: str,
    user: dict = Depends(require_role("tenant_admin")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = "archived"
    await db.flush()

    await log_action(
        db, action="project.archive", tenant_id=user["tenant_id"],
        user_id=user["sub"], resource_type="project", resource_id=pid,
    )
    return {"status": "archived", "project_id": pid}


# ── Source management ────────────────────────────────────────────────

@router.post("/v1/projects/{pid}/sources/upload")
async def upload_source(
    pid: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    # Verify project exists and belongs to tenant
    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    storage_key, checksum = upload_file(
        tenant_id=user["tenant_id"],
        project_id=pid,
        category="sources",
        filename=file.filename or "upload",
        data=data,
        content_type=file.content_type or "application/octet-stream",
    )

    source = Source(
        id=uuid.uuid4(),
        project_id=uuid.UUID(pid),
        tenant_id=uuid.UUID(user["tenant_id"]),
        source_type="policy_document",
        title=file.filename or "Uploaded document",
        storage_key=storage_key,
        checksum=checksum,
        extraction_status="pending",
    )
    db.add(source)
    await db.flush()

    # Launch background document processing (extract objectives from PDF)
    asyncio.create_task(
        process_source(user["tenant_id"], str(source.id), pid)
    )

    return SourceResponse(
        id=str(source.id), source_type=source.source_type, title=source.title,
        url=None, extraction_status="processing",
        created_at=source.created_at.isoformat(),
    )


@router.post("/v1/projects/{pid}/sources/link")
async def add_link_source(
    pid: str,
    req: AddLinkRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    result = await db.execute(
        select(Project).where(
            Project.id == uuid.UUID(pid),
            Project.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    source = Source(
        id=uuid.uuid4(),
        project_id=uuid.UUID(pid),
        tenant_id=uuid.UUID(user["tenant_id"]),
        source_type="web_link",
        title=req.title,
        url=req.url,
        extraction_status="pending",
    )
    db.add(source)
    await db.flush()

    # Launch background document processing for ALL links (PDFs + HTML pages)
    asyncio.create_task(
        process_source(user["tenant_id"], str(source.id), pid)
    )

    return SourceResponse(
        id=str(source.id), source_type=source.source_type, title=source.title,
        url=source.url, extraction_status="processing",
        created_at=source.created_at.isoformat(),
    )


@router.get("/v1/projects/{pid}/sources")
async def list_sources(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Source).where(
            Source.project_id == uuid.UUID(pid),
            Source.tenant_id == uuid.UUID(user["tenant_id"]),
        ).order_by(Source.created_at.desc())
    )
    sources = result.scalars().all()
    return {
        "sources": [
            SourceResponse(
                id=str(s.id), source_type=s.source_type, title=s.title,
                url=s.url, extraction_status=s.extraction_status,
                created_at=s.created_at.isoformat(),
            )
            for s in sources
        ],
        "total": len(sources),
    }


@router.delete("/v1/projects/{pid}/sources/{sid}")
async def delete_source(
    pid: str, sid: str,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(Source).where(
            Source.id == uuid.UUID(sid),
            Source.project_id == uuid.UUID(pid),
            Source.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    await db.delete(source)
    await db.flush()
    return {"status": "deleted", "source_id": sid}


# ── Policy Model ─────────────────────────────────────────────────────

@router.get("/v1/projects/{pid}/policy-model")
async def get_policy_model(
    pid: str,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(PolicyModel).where(
            PolicyModel.project_id == uuid.UUID(pid),
            PolicyModel.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    pm = result.scalar_one_or_none()
    if not pm:
        return {"policy_model": None, "message": "No policy model extracted yet"}

    return PolicyModelResponse(
        objectives=pm.objectives, kpis=pm.kpis, sectors=pm.sectors,
        stakeholder_config=pm.stakeholder_config or {}, version=pm.version,
    )


@router.put("/v1/projects/{pid}/policy-model")
async def update_policy_model(
    pid: str,
    req: UpdatePolicyModelRequest,
    user: dict = Depends(require_role("analyst")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])
    result = await db.execute(
        select(PolicyModel).where(
            PolicyModel.project_id == uuid.UUID(pid),
            PolicyModel.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    pm = result.scalar_one_or_none()
    if not pm:
        # Create new policy model
        pm = PolicyModel(
            id=uuid.uuid4(),
            project_id=uuid.UUID(pid),
            tenant_id=uuid.UUID(user["tenant_id"]),
        )
        db.add(pm)

    if req.objectives is not None:
        pm.objectives = req.objectives
    if req.kpis is not None:
        pm.kpis = req.kpis
    if req.sectors is not None:
        pm.sectors = req.sectors
    if req.stakeholder_config is not None:
        pm.stakeholder_config = req.stakeholder_config
    pm.version = (pm.version or 0) + 1
    await db.flush()

    return PolicyModelResponse(
        objectives=pm.objectives, kpis=pm.kpis, sectors=pm.sectors,
        stakeholder_config=pm.stakeholder_config or {}, version=pm.version,
    )
