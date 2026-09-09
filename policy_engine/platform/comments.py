"""Threaded comments on runs, reports, and artifacts."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, select, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, set_tenant_context
from .middleware import require_role
from .models import Base, Run

router = APIRouter(tags=["comments"])


# ── Comment model (add to DB via next migration) ─────────────────────
# For now we store comments in a simple in-memory dict keyed by run_id.
# This will be moved to Postgres in a proper migration.

_comments: dict[str, list[dict]] = {}


class CreateCommentRequest(BaseModel):
    text: str
    section: Optional[str] = None  # e.g., "gdp_projections", "scorecard_OBJ1", "executive_summary"
    parent_id: Optional[str] = None  # for threaded replies


class CommentResponse(BaseModel):
    id: str
    run_id: str
    user_id: str
    user_name: str
    text: str
    section: Optional[str]
    parent_id: Optional[str]
    created_at: str


@router.post("/v1/runs/{rid}/comments", response_model=CommentResponse)
async def add_comment(
    rid: str,
    req: CreateCommentRequest,
    user: dict = Depends(require_role("viewer")),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, user["tenant_id"])

    # Verify run belongs to tenant
    result = await db.execute(
        select(Run).where(
            Run.id == uuid.UUID(rid),
            Run.tenant_id == uuid.UUID(user["tenant_id"]),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Run not found")

    comment = {
        "id": str(uuid.uuid4()),
        "run_id": rid,
        "user_id": user["sub"],
        "user_name": user.get("name", "User"),
        "text": req.text,
        "section": req.section,
        "parent_id": req.parent_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if rid not in _comments:
        _comments[rid] = []
    _comments[rid].append(comment)

    return CommentResponse(**comment)


@router.get("/v1/runs/{rid}/comments")
async def list_comments(
    rid: str,
    section: Optional[str] = Query(None),
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
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Run not found")

    comments = _comments.get(rid, [])
    if section:
        comments = [c for c in comments if c.get("section") == section]

    return {"comments": comments, "total": len(comments)}


@router.delete("/v1/runs/{rid}/comments/{cid}")
async def delete_comment(
    rid: str, cid: str,
    user: dict = Depends(require_role("analyst")),
):
    comments = _comments.get(rid, [])
    for i, c in enumerate(comments):
        if c["id"] == cid and (c["user_id"] == user["sub"] or user.get("role") in ("tenant_admin", "platform_admin")):
            comments.pop(i)
            return {"status": "deleted", "comment_id": cid}
    raise HTTPException(status_code=404, detail="Comment not found")
