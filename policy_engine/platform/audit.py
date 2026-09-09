"""Audit logging helper."""

import uuid
from typing import Any, Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from .models import AuditLog


def _safe_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    """Convert string to UUID, returning None if invalid."""
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return None


async def log_action(
    db: AsyncSession,
    *,
    action: str,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    """Insert an audit log entry."""
    entry = AuditLog(
        id=uuid.uuid4(),
        tenant_id=_safe_uuid(tenant_id),
        user_id=_safe_uuid(user_id),
        action=action,
        resource_type=resource_type,
        resource_id=_safe_uuid(resource_id),
        ip_address=ip_address,
    )
    if metadata:
        entry.metadata_ = metadata
    db.add(entry)


def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
