"""Async Postgres connection, session factory, and RLS tenant context."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

_raw_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://pse:pse_dev@localhost:5432/pse")
# Normalize URL: Fly.io uses postgres://, we need postgresql+asyncpg://
DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://").replace("postgresql://", "postgresql+asyncpg://")
if "asyncpg+asyncpg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("asyncpg+asyncpg", "asyncpg")

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session, commits on success, rolls back on error."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def set_tenant_context(session: AsyncSession, tenant_id: str) -> None:
    """Set the Postgres session variable used by RLS policies."""
    await session.execute(text(f"SET app.current_tenant_id = '{tenant_id}'"))


@asynccontextmanager
async def tenant_session(tenant_id: str) -> AsyncGenerator[AsyncSession, None]:
    """Context manager that opens a session with tenant RLS already set."""
    async with async_session_factory() as session:
        await set_tenant_context(session, tenant_id)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
