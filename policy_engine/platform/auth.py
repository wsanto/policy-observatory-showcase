"""JWT-based authentication: registration, login, token management."""

import re
import uuid

import bcrypt as _bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .middleware import get_current_user
from .models import Membership, Tenant, User
from .tokens import create_access_token, create_refresh_token, decode_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


# ── Password hashing ────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode(), hashed.encode())


# ── Request / response schemas ───────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    tenant_name: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    role: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    tenant_id: str
    tenant_name: str
    role: str


# ── Slug generation ──────────────────────────────────────────────────

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{uuid.uuid4().hex[:6]}"


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if email already exists
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create user
    user = User(
        id=uuid.uuid4(),
        email=req.email,
        password_hash=hash_password(req.password),
        name=req.name,
    )
    db.add(user)

    # Create tenant
    tenant = Tenant(
        id=uuid.uuid4(),
        name=req.tenant_name,
        slug=_slugify(req.tenant_name),
    )
    db.add(tenant)

    # Create admin membership
    membership = Membership(
        tenant_id=tenant.id,
        user_id=user.id,
        role="tenant_admin",
    )
    db.add(membership)
    await db.flush()

    access = create_access_token(str(user.id), str(tenant.id), "tenant_admin")
    refresh = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user_id=str(user.id),
        tenant_id=str(tenant.id),
        role="tenant_admin",
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email.lower().strip()))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Get first membership (primary tenant)
    mem_result = await db.execute(
        select(Membership).where(Membership.user_id == user.id).limit(1)
    )
    membership = mem_result.scalar_one_or_none()
    if not membership:
        raise HTTPException(status_code=403, detail="No tenant membership found")

    access = create_access_token(str(user.id), str(membership.tenant_id), membership.role)
    refresh = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user_id=str(user.id),
        tenant_id=str(membership.tenant_id),
        role=membership.role,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload["sub"]
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    mem_result = await db.execute(
        select(Membership).where(Membership.user_id == user.id).limit(1)
    )
    membership = mem_result.scalar_one_or_none()
    if not membership:
        raise HTTPException(status_code=403, detail="No tenant membership found")

    access = create_access_token(str(user.id), str(membership.tenant_id), membership.role)
    new_refresh = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        user_id=str(user.id),
        tenant_id=str(membership.tenant_id),
        role=membership.role,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return current user info from JWT."""
    result = await db.execute(select(User).where(User.id == uuid.UUID(current_user["sub"])))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    mem_result = await db.execute(
        select(Membership).where(Membership.user_id == user.id).limit(1)
    )
    membership = mem_result.scalar_one_or_none()

    tenant_name = ""
    if membership:
        tenant_result = await db.execute(select(Tenant).where(Tenant.id == membership.tenant_id))
        tenant = tenant_result.scalar_one_or_none()
        tenant_name = tenant.name if tenant else ""

    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        tenant_id=current_user.get("tenant_id", ""),
        tenant_name=tenant_name,
        role=current_user.get("role", "viewer"),
    )
