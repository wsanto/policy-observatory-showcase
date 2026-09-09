"""SQLAlchemy ORM models for the SaaS platform layer."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index, Integer, Numeric,
    String, Text, UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Tenants ──────────────────────────────────────────────────────────

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    slug = Column(Text, unique=True, nullable=False)
    plan = Column(Text, nullable=False, server_default="free")
    region = Column(Text, nullable=False, server_default="us-east-1")
    settings = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    memberships = relationship("Membership", back_populates="tenant")
    projects = relationship("Project", back_populates="tenant")


# ── Users ────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=True)
    name = Column(Text, nullable=False)
    mfa_secret = Column(Text, nullable=True)
    is_platform_admin = Column(Boolean, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    memberships = relationship("Membership", back_populates="user")


# ── Memberships (tenant + user + role) ───────────────────────────────

class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(Text, nullable=False, server_default="viewer")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    tenant = relationship("Tenant", back_populates="memberships")
    user = relationship("User", back_populates="memberships")


# ── Projects ─────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    industry_tags = Column(ARRAY(Text), server_default=text("'{}'::text[]"))
    geography = Column(Text, nullable=True)
    status = Column(Text, nullable=False, server_default="draft")
    settings = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    tenant = relationship("Tenant", back_populates="projects")
    sources = relationship("Source", back_populates="project")
    runs = relationship("Run", back_populates="project")


# ── Sources (uploaded docs, links, datasets) ─────────────────────────

class Source(Base):
    __tablename__ = "sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    source_type = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    storage_key = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    checksum = Column(Text, nullable=True)
    extraction_status = Column(Text, server_default="pending")
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    project = relationship("Project", back_populates="sources")


# ── Policy Model ─────────────────────────────────────────────────────

class PolicyModel(Base):
    __tablename__ = "policy_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), unique=True, nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    objectives = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    kpis = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    sectors = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    stakeholder_config = Column(JSONB, server_default=text("'{}'::jsonb"))
    version = Column(Integer, nullable=False, server_default="1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Simulation Runs ──────────────────────────────────────────────────

class Run(Base):
    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(Text, nullable=True)
    run_mode = Column(Text, nullable=False, server_default="smoke")
    status = Column(Text, nullable=False, server_default="queued")
    config_snapshot = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    model_selections = Column(ARRAY(Text), nullable=False, server_default=text("'{}'::text[]"))
    progress = Column(JSONB, server_default=text("'{}'::jsonb"))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    project = relationship("Project", back_populates="runs")


# ── Artifacts ────────────────────────────────────────────────────────

class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    artifact_type = Column(Text, nullable=False)
    storage_key = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, server_default="1")
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Scenarios + Shocks ───────────────────────────────────────────────

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    shocks = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Reports ──────────────────────────────────────────────────────────

class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    storage_key = Column(Text, nullable=False)
    sections = Column(JSONB, server_default=text("'[]'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Share Links ──────────────────────────────────────────────────────

class ShareLink(Base):
    __tablename__ = "share_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    token = Column(Text, unique=True, nullable=False)
    passcode_hash = Column(Text, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    access = Column(Text, nullable=False, server_default="view_only")
    access_count = Column(Integer, server_default="0")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Model Registry ───────────────────────────────────────────────────

class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    assumptions = Column(Text, nullable=True)
    inputs_schema = Column(JSONB, nullable=True)
    outputs_schema = Column(JSONB, nullable=True)
    version = Column(Text, nullable=False)
    status = Column(Text, nullable=False, server_default="stable")
    is_default = Column(Boolean, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


# ── Audit Log ────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    action = Column(Text, nullable=False)
    resource_type = Column(Text, nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    ip_address = Column(INET, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    __table_args__ = (Index("ix_audit_log_tenant_created", "tenant_id", "created_at"),)


# ── Usage Metering ───────────────────────────────────────────────────

class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    event_type = Column(Text, nullable=False)
    quantity = Column(Numeric, nullable=False, server_default="1")
    metadata_ = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    __table_args__ = (Index("ix_usage_events_tenant_type", "tenant_id", "event_type"),)
