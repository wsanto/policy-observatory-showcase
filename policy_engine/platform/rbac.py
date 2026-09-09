"""Role-based access control utilities."""

from .middleware import ROLE_HIERARCHY


def check_permission(user: dict, required_role: str) -> bool:
    """Check if user has at least the required role level."""
    user_level = ROLE_HIERARCHY.get(user.get("role", "viewer"), 0)
    required_level = ROLE_HIERARCHY.get(required_role, 0)
    return user_level >= required_level


def is_tenant_admin(user: dict) -> bool:
    return check_permission(user, "tenant_admin")


def is_analyst(user: dict) -> bool:
    return check_permission(user, "analyst")


def is_platform_admin(user: dict) -> bool:
    return check_permission(user, "platform_admin")
