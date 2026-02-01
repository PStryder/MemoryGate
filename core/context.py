"""
Request-scoped context objects for core services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import contextvars

import core.config as config
from core.errors import ValidationIssue


@dataclass(frozen=True)
class TenantContext:
    tenant_id: Optional[str] = None
    actor_type: Optional[str] = None
    actor_id: Optional[str] = None
    scopes: tuple[str, ...] = ()

    @staticmethod
    def from_values(
        tenant_id: Optional[str],
        actor_type: Optional[str] = None,
        actor_id: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> "TenantContext":
        scope_values = tuple(scopes or ())
        return TenantContext(
            tenant_id=tenant_id,
            actor_type=actor_type,
            actor_id=actor_id,
            scopes=scope_values,
        )


@dataclass(frozen=True)
class AuthContext:
    user_id: Optional[int] = None
    tenant_id: Optional[str] = None
    actor: Optional[str] = None


@dataclass(frozen=True)
class RequestContext:
    auth: AuthContext
    tenant: Optional[TenantContext] = None
    request_id: Optional[str] = None
    source: Optional[str] = None
    agent_uuid: Optional[str] = None


_CURRENT_REQUEST_CONTEXT: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "memorygate_request_context",
    default=None,
)


def get_current_request_context() -> Optional["RequestContext"]:
    return _CURRENT_REQUEST_CONTEXT.get()


def set_current_request_context(context: Optional["RequestContext"]) -> contextvars.Token:
    return _CURRENT_REQUEST_CONTEXT.set(context)


def reset_current_request_context(token: contextvars.Token) -> None:
    _CURRENT_REQUEST_CONTEXT.reset(token)


def resolve_tenant_context(context: Optional["RequestContext"]) -> Optional[TenantContext]:
    if context is None:
        return None
    if context.tenant is not None:
        return context.tenant
    auth = context.auth
    if auth and auth.tenant_id:
        actor_id = str(auth.user_id) if auth.user_id is not None else None
        return TenantContext(
            tenant_id=auth.tenant_id,
            actor_type=auth.actor,
            actor_id=actor_id,
            scopes=(),
        )
    return None


def resolve_tenant_id(
    context: Optional["RequestContext"],
    *,
    required: Optional[bool] = None,
) -> Optional[str]:
    tenant_ctx = resolve_tenant_context(context)
    tenant_id = tenant_ctx.tenant_id if tenant_ctx else None
    enforce = required if required is not None else config.TENANCY_MODE == config.TENANCY_REQUIRED
    if not tenant_id:
        if config.TENANCY_MODE != config.TENANCY_REQUIRED:
            return config.DEFAULT_TENANT_ID
        if enforce:
            raise ValidationIssue(
                "tenant_id is required for this operation",
                field="tenant_id",
                error_type="required",
            )
        return None
    if enforce and config.TENANCY_MODE == config.TENANCY_REQUIRED and tenant_id == config.DEFAULT_TENANT_ID:
        raise ValidationIssue(
            "tenant_id is required for this operation",
            field="tenant_id",
            error_type="required",
        )
    return tenant_id


def require_tenant_id_value(tenant_id: Optional[str]) -> str:
    if config.TENANCY_MODE == config.TENANCY_REQUIRED and (
        not tenant_id or tenant_id == config.DEFAULT_TENANT_ID
    ):
        raise ValidationIssue(
            "tenant_id is required for this operation",
            field="tenant_id",
            error_type="required",
        )
    return tenant_id or config.DEFAULT_TENANT_ID


__all__ = [
    "TenantContext",
    "AuthContext",
    "RequestContext",
    "get_current_request_context",
    "set_current_request_context",
    "reset_current_request_context",
    "resolve_tenant_context",
    "resolve_tenant_id",
    "require_tenant_id_value",
]
