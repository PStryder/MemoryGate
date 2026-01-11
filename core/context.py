"""
Request-scoped context objects for core services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthContext:
    user_id: Optional[int] = None
    tenant_id: Optional[str] = None
    actor: Optional[str] = None


@dataclass(frozen=True)
class RequestContext:
    auth: AuthContext
    request_id: Optional[str] = None
    source: Optional[str] = None
