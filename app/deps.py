"""
Dependency helpers for the standalone FastAPI app.
"""

from __future__ import annotations

from typing import Generator, Optional

from fastapi import Depends

from core.context import AuthContext, RequestContext
from core.db import DB
from app.auth import get_current_user


def get_db_session() -> Generator:
    if DB.SessionLocal is None:
        raise RuntimeError("Database not initialized - SessionLocal is None")
    db = DB.SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_auth_context(
    user=Depends(get_current_user),
) -> AuthContext:
    if user:
        actor = getattr(user, "email", None) or getattr(user, "name", None) or "user"
        return AuthContext(user_id=user.id, actor=actor)
    return AuthContext(actor="anonymous")


async def get_request_context(
    auth: AuthContext = Depends(get_auth_context),
) -> RequestContext:
    return RequestContext(auth=auth)
