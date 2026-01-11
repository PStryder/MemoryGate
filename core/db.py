"""
Database initialization and migration helpers.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import core.config as config


class DB:
    """Database state holder (avoids global scoping issues)."""

    engine = None
    SessionLocal = None


def _get_alembic_config():
    try:
        from alembic.config import Config
    except ImportError as exc:
        raise RuntimeError("Alembic is required for migrations") from exc

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
    alembic_cfg.set_main_option("script_location", os.path.join(base_dir, "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", config.DATABASE_URL)
    return alembic_cfg


def _get_schema_revisions(engine) -> tuple[Optional[str], Optional[str]]:
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    alembic_cfg = _get_alembic_config()
    script = ScriptDirectory.from_config(alembic_cfg)
    head_revision = script.get_current_head()
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_revision = context.get_current_revision()
    return current_revision, head_revision


def _ensure_schema_up_to_date(engine) -> None:
    from alembic import command

    current_rev, head_rev = _get_schema_revisions(engine)
    if current_rev == head_rev:
        return

    if config.AUTO_MIGRATE_ON_STARTUP:
        alembic_cfg = _get_alembic_config()
        command.upgrade(alembic_cfg, "head")
        new_current, _ = _get_schema_revisions(engine)
        if new_current != head_rev:
            raise RuntimeError("Database migration did not reach expected revision")
    else:
        raise RuntimeError(
            f"Database schema out of date (current={current_rev}, expected={head_rev}). "
            "Run 'alembic upgrade head' or set AUTO_MIGRATE_ON_STARTUP=true for dev."
        )


def init_db() -> None:
    """Initialize database connection and create tables."""
    config.validate_and_prepare_config()

    config.logger.info("Connecting to database...")
    engine_kwargs = {"pool_pre_ping": True}
    if config.DB_BACKEND == "sqlite":
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    DB.engine = create_engine(config.DATABASE_URL, **engine_kwargs)
    DB.SessionLocal = sessionmaker(bind=DB.engine)

    # FIRST: Ensure pgvector extension exists (optional)
    if (
        config.AUTO_CREATE_EXTENSIONS
        and config.DB_BACKEND == "postgres"
        and config.VECTOR_BACKEND_EFFECTIVE == "pgvector"
    ):
        config.logger.info("Ensuring pgvector extension...")
        with DB.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    else:
        config.logger.info("Skipping pgvector extension creation")

    # Import OAuth models to register tables with Base
    import oauth_models  # noqa: F401

    # Ensure schema is up to date via Alembic migrations
    _ensure_schema_up_to_date(DB.engine)

    config.logger.info("Database initialized")
