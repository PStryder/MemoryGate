"""
Shared configuration for MemoryGate core.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memorygate")


def _get_bool(env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(env_name: str, default: float) -> float:
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _derive_effective_backends(db_backend: str, vector_backend: str) -> tuple[str, str]:
    db_effective = db_backend if db_backend in {"postgres", "sqlite"} else "postgres"
    vector_effective = (
        vector_backend if vector_backend in {"pgvector", "sqlite_vss", "none"} else "none"
    )
    if db_effective == "sqlite" and vector_effective == "pgvector":
        vector_effective = "none"
    if db_effective == "postgres" and vector_effective == "sqlite_vss":
        vector_effective = "none"
    if vector_backend == "sqlite_vss":
        vector_effective = "none"
    return db_effective, vector_effective


# Database settings
DB_BACKEND = os.environ.get("DB_BACKEND", "postgres").strip().lower()
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "pgvector").strip().lower()
SQLITE_PATH = os.environ.get("SQLITE_PATH", "/data/memorygate.db")
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_BACKEND_EFFECTIVE, VECTOR_BACKEND_EFFECTIVE = _derive_effective_backends(
    DB_BACKEND,
    VECTOR_BACKEND,
)

# Embedding settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Database initialization controls
AUTO_CREATE_EXTENSIONS = _get_bool("AUTO_CREATE_EXTENSIONS", True)
AUTO_MIGRATE_ON_STARTUP = _get_bool("AUTO_MIGRATE_ON_STARTUP", True)

# Tenancy mode (single tenant only for now)
TENANCY_MODE = os.environ.get("MEMORYGATE_TENANCY_MODE", "single").strip().lower()

# Cleanup cadence for OAuth state/session tables
CLEANUP_INTERVAL_SECONDS = _get_int("CLEANUP_INTERVAL_SECONDS", 900)

# Request/input limits
MAX_RESULT_LIMIT = _get_int("MEMORYGATE_MAX_RESULT_LIMIT", 100)
MAX_QUERY_LENGTH = _get_int("MEMORYGATE_MAX_QUERY_LENGTH", 4000)
MAX_TEXT_LENGTH = _get_int("MEMORYGATE_MAX_TEXT_LENGTH", 8000)
MAX_SHORT_TEXT_LENGTH = _get_int("MEMORYGATE_MAX_SHORT_TEXT_LENGTH", 255)
MAX_DOMAIN_LENGTH = _get_int("MEMORYGATE_MAX_DOMAIN_LENGTH", 100)
MAX_TITLE_LENGTH = _get_int("MEMORYGATE_MAX_TITLE_LENGTH", 500)
MAX_URL_LENGTH = _get_int("MEMORYGATE_MAX_URL_LENGTH", 1000)
MAX_DOC_TYPE_LENGTH = _get_int("MEMORYGATE_MAX_DOC_TYPE_LENGTH", 50)
MAX_CONCEPT_TYPE_LENGTH = _get_int("MEMORYGATE_MAX_CONCEPT_TYPE_LENGTH", 50)
MAX_STATUS_LENGTH = _get_int("MEMORYGATE_MAX_STATUS_LENGTH", 50)
MAX_METADATA_BYTES = _get_int("MEMORYGATE_MAX_METADATA_BYTES", 20000)
MAX_LIST_ITEMS = _get_int("MEMORYGATE_MAX_LIST_ITEMS", 50)
MAX_LIST_ITEM_LENGTH = _get_int("MEMORYGATE_MAX_LIST_ITEM_LENGTH", 1000)
MAX_TAG_ITEMS = _get_int("MEMORYGATE_MAX_TAG_ITEMS", MAX_LIST_ITEMS)
MAX_RELATIONSHIP_ITEMS = _get_int("MEMORYGATE_MAX_RELATIONSHIP_ITEMS", MAX_LIST_ITEMS)
TOOL_INVENTORY_RETRY_SECONDS = _get_int("MEMORYGATE_TOOL_INVENTORY_RETRY_SECONDS", 5)
MAX_EMBEDDING_TEXT_LENGTH = _get_int("MEMORYGATE_MAX_EMBEDDING_TEXT_LENGTH", 8000)

# OpenAI retry/backoff
EMBEDDING_TIMEOUT_SECONDS = _get_float("EMBEDDING_TIMEOUT_SECONDS", 30.0)
EMBEDDING_RETRY_MAX = _get_int("EMBEDDING_RETRY_MAX", 2)
EMBEDDING_RETRY_BACKOFF_SECONDS = _get_float("EMBEDDING_RETRY_BACKOFF_SECONDS", 0.5)
EMBEDDING_RETRY_JITTER_SECONDS = _get_float("EMBEDDING_RETRY_JITTER_SECONDS", 0.25)
EMBEDDING_FAILURE_THRESHOLD = _get_int("EMBEDDING_FAILURE_THRESHOLD", 5)
EMBEDDING_COOLDOWN_SECONDS = _get_int("EMBEDDING_COOLDOWN_SECONDS", 60)
EMBEDDING_HEALTHCHECK_ENABLED = _get_bool("EMBEDDING_HEALTHCHECK_ENABLED", True)
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_BACKFILL_ENABLED = _get_bool("EMBEDDING_BACKFILL_ENABLED", True)
EMBEDDING_BACKFILL_INTERVAL_SECONDS = _get_int("EMBEDDING_BACKFILL_INTERVAL_SECONDS", 300)
EMBEDDING_BACKFILL_BATCH_LIMIT = _get_int("EMBEDDING_BACKFILL_BATCH_LIMIT", 50)

# Retention & scoring
SCORE_BUMP_ALPHA = _get_float("SCORE_BUMP_ALPHA", 0.4)
REHYDRATE_BUMP_ALPHA = _get_float("REHYDRATE_BUMP_ALPHA", 0.2)
SCORE_DECAY_BETA = _get_float("SCORE_DECAY_BETA", 0.02)
SCORE_CLAMP_MIN = _get_float("SCORE_CLAMP_MIN", -3.0)
SCORE_CLAMP_MAX = _get_float("SCORE_CLAMP_MAX", 1.0)
SUMMARY_TRIGGER_SCORE = _get_float("SUMMARY_TRIGGER_SCORE", -1.0)
PURGE_TRIGGER_SCORE = _get_float("PURGE_TRIGGER_SCORE", -2.0)
RETENTION_PRESSURE = _get_float("RETENTION_PRESSURE", 1.0)
RETENTION_BUDGET = _get_int("RETENTION_BUDGET", 100000)
RETENTION_TICK_SECONDS = _get_int("RETENTION_TICK_SECONDS", 900)
COLD_DECAY_MULTIPLIER = _get_float("COLD_DECAY_MULTIPLIER", 0.25)
FORGET_MODE = os.environ.get("FORGET_MODE", "soft").strip().lower()
COLD_SEARCH_ENABLED = _get_bool("COLD_SEARCH_ENABLED", True)
ARCHIVE_LIMIT_DEFAULT = _get_int("ARCHIVE_LIMIT_DEFAULT", 200)
ARCHIVE_LIMIT_MAX = _get_int("ARCHIVE_LIMIT_MAX", 500)
REHYDRATE_LIMIT_MAX = _get_int("REHYDRATE_LIMIT_MAX", 200)
TOMBSTONES_ENABLED = _get_bool("TOMBSTONES_ENABLED", True)
SUMMARY_MAX_LENGTH = _get_int("SUMMARY_MAX_LENGTH", 800)
SUMMARY_BATCH_LIMIT = _get_int("SUMMARY_BATCH_LIMIT", 100)
RETENTION_PURGE_LIMIT = _get_int("RETENTION_PURGE_LIMIT", 100)
ALLOW_HARD_PURGE_WITHOUT_SUMMARY = _get_bool("ALLOW_HARD_PURGE_WITHOUT_SUMMARY", False)
STORAGE_QUOTA_BYTES = _get_int("STORAGE_QUOTA_BYTES", 10_000_000_000)
ARCHIVE_MULTIPLIER = _get_float("ARCHIVE_MULTIPLIER", 2.0)


def validate_and_prepare_config() -> None:
    """Validate configuration and apply derived settings at startup."""
    global DATABASE_URL, DB_BACKEND_EFFECTIVE, VECTOR_BACKEND_EFFECTIVE

    errors = []
    if TENANCY_MODE != "single":
        errors.append("MEMORYGATE_TENANCY_MODE must be 'single'")
    if DB_BACKEND not in {"postgres", "sqlite"}:
        errors.append("DB_BACKEND must be 'postgres' or 'sqlite'")

    if VECTOR_BACKEND not in {"pgvector", "sqlite_vss", "none"}:
        errors.append("VECTOR_BACKEND must be 'pgvector', 'sqlite_vss', or 'none'")

    if DB_BACKEND == "sqlite" and VECTOR_BACKEND == "pgvector":
        errors.append("VECTOR_BACKEND=pgvector requires DB_BACKEND=postgres")

    if DB_BACKEND == "postgres" and VECTOR_BACKEND == "sqlite_vss":
        errors.append("VECTOR_BACKEND=sqlite_vss requires DB_BACKEND=sqlite")

    if not DATABASE_URL:
        if DB_BACKEND == "sqlite":
            if not SQLITE_PATH:
                errors.append("SQLITE_PATH environment variable is required for sqlite")
            else:
                DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
        else:
            errors.append("DATABASE_URL environment variable is required")
    else:
        url_lower = DATABASE_URL.lower()
        is_sqlite_url = url_lower.startswith("sqlite")
        if DB_BACKEND == "sqlite" and not is_sqlite_url:
            errors.append("DATABASE_URL must be a sqlite URL when DB_BACKEND=sqlite")
        if DB_BACKEND == "postgres" and is_sqlite_url:
            errors.append("DATABASE_URL must be a postgres URL when DB_BACKEND=postgres")

    if VECTOR_BACKEND == "sqlite_vss":
        logger.warning(
            "VECTOR_BACKEND=sqlite_vss is not configured; falling back to keyword search."
        )
    DB_BACKEND_EFFECTIVE, VECTOR_BACKEND_EFFECTIVE = _derive_effective_backends(
        DB_BACKEND,
        VECTOR_BACKEND,
    )

    if VECTOR_BACKEND_EFFECTIVE == "pgvector" and EMBEDDING_PROVIDER == "none":
        errors.append("VECTOR_BACKEND=pgvector requires EMBEDDING_PROVIDER to be set")

    if VECTOR_BACKEND_EFFECTIVE == "pgvector":
        from core.models import PGVECTOR_AVAILABLE

        if not PGVECTOR_AVAILABLE:
            errors.append("pgvector package is required when VECTOR_BACKEND=pgvector")

    if errors:
        raise RuntimeError("Configuration invalid: " + "; ".join(errors))
