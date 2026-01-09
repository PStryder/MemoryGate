"""
MemoryGate - Persistent Memory-as-a-Service for AI Agents
MCP Server with PostgreSQL + pgvector backend
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from typing import Optional, List, Sequence
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import create_engine, text, func, desc, case, and_, or_
from sqlalchemy.orm import sessionmaker
import numpy as np

from models import (
    Base, AIInstance, Session, Observation, Pattern, 
    Concept, ConceptAlias, ConceptRelationship, Document, Embedding,
    MemorySummary, MemoryTombstone, MemoryTier, TombstoneAction
)
from oauth_models import cleanup_expired_sessions, cleanup_expired_states
from rate_limiter import (
    RateLimitMiddleware,
    build_rate_limiter_from_env,
    load_rate_limit_config_from_env,
)
from security_middleware import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    load_request_size_limit_config_from_env,
    load_security_headers_config_from_env,
)
from retention import (
    apply_fetch_bump,
    apply_decay_tick,
    clamp_score,
    apply_floor,
)

# =============================================================================
# Configuration
# =============================================================================

DB_BACKEND = os.environ.get("DB_BACKEND", "postgres").strip().lower()
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "pgvector").strip().lower()
SQLITE_PATH = os.environ.get("SQLITE_PATH", "/data/memorygate.db")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    if DB_BACKEND == "sqlite":
        if not SQLITE_PATH:
            raise RuntimeError("SQLITE_PATH environment variable is required for sqlite")
        DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    else:
        raise RuntimeError("DATABASE_URL environment variable is required")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

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

# Rate limiting configuration (optional Redis backend)
rate_limit_config = load_rate_limit_config_from_env()
rate_limiter = build_rate_limiter_from_env(rate_limit_config)

# Request size and security headers
request_size_config = load_request_size_limit_config_from_env()
security_headers_config = load_security_headers_config_from_env()

if DB_BACKEND not in {"postgres", "sqlite"}:
    raise RuntimeError("DB_BACKEND must be 'postgres' or 'sqlite'")

if VECTOR_BACKEND not in {"pgvector", "sqlite_vss", "none"}:
    raise RuntimeError("VECTOR_BACKEND must be 'pgvector', 'sqlite_vss', or 'none'")

if DB_BACKEND == "sqlite" and VECTOR_BACKEND == "pgvector":
    raise RuntimeError("VECTOR_BACKEND=pgvector requires DB_BACKEND=postgres")

if DB_BACKEND == "postgres" and VECTOR_BACKEND == "sqlite_vss":
    raise RuntimeError("VECTOR_BACKEND=sqlite_vss requires DB_BACKEND=sqlite")

url_lower = DATABASE_URL.lower()
is_sqlite_url = url_lower.startswith("sqlite")
if is_sqlite_url and DB_BACKEND != "sqlite":
    raise RuntimeError("DATABASE_URL is sqlite but DB_BACKEND is not 'sqlite'")
if not is_sqlite_url and DB_BACKEND == "sqlite":
    raise RuntimeError("DB_BACKEND=sqlite requires a sqlite DATABASE_URL")

if TENANCY_MODE != "single":
    raise RuntimeError(
        "Only single-tenant mode is supported. Set MEMORYGATE_TENANCY_MODE=single."
    )

if EMBEDDING_PROVIDER not in {"openai", "local_cpd", "none"}:
    raise RuntimeError(
        "Unknown EMBEDDING_PROVIDER. Use 'openai', 'local_cpd', or 'none'."
    )

if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

if FORGET_MODE not in {"soft", "hard"}:
    raise RuntimeError("FORGET_MODE must be 'soft' or 'hard'")

VECTOR_BACKEND_EFFECTIVE = VECTOR_BACKEND
if VECTOR_BACKEND == "sqlite_vss":
    VECTOR_BACKEND_EFFECTIVE = "none"
    logger.warning("VECTOR_BACKEND=sqlite_vss is not configured; falling back to keyword search.")

if VECTOR_BACKEND_EFFECTIVE == "pgvector" and EMBEDDING_PROVIDER == "none":
    raise RuntimeError("VECTOR_BACKEND=pgvector requires EMBEDDING_PROVIDER to be set")

# =============================================================================
# Global State
# =============================================================================

# Database state holder (avoids global scoping issues)
class DB:
    engine = None
    SessionLocal = None

http_client = None  # Reusable HTTP client for OpenAI API
cleanup_task = None  # Background cleanup loop
retention_task = None  # Background retention loop


def _get_alembic_config():
    try:
        from alembic.config import Config
    except ImportError as exc:
        raise RuntimeError("Alembic is required for migrations") from exc

    base_dir = os.path.dirname(os.path.abspath(__file__))
    alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
    alembic_cfg.set_main_option("script_location", os.path.join(base_dir, "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
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

    if AUTO_MIGRATE_ON_STARTUP:
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


def init_http_client():
    """Initialize HTTP client for OpenAI API calls."""
    global http_client
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    http_client = httpx.Client(
        timeout=httpx.Timeout(EMBEDDING_TIMEOUT_SECONDS),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
        headers=headers,
    )
    logger.info("HTTP client initialized")


def cleanup_http_client():
    """Clean up HTTP client on shutdown."""
    global http_client
    if http_client:
        http_client.close()
        logger.info("HTTP client closed")


async def _run_cleanup_once() -> None:
    if DB.SessionLocal is None:
        return
    db = DB.SessionLocal()
    try:
        cleanup_expired_states(db)
        cleanup_expired_sessions(db)
    finally:
        db.close()


async def _cleanup_loop() -> None:
    if CLEANUP_INTERVAL_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            await _run_cleanup_once()
        except Exception as exc:
            logger.warning(f"Cleanup task error: {exc}")


async def _retention_loop() -> None:
    if RETENTION_TICK_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(RETENTION_TICK_SECONDS)
        try:
            _run_retention_tick()
        except Exception as exc:
            logger.warning(f"Retention task error: {exc}")


def _summarize_and_archive(db) -> dict:
    archived = 0
    summaries_created = 0
    reason = "auto_summarize"
    for mem_type, model in MEMORY_MODELS.items():
        records = (
            db.query(model)
            .filter(model.tier == MemoryTier.hot)
            .filter(model.score <= SUMMARY_TRIGGER_SCORE)
            .order_by(model.score.asc())
            .limit(SUMMARY_BATCH_LIMIT)
            .all()
        )
        for record in records:
            summary_text = _summary_text_for_record(mem_type, record)
            if not summary_text:
                continue
            summary = _find_summary_for_source(db, mem_type, record.id)
            if summary:
                summary.summary_text = summary_text
            else:
                summary = MemorySummary(
                    source_type=mem_type,
                    source_id=record.id,
                    source_ids=[record.id],
                    summary_text=summary_text,
                    metadata_={"reason": reason},
                )
                db.add(summary)
                summaries_created += 1
            _write_tombstone(
                db,
                _serialize_memory_id(mem_type, record.id),
                TombstoneAction.summarized,
                from_tier=record.tier,
                to_tier=record.tier,
                reason=reason,
                actor="system",
            )
            record.tier = MemoryTier.cold
            record.archived_at = datetime.utcnow()
            record.archived_reason = reason
            record.archived_by = "system"
            record.purge_eligible = False
            _write_tombstone(
                db,
                _serialize_memory_id(mem_type, record.id),
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor="system",
            )
            archived += 1

    db.commit()
    return {"archived": archived, "summaries_created": summaries_created}


def _purge_cold_records(db) -> dict:
    purged = 0
    marked = 0
    for mem_type, model in MEMORY_MODELS.items():
        records = (
            db.query(model)
            .filter(model.tier == MemoryTier.cold)
            .filter(model.score <= PURGE_TRIGGER_SCORE)
            .order_by(model.score.asc())
            .limit(RETENTION_PURGE_LIMIT)
            .all()
        )
        for record in records:
            if FORGET_MODE == "soft":
                record.purge_eligible = True
                _write_tombstone(
                    db,
                    _serialize_memory_id(mem_type, record.id),
                    TombstoneAction.purged,
                    from_tier=MemoryTier.cold,
                    to_tier=MemoryTier.cold,
                    reason="soft_purge_marked",
                    actor="system",
                    metadata={"mode": "soft"},
                )
                marked += 1
                continue

            summary = _find_summary_for_source(db, mem_type, record.id)
            if summary is None and not ALLOW_HARD_PURGE_WITHOUT_SUMMARY:
                continue

            db.query(Embedding).filter(
                Embedding.source_type == mem_type,
                Embedding.source_id == record.id
            ).delete()
            db.delete(record)
            _write_tombstone(
                db,
                _serialize_memory_id(mem_type, record.id),
                TombstoneAction.purged,
                from_tier=MemoryTier.cold,
                to_tier=None,
                reason="hard_purge",
                actor="system",
                metadata={"mode": "hard"},
            )
            purged += 1

    # Purge summaries
    summaries = (
        db.query(MemorySummary)
        .filter(MemorySummary.tier == MemoryTier.cold)
        .filter(MemorySummary.score <= PURGE_TRIGGER_SCORE)
        .order_by(MemorySummary.score.asc())
        .limit(RETENTION_PURGE_LIMIT)
        .all()
    )
    for summary in summaries:
        if FORGET_MODE == "soft":
            summary.purge_eligible = True
            _write_tombstone(
                db,
                f"summary:{summary.id}",
                TombstoneAction.purged,
                from_tier=MemoryTier.cold,
                to_tier=MemoryTier.cold,
                reason="soft_purge_marked",
                actor="system",
                metadata={"mode": "soft"},
            )
            marked += 1
            continue

        db.delete(summary)
        _write_tombstone(
            db,
            f"summary:{summary.id}",
            TombstoneAction.purged,
            from_tier=MemoryTier.cold,
            to_tier=None,
            reason="hard_purge",
            actor="system",
            metadata={"mode": "hard"},
        )
        purged += 1

    db.commit()
    return {"purged": purged, "marked": marked}


def _run_retention_tick() -> None:
    if DB.SessionLocal is None:
        return
    db = DB.SessionLocal()
    try:
        pressure = _calculate_pressure_multiplier(db)
        hot_updates = 0
        cold_updates = 0
        for model in MEMORY_MODELS.values():
            hot_updates += _apply_decay_to_model(db, model, MemoryTier.hot, pressure, 1.0)
            cold_updates += _apply_decay_to_model(db, model, MemoryTier.cold, pressure, COLD_DECAY_MULTIPLIER)
        hot_updates += _apply_decay_to_model(db, MemorySummary, MemoryTier.hot, pressure, 1.0)
        cold_updates += _apply_decay_to_model(db, MemorySummary, MemoryTier.cold, pressure, COLD_DECAY_MULTIPLIER)
        db.commit()

        summary_stats = _summarize_and_archive(db)
        purge_stats = _purge_cold_records(db)
        logger.info(
            "Retention tick complete",
            extra={
                "decayed_hot": hot_updates,
                "decayed_cold": cold_updates,
                **summary_stats,
                **purge_stats,
            },
        )
    finally:
        db.close()


def init_db():
    """Initialize database connection and create tables."""
    
    logger.info("Connecting to database...")
    engine_kwargs = {"pool_pre_ping": True}
    if DB_BACKEND == "sqlite":
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    DB.engine = create_engine(DATABASE_URL, **engine_kwargs)
    DB.SessionLocal = sessionmaker(bind=DB.engine)
    
    # FIRST: Ensure pgvector extension exists (optional)
    if AUTO_CREATE_EXTENSIONS and DB_BACKEND == "postgres" and VECTOR_BACKEND_EFFECTIVE == "pgvector":
        logger.info("Ensuring pgvector extension...")
        with DB.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    else:
        logger.info("Skipping pgvector extension creation")
    
    # Import OAuth models to register tables with Base
    import oauth_models  # noqa: F401

    # Ensure schema is up to date via Alembic migrations
    _ensure_schema_up_to_date(DB.engine)
    
    logger.info("Database initialized")


def _embed_text_local_cpd_sync(text: str) -> List[float]:
    """
    Stub for local CPD embeddings (CPU).

    To enable, install sentence-transformers and replace this stub:
      - pip install sentence-transformers
      - from sentence_transformers import SentenceTransformer
      - model = SentenceTransformer("all-MiniLM-L6-v2")
      - return model.encode([text])[0].tolist()
    """
    _raise_embedding_unavailable("local_cpd embedding not configured")


async def _embed_text_local_cpd_async(text: str) -> List[float]:
    return await asyncio.to_thread(_embed_text_local_cpd_sync, text)


async def embed_text(text: str) -> List[float]:
    """Generate embedding using configured provider."""
    _validate_embedding_text(text)
    if EMBEDDING_PROVIDER == "none":
        _raise_embedding_unavailable("embedding provider disabled")
    if EMBEDDING_PROVIDER == "local_cpd":
        return await _embed_text_local_cpd_async(text)
    if embedding_circuit_breaker.is_open():
        _raise_embedding_unavailable("circuit breaker open")
    timeout = httpx.Timeout(EMBEDDING_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(EMBEDDING_RETRY_MAX + 1):
            try:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": text
                    }
                )
            except httpx.RequestError as exc:
                if attempt >= EMBEDDING_RETRY_MAX:
                    embedding_circuit_breaker.record_failure(str(exc))
                    _raise_embedding_unavailable(str(exc))
                await _async_sleep_backoff(attempt)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt >= EMBEDDING_RETRY_MAX:
                    embedding_circuit_breaker.record_failure(
                        f"status {response.status_code}"
                    )
                    _raise_embedding_unavailable(f"status {response.status_code}")
                await _async_sleep_backoff(attempt)
                continue
            if response.status_code >= 400:
                embedding_circuit_breaker.record_failure(
                    f"status {response.status_code}"
                )
                _raise_embedding_unavailable(f"status {response.status_code}")

            response.raise_for_status()
            data = response.json()
            embedding_circuit_breaker.record_success()
            return data["data"][0]["embedding"]


def embed_text_sync(text: str) -> List[float]:
    """Synchronous version of embed_text using pooled HTTP client."""
    _validate_embedding_text(text)
    if EMBEDDING_PROVIDER == "none":
        _raise_embedding_unavailable("embedding provider disabled")
    if EMBEDDING_PROVIDER == "local_cpd":
        return _embed_text_local_cpd_sync(text)
    if embedding_circuit_breaker.is_open():
        _raise_embedding_unavailable("circuit breaker open")
    global http_client
    if http_client is None:
        init_http_client()

    for attempt in range(EMBEDDING_RETRY_MAX + 1):
        try:
            response = http_client.post(
                "https://api.openai.com/v1/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text
                }
            )
        except httpx.RequestError:
            if attempt >= EMBEDDING_RETRY_MAX:
                embedding_circuit_breaker.record_failure("request error")
                _raise_embedding_unavailable("request error")
            _sleep_backoff(attempt)
            continue

        if response.status_code in {429, 500, 502, 503, 504}:
            if attempt >= EMBEDDING_RETRY_MAX:
                embedding_circuit_breaker.record_failure(
                    f"status {response.status_code}"
                )
                _raise_embedding_unavailable(f"status {response.status_code}")
            _sleep_backoff(attempt)
            continue
        if response.status_code >= 400:
            embedding_circuit_breaker.record_failure(
                f"status {response.status_code}"
            )
            _raise_embedding_unavailable(f"status {response.status_code}")

        response.raise_for_status()
        data = response.json()
        embedding_circuit_breaker.record_success()
        return data["data"][0]["embedding"]


def _embed_or_raise(text: str) -> List[float]:
    try:
        return embed_text_sync(text)
    except EmbeddingProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail="embedding provider unavailable",
        ) from exc


def _vector_search_enabled() -> bool:
    return DB_BACKEND == "postgres" and VECTOR_BACKEND_EFFECTIVE == "pgvector"


def _store_embedding(
    db,
    source_type: str,
    source_id: int,
    text_value: str,
    replace: bool = False,
) -> None:
    if not _vector_search_enabled():
        return
    embedding_vector = _embed_or_raise(text_value)
    if replace:
        db.query(Embedding).filter(
            Embedding.source_type == source_type,
            Embedding.source_id == source_id,
        ).delete()
    emb = Embedding(
        source_type=source_type,
        source_id=source_id,
        model_version=EMBEDDING_MODEL,
        embedding=embedding_vector,
        normalized=True,
    )
    db.add(emb)
    db.commit()


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_required_text(value: str, field: str, max_len: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > max_len:
        raise ValueError(f"{field} exceeds max length {max_len}")


def _validate_optional_text(value: Optional[str], field: str, max_len: int) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) > max_len:
        raise ValueError(f"{field} exceeds max length {max_len}")


def _validate_limit(value: int, field: str, max_value: int) -> None:
    if value <= 0 or value > max_value:
        raise ValueError(f"{field} must be between 1 and {max_value}")


def _validate_confidence(value: float, field: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be between 0.0 and 1.0")


def _validate_list(values: Optional[Sequence], field: str, max_items: int) -> None:
    if values is None:
        return
    if len(values) > max_items:
        raise ValueError(f"{field} exceeds max items {max_items}")


def _validate_string_list(
    values: Optional[Sequence[str]],
    field: str,
    max_items: int,
    max_item_length: int
) -> None:
    if values is None:
        return
    if len(values) > max_items:
        raise ValueError(f"{field} exceeds max items {max_items}")
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain only strings")
        if len(item) > max_item_length:
            raise ValueError(f"{field} item exceeds max length {max_item_length}")


def _validate_metadata(metadata: Optional[dict], field: str) -> None:
    if metadata is None:
        return
    try:
        size = len(json.dumps(metadata))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serializable") from exc
    if size > MAX_METADATA_BYTES:
        raise ValueError(f"{field} exceeds max size {MAX_METADATA_BYTES} bytes")


def _validate_embedding_text(text: str) -> None:
    _validate_required_text(text, "text", MAX_EMBEDDING_TEXT_LENGTH)


class EmbeddingProviderError(RuntimeError):
    """Raised when the embedding provider is unavailable."""


class EmbeddingCircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: int):
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(1, cooldown_seconds)
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        self._last_error: Optional[str] = None
        self._last_failure_ts: Optional[float] = None
        self._last_success_ts: Optional[float] = None

    def is_open(self) -> bool:
        with self._lock:
            return time.time() < self._cooldown_until

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._cooldown_until = 0.0
            self._last_success_ts = time.time()

    def record_failure(self, error: str) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = error
            self._last_failure_ts = time.time()
            if self._consecutive_failures >= self._failure_threshold:
                self._cooldown_until = time.time() + self._cooldown_seconds

    def status(self) -> dict:
        with self._lock:
            return {
                "open": time.time() < self._cooldown_until,
                "consecutive_failures": self._consecutive_failures,
                "cooldown_until_epoch": int(self._cooldown_until) if self._cooldown_until else None,
                "last_error": self._last_error,
                "last_failure_epoch": int(self._last_failure_ts) if self._last_failure_ts else None,
                "last_success_epoch": int(self._last_success_ts) if self._last_success_ts else None,
            }


embedding_circuit_breaker = EmbeddingCircuitBreaker(
    failure_threshold=EMBEDDING_FAILURE_THRESHOLD,
    cooldown_seconds=EMBEDDING_COOLDOWN_SECONDS,
)


def _raise_embedding_unavailable(detail: str) -> None:
    logger.warning(f"Embedding provider unavailable: {detail}")
    raise EmbeddingProviderError("embedding provider unavailable")


def _sleep_backoff(attempt: int) -> None:
    base = EMBEDDING_RETRY_BACKOFF_SECONDS * (2 ** attempt)
    jitter = random.uniform(0, EMBEDDING_RETRY_JITTER_SECONDS)
    time.sleep(base + jitter)


async def _async_sleep_backoff(attempt: int) -> None:
    base = EMBEDDING_RETRY_BACKOFF_SECONDS * (2 ** attempt)
    jitter = random.uniform(0, EMBEDDING_RETRY_JITTER_SECONDS)
    await asyncio.sleep(base + jitter)


def _apply_fetch_bump(record, alpha: float) -> None:
    record.access_count = (record.access_count or 0) + 1
    record.last_accessed_at = datetime.utcnow()
    bumped = apply_fetch_bump(record.score, alpha, bump_clamp_min=-2.0, bump_clamp_max=1.0)
    bumped = clamp_score(bumped, SCORE_CLAMP_MIN, SCORE_CLAMP_MAX)
    record.score = apply_floor(bumped, record.floor_score)


def _apply_rehydrate_bump(record) -> None:
    bumped = apply_fetch_bump(record.score, REHYDRATE_BUMP_ALPHA, bump_clamp_min=-2.0, bump_clamp_max=1.0)
    bumped = clamp_score(bumped, SCORE_CLAMP_MIN, SCORE_CLAMP_MAX)
    record.score = apply_floor(bumped, record.floor_score)


def _serialize_memory_id(memory_type: str, memory_id: int) -> str:
    return f"{memory_type}:{memory_id}"


def _write_tombstone(
    db,
    memory_id: str,
    action: TombstoneAction,
    from_tier: Optional[MemoryTier],
    to_tier: Optional[MemoryTier],
    reason: Optional[str],
    actor: Optional[str],
    metadata: Optional[dict] = None,
) -> None:
    if not TOMBSTONES_ENABLED:
        return
    tombstone = MemoryTombstone(
        memory_id=memory_id,
        action=action,
        from_tier=from_tier,
        to_tier=to_tier,
        reason=reason,
        actor=actor,
        metadata_=metadata or {},
    )
    db.add(tombstone)


def _summary_text_for_record(memory_type: str, record) -> str:
    if memory_type == "observation":
        source = record.observation
    elif memory_type == "pattern":
        source = record.pattern_text
    elif memory_type == "concept":
        source = record.description or ""
    elif memory_type == "document":
        source = record.content_summary or ""
    else:
        source = ""
    source = source.strip()
    return source[:SUMMARY_MAX_LENGTH]


def _calculate_pressure_multiplier(db) -> float:
    total = (
        db.query(func.count(Observation.id)).scalar()
        + db.query(func.count(Pattern.id)).scalar()
        + db.query(func.count(Concept.id)).scalar()
        + db.query(func.count(Document.id)).scalar()
        + db.query(func.count(MemorySummary.id)).scalar()
    )
    budget = max(1, RETENTION_BUDGET)
    multiplier = RETENTION_PRESSURE * (total / budget)
    return max(1.0, multiplier)


def _apply_decay_to_model(db, model, tier: MemoryTier, pressure_multiplier: float, decay_multiplier: float) -> int:
    beta = SCORE_DECAY_BETA * decay_multiplier
    if beta <= 0:
        return 0
    score_expr = model.score - beta * pressure_multiplier
    clamped = case(
        (score_expr < SCORE_CLAMP_MIN, SCORE_CLAMP_MIN),
        (score_expr > SCORE_CLAMP_MAX, SCORE_CLAMP_MAX),
        else_=score_expr,
    )
    floored = case(
        (clamped < model.floor_score, model.floor_score),
        else_=clamped,
    )
    result = (
        db.query(model)
        .filter(model.tier == tier)
        .update({model.score: floored}, synchronize_session=False)
    )
    return result or 0


def _find_summary_for_source(db, memory_type: str, source_id: int) -> Optional[MemorySummary]:
    return db.query(MemorySummary).filter(
        MemorySummary.source_type == memory_type,
        MemorySummary.source_id == source_id
    ).first()


def _parse_memory_ref(raw) -> tuple[str, int]:
    if isinstance(raw, int):
        return "observation", raw
    if isinstance(raw, str):
        value = raw.strip()
        if ":" in value:
            mem_type, mem_id = value.split(":", 1)
            return mem_type.strip().lower(), int(mem_id)
        return "observation", int(value)
    raise ValueError("memory_ids must be int or 'type:id' string")


def _collect_records_by_refs(db, refs: list[tuple[str, int]]) -> list[tuple[str, object]]:
    records = []
    for mem_type, mem_id in refs:
        model = MEMORY_MODELS.get(mem_type)
        if not model:
            raise ValueError(f"Unknown memory type: {mem_type}")
        record = db.query(model).filter(model.id == mem_id).first()
        if record:
            records.append((mem_type, record))
    return records


def _collect_threshold_records(
    db,
    tier: MemoryTier,
    below_score: Optional[float],
    above_score: Optional[float],
    types: list[str],
    limit: int,
) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    for mem_type in types:
        model = MEMORY_MODELS.get(mem_type)
        if not model:
            continue
        query = db.query(model).filter(model.tier == tier)
        if below_score is not None:
            query = query.filter(model.score <= below_score)
        if above_score is not None:
            query = query.filter(model.score >= above_score)
        if below_score is not None:
            query = query.order_by(model.score.asc())
        elif above_score is not None:
            query = query.order_by(model.score.desc())
        rows = query.limit(limit).all()
        for row in rows:
            candidates.append((mem_type, row))

    if below_score is not None:
        candidates.sort(key=lambda item: item[1].score)
    elif above_score is not None:
        candidates.sort(key=lambda item: item[1].score, reverse=True)
    return candidates[:limit]


def _collect_summary_threshold_records(
    db,
    tier: MemoryTier,
    below_score: Optional[float],
    above_score: Optional[float],
    limit: int,
) -> list[MemorySummary]:
    query = db.query(MemorySummary).filter(MemorySummary.tier == tier)
    if below_score is not None:
        query = query.filter(MemorySummary.score <= below_score)
        query = query.order_by(MemorySummary.score.asc())
    if above_score is not None:
        query = query.filter(MemorySummary.score >= above_score)
        query = query.order_by(MemorySummary.score.desc())
    return query.limit(limit).all()
def get_or_create_ai_instance(db, name: str, platform: str) -> AIInstance:
    """Get or create an AI instance by name."""
    instance = db.query(AIInstance).filter(AIInstance.name == name).first()
    if not instance:
        instance = AIInstance(name=name, platform=platform)
        db.add(instance)
        db.commit()
        db.refresh(instance)
    return instance


def get_or_create_session(
    db, 
    conversation_id: str, 
    title: Optional[str] = None,
    ai_instance_id: Optional[int] = None,
    source_url: Optional[str] = None
) -> Session:
    """Get or create a session by conversation_id."""
    session = db.query(Session).filter(Session.conversation_id == conversation_id).first()
    if not session:
        session = Session(
            conversation_id=conversation_id,
            title=title,
            ai_instance_id=ai_instance_id,
            source_url=source_url
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    elif title and session.title != title:
        session.title = title
        session.last_active = datetime.utcnow()
        db.commit()
    return session


# =============================================================================
# FastMCP Server
# =============================================================================

mcp = FastMCP("MemoryGate")

MEMORY_MODELS = {
    "observation": Observation,
    "pattern": Pattern,
    "concept": Concept,
    "document": Document,
}


def _search_memory_impl(
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    include_evidence: bool = True,
    bump_score: bool = True,
) -> dict:
    if not _vector_search_enabled():
        return _search_memory_keyword_impl(
            query=query,
            limit=limit,
            min_confidence=min_confidence,
            domain=domain,
            tier_filter=tier_filter,
            min_score=min_score,
            max_score=max_score,
            include_evidence=include_evidence,
            bump_score=bump_score,
        )

    db = DB.SessionLocal()
    try:
        query_embedding = _embed_or_raise(query)

        sql = text("""
            SELECT 
                e.source_type,
                e.source_id,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.observation
                    WHEN e.source_type = 'pattern' THEN p.pattern_text
                    WHEN e.source_type = 'concept' THEN c.description
                    WHEN e.source_type = 'document' THEN d.content_summary
                END as content,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.confidence
                    WHEN e.source_type = 'pattern' THEN p.confidence
                    ELSE 1.0
                END as confidence,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.domain
                    WHEN e.source_type = 'pattern' THEN p.category
                    WHEN e.source_type = 'concept' THEN c.domain
                    WHEN e.source_type = 'document' THEN d.doc_type
                END as domain_or_category,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.timestamp
                    WHEN e.source_type = 'pattern' THEN p.last_updated
                    WHEN e.source_type = 'concept' THEN c.created_at
                    WHEN e.source_type = 'document' THEN d.created_at
                END as timestamp,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.evidence
                    WHEN e.source_type = 'pattern' THEN p.evidence_observation_ids
                    WHEN e.source_type = 'concept' THEN c.metadata
                    WHEN e.source_type = 'document' THEN d.key_concepts
                END as metadata,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.score
                    WHEN e.source_type = 'pattern' THEN p.score
                    WHEN e.source_type = 'concept' THEN c.score
                    WHEN e.source_type = 'document' THEN d.score
                END as score,
                CASE 
                    WHEN e.source_type = 'observation' THEN o.tier
                    WHEN e.source_type = 'pattern' THEN p.tier
                    WHEN e.source_type = 'concept' THEN c.tier
                    WHEN e.source_type = 'document' THEN d.tier
                END as tier,
                CASE 
                    WHEN e.source_type = 'observation' THEN obs_ai.name
                    WHEN e.source_type = 'pattern' THEN pat_ai.name
                    WHEN e.source_type = 'concept' THEN con_ai.name
                    ELSE NULL
                END as ai_name,
                CASE 
                    WHEN e.source_type = 'observation' THEN obs_s.title
                    WHEN e.source_type = 'pattern' THEN pat_s.title
                    ELSE NULL
                END as session_title,
                CASE 
                    WHEN e.source_type = 'concept' THEN c.name
                    WHEN e.source_type = 'pattern' THEN p.pattern_name
                    WHEN e.source_type = 'document' THEN d.title
                    ELSE NULL
                END as item_name,
                1 - (e.embedding <=> cast(:embedding as vector)) as similarity
            FROM embeddings e
            LEFT JOIN observations o ON e.source_type = 'observation' AND e.source_id = o.id
            LEFT JOIN patterns p ON e.source_type = 'pattern' AND e.source_id = p.id
            LEFT JOIN concepts c ON e.source_type = 'concept' AND e.source_id = c.id
            LEFT JOIN documents d ON e.source_type = 'document' AND e.source_id = d.id
            LEFT JOIN ai_instances obs_ai ON o.ai_instance_id = obs_ai.id
            LEFT JOIN ai_instances pat_ai ON p.ai_instance_id = pat_ai.id
            LEFT JOIN ai_instances con_ai ON c.ai_instance_id = con_ai.id
            LEFT JOIN sessions obs_s ON o.session_id = obs_s.id
            LEFT JOIN sessions pat_s ON p.session_id = pat_s.id
            WHERE (
                CASE 
                    WHEN e.source_type = 'observation' THEN o.confidence
                    WHEN e.source_type = 'pattern' THEN p.confidence
                    ELSE 1.0
                END >= :min_confidence
            )
            AND (
                :domain IS NULL 
                OR (e.source_type = 'observation' AND o.domain = :domain)
            )
            AND (
                :tier IS NULL
                OR (e.source_type = 'observation' AND o.tier = :tier)
                OR (e.source_type = 'pattern' AND p.tier = :tier)
                OR (e.source_type = 'concept' AND c.tier = :tier)
                OR (e.source_type = 'document' AND d.tier = :tier)
            )
            AND (
                :min_score IS NULL
                OR (
                    CASE 
                        WHEN e.source_type = 'observation' THEN o.score
                        WHEN e.source_type = 'pattern' THEN p.score
                        WHEN e.source_type = 'concept' THEN c.score
                        WHEN e.source_type = 'document' THEN d.score
                    END >= :min_score
                )
            )
            AND (
                :max_score IS NULL
                OR (
                    CASE 
                        WHEN e.source_type = 'observation' THEN o.score
                        WHEN e.source_type = 'pattern' THEN p.score
                        WHEN e.source_type = 'concept' THEN c.score
                        WHEN e.source_type = 'document' THEN d.score
                    END <= :max_score
                )
            )
            ORDER BY e.embedding <=> cast(:embedding as vector)
            LIMIT :limit
        """)

        results = db.execute(sql, {
            "embedding": str(query_embedding),
            "min_confidence": min_confidence,
            "domain": domain,
            "limit": limit,
            "tier": tier_filter.value if tier_filter else None,
            "min_score": min_score,
            "max_score": max_score,
        }).fetchall()

        if bump_score:
            for row in results:
                if row.tier != MemoryTier.hot.value:
                    continue
                if row.source_type == 'observation':
                    record = db.query(Observation).filter(Observation.id == row.source_id).first()
                elif row.source_type == 'pattern':
                    record = db.query(Pattern).filter(Pattern.id == row.source_id).first()
                elif row.source_type == 'concept':
                    record = db.query(Concept).filter(Concept.id == row.source_id).first()
                elif row.source_type == 'document':
                    record = db.query(Document).filter(Document.id == row.source_id).first()
                else:
                    record = None
                if record:
                    _apply_fetch_bump(record, SCORE_BUMP_ALPHA)
            db.commit()

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "source_type": row.source_type,
                    "id": row.source_id,
                    "content": row.content,
                    "snippet": (row.content or "")[:200],
                    "name": row.item_name,
                    "confidence": row.confidence,
                    "domain": row.domain_or_category,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "metadata": row.metadata if include_evidence else None,
                    "ai_name": row.ai_name,
                    "session_title": row.session_title,
                    "similarity": float(row.similarity),
                    "score": float(row.score) if row.score is not None else None,
                    "tier": row.tier,
                }
                for row in results
            ]
        }
    finally:
        db.close()


def _search_memory_keyword_impl(
    query: str,
    limit: int,
    min_confidence: float,
    domain: Optional[str],
    tier_filter: Optional[MemoryTier],
    min_score: Optional[float],
    max_score: Optional[float],
    include_evidence: bool,
    bump_score: bool,
) -> dict:
    db = DB.SessionLocal()
    try:
        pattern = f"%{query}%"
        candidates: list[tuple[str, object]] = []

        def add_candidates(
            model,
            source_type: str,
            content_attr: str,
            timestamp_attr: str,
        ) -> None:
            query_builder = db.query(model)
            if tier_filter is not None:
                query_builder = query_builder.filter(model.tier == tier_filter)
            if min_score is not None:
                query_builder = query_builder.filter(model.score >= min_score)
            if max_score is not None:
                query_builder = query_builder.filter(model.score <= max_score)
            if source_type in {"observation", "pattern"} and min_confidence > 0:
                query_builder = query_builder.filter(model.confidence >= min_confidence)
            if source_type == "observation" and domain:
                query_builder = query_builder.filter(model.domain == domain)

            content_column = getattr(model, content_attr)
            query_builder = query_builder.filter(content_column.ilike(pattern))
            query_builder = query_builder.order_by(desc(getattr(model, timestamp_attr)))
            rows = query_builder.limit(limit).all()
            for row in rows:
                candidates.append((source_type, row))

        add_candidates(Observation, "observation", "observation", "timestamp")
        add_candidates(Pattern, "pattern", "pattern_text", "last_updated")
        add_candidates(Concept, "concept", "description", "created_at")
        add_candidates(Document, "document", "content_summary", "created_at")

        def candidate_timestamp(item: tuple[str, object]) -> float:
            source_type, row = item
            if source_type == "observation":
                ts = row.timestamp
            elif source_type == "pattern":
                ts = row.last_updated
            elif source_type == "concept":
                ts = row.created_at
            else:
                ts = row.created_at
            return ts.timestamp() if ts else 0.0

        candidates.sort(key=candidate_timestamp, reverse=True)
        selected = candidates[:limit]

        if bump_score:
            for source_type, row in selected:
                if row.tier == MemoryTier.hot:
                    _apply_fetch_bump(row, SCORE_BUMP_ALPHA)
            db.commit()

        results = []
        for source_type, row in selected:
            if source_type == "observation":
                content = row.observation
                confidence = row.confidence
                domain_or_category = row.domain
                timestamp = row.timestamp
                metadata = row.evidence
                ai_name = row.ai_instance.name if row.ai_instance else None
                session_title = row.session.title if row.session else None
                item_name = None
            elif source_type == "pattern":
                content = row.pattern_text
                confidence = row.confidence
                domain_or_category = row.category
                timestamp = row.last_updated
                metadata = row.evidence_observation_ids
                ai_name = row.ai_instance.name if row.ai_instance else None
                session_title = row.session.title if row.session else None
                item_name = row.pattern_name
            elif source_type == "concept":
                content = row.description
                confidence = 1.0
                domain_or_category = row.domain
                timestamp = row.created_at
                metadata = row.metadata_
                ai_name = row.ai_instance.name if row.ai_instance else None
                session_title = None
                item_name = row.name
            else:
                content = row.content_summary
                confidence = 1.0
                domain_or_category = row.doc_type
                timestamp = row.created_at
                metadata = row.key_concepts
                ai_name = None
                session_title = None
                item_name = row.title

            results.append(
                {
                    "source_type": source_type,
                    "id": row.id,
                    "content": content,
                    "snippet": (content or "")[:200],
                    "name": item_name,
                    "confidence": confidence,
                    "domain": domain_or_category,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "metadata": metadata if include_evidence else None,
                    "ai_name": ai_name,
                    "session_title": session_title,
                    "similarity": 0.0,
                    "score": float(row.score) if row.score is not None else None,
                    "tier": row.tier.value if isinstance(row.tier, MemoryTier) else row.tier,
                }
            )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }
    finally:
        db.close()
@mcp.tool()
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None,
    include_cold: bool = False
) -> dict:
    """
    Unified semantic search across all memory types (observations, patterns, concepts, documents).
    
    Args:
        query: Search query text
        limit: Maximum results to return (default 5)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        domain: Optional domain filter (applies to observations only)
        include_cold: Include cold tier records
    
    Returns:
        List of matching items from all sources with similarity scores and source_type
    """
    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)

    tier_filter = None if include_cold else MemoryTier.hot
    return _search_memory_impl(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        domain=domain,
        tier_filter=tier_filter,
        include_evidence=True,
        bump_score=True,
    )


@mcp.tool()
def search_cold_memory(
    query: str,
    top_k: int = 10,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    type_filter: Optional[str] = None,
    source: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags: Optional[List[str]] = None,
    include_evidence: bool = True,
    bump_score: bool = False
) -> dict:
    """
    Explicit search over cold-tier memory records.
    """
    if not COLD_SEARCH_ENABLED:
        return {"status": "error", "message": "Cold search is disabled"}

    _validate_required_text(query, "query", MAX_QUERY_LENGTH)
    _validate_limit(top_k, "top_k", 50)
    _validate_optional_text(type_filter, "type_filter", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(source, "source", MAX_SHORT_TEXT_LENGTH)
    _validate_string_list(tags, "tags", MAX_LIST_ITEMS, MAX_LIST_ITEM_LENGTH)

    dt_from = None
    dt_to = None
    if date_from:
        _validate_optional_text(date_from, "date_from", MAX_SHORT_TEXT_LENGTH)
        dt_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
    if date_to:
        _validate_optional_text(date_to, "date_to", MAX_SHORT_TEXT_LENGTH)
        dt_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

    fetch_limit = min(max(top_k * 5, top_k), MAX_RESULT_LIMIT)
    results = _search_memory_impl(
        query=query,
        limit=fetch_limit,
        min_confidence=0.0,
        domain=None,
        tier_filter=MemoryTier.cold,
        min_score=min_score,
        max_score=max_score,
        include_evidence=include_evidence or bool(tags),
        bump_score=bump_score,
    )

    filtered = []
    for row in results["results"]:
        if type_filter and row["source_type"] != type_filter:
            continue
        if source and row.get("ai_name") != source:
            continue
        if dt_from or dt_to:
            if not row.get("timestamp"):
                continue
            timestamp = datetime.fromisoformat(row["timestamp"])
            if dt_from and timestamp < dt_from:
                continue
            if dt_to and timestamp > dt_to:
                continue
        if tags:
            metadata = row.get("metadata") or []
            tag_set = set(tags)
            match = False
            if isinstance(metadata, list):
                match = bool(tag_set.intersection({str(item) for item in metadata}))
            elif isinstance(metadata, dict):
                meta_tags = metadata.get("tags", [])
                if isinstance(meta_tags, list):
                    match = bool(tag_set.intersection({str(item) for item in meta_tags}))
            if not match:
                continue
        filtered.append(row)
        if len(filtered) >= top_k:
            break

    return {
        "query": query,
        "count": len(filtered),
        "results": filtered,
    }


@mcp.tool()
def archive_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    cluster_ids: Optional[List[str]] = None,
    threshold: Optional[dict] = None,
    mode: str = "archive_and_tombstone",
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = True,
    limit: int = ARCHIVE_LIMIT_DEFAULT
) -> dict:
    """
    Archive hot records into the cold tier.
    """
    if cluster_ids:
        return {"status": "error", "message": "cluster_ids not supported"}
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)

    mode = mode.strip().lower()
    valid_modes = {"archive_only", "archive_and_tombstone", "archive_and_summarize_then_archive"}
    if mode not in valid_modes:
        return {"status": "error", "message": f"Invalid mode. Must be one of: {', '.join(sorted(valid_modes))}"}

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        candidates: list[tuple[str, object]] = []
        summary_records: list[MemorySummary] = []

        if memory_ids:
            refs = [_parse_memory_ref(raw) for raw in memory_ids]
            candidates.extend(_collect_records_by_refs(db, refs))

        if summary_ids:
            summary_records.extend(
                db.query(MemorySummary).filter(MemorySummary.id.in_(summary_ids)).all()
            )

        if threshold:
            below_score = threshold.get("below_score")
            threshold_type = threshold.get("type", "memory").lower()
            if below_score is None:
                return {"status": "error", "message": "threshold requires below_score"}
            if threshold_type not in {"memory", "summary", "any"}:
                return {"status": "error", "message": "threshold.type must be memory|summary|any"}

            if threshold_type in {"memory", "any"}:
                candidates.extend(
                    _collect_threshold_records(
                        db,
                        tier=MemoryTier.hot,
                        below_score=below_score,
                        above_score=None,
                        types=list(MEMORY_MODELS.keys()),
                        limit=limit,
                    )
                )
            if threshold_type in {"summary", "any"}:
                summary_records.extend(
                    _collect_summary_threshold_records(
                        db,
                        tier=MemoryTier.hot,
                        below_score=below_score,
                        above_score=None,
                        limit=limit,
                    )
                )

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for mem_type, record in candidates:
            key = (mem_type, record.id)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append((mem_type, record))
        candidates = unique_candidates[:limit]

        summary_records = list({summary.id: summary for summary in summary_records}.values())[:limit]

        if dry_run:
            return {
                "status": "dry_run",
                "candidate_count": len(candidates),
                "summary_candidate_count": len(summary_records),
                "candidates": [
                    {"type": mem_type, "id": record.id, "score": record.score}
                    for mem_type, record in candidates
                ],
                "summary_candidates": [
                    {"id": summary.id, "score": summary.score}
                    for summary in summary_records
                ],
            }

        archived_ids = []
        tombstones_written = 0
        summaries_created = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.hot:
                continue

            if mode == "archive_and_summarize_then_archive":
                summary = _find_summary_for_source(db, mem_type, record.id)
                summary_text = _summary_text_for_record(mem_type, record)
                if summary:
                    summary.summary_text = summary_text
                else:
                    summary = MemorySummary(
                        source_type=mem_type,
                        source_id=record.id,
                        source_ids=[record.id],
                        summary_text=summary_text,
                        metadata_={"reason": reason},
                    )
                    db.add(summary)
                    summaries_created += 1
                _write_tombstone(
                    db,
                    _serialize_memory_id(mem_type, record.id),
                    TombstoneAction.summarized,
                    from_tier=record.tier,
                    to_tier=record.tier,
                    reason=reason,
                    actor=actor_name,
                )
                tombstones_written += 1 if TOMBSTONES_ENABLED else 0

            record.tier = MemoryTier.cold
            record.archived_at = datetime.utcnow()
            record.archived_reason = reason
            record.archived_by = actor_name
            record.purge_eligible = False
            archived_ids.append(_serialize_memory_id(mem_type, record.id))
            _write_tombstone(
                db,
                _serialize_memory_id(mem_type, record.id),
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        for summary in summary_records:
            if summary.tier != MemoryTier.hot:
                continue
            summary.tier = MemoryTier.cold
            summary.archived_at = datetime.utcnow()
            summary.archived_reason = reason
            summary.archived_by = actor_name
            summary.purge_eligible = False
            archived_ids.append(f"summary:{summary.id}")
            _write_tombstone(
                db,
                f"summary:{summary.id}",
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        db.commit()

        return {
            "status": "archived",
            "archived_count": len(archived_ids),
            "archived_ids": archived_ids,
            "tombstones_written": tombstones_written,
            "summaries_created": summaries_created,
        }
    finally:
        db.close()


@mcp.tool()
def rehydrate_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    cluster_ids: Optional[List[str]] = None,
    threshold: Optional[dict] = None,
    query: Optional[str] = None,
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = False,
    limit: int = 50,
    bump_score: bool = True
) -> dict:
    """
    Rehydrate cold records back into hot tier.
    """
    if cluster_ids:
        return {"status": "error", "message": "cluster_ids not supported"}
    if not reason or not reason.strip():
        return {"status": "error", "message": "reason is required"}
    _validate_limit(limit, "limit", REHYDRATE_LIMIT_MAX)

    db = DB.SessionLocal()
    try:
        actor_name = actor or "mcp"
        candidates: list[tuple[str, object]] = []
        summary_records: list[MemorySummary] = []

        if memory_ids:
            refs = [_parse_memory_ref(raw) for raw in memory_ids]
            candidates.extend(_collect_records_by_refs(db, refs))

        if summary_ids:
            summary_records.extend(
                db.query(MemorySummary).filter(MemorySummary.id.in_(summary_ids)).all()
            )

        if query:
            cold_results = search_cold_memory(query=query, top_k=limit)
            for row in cold_results.get("results", []):
                candidates.extend(_collect_records_by_refs(
                    db,
                    [(_parse_memory_ref(_serialize_memory_id(row["source_type"], row["id"])))],
                ))

        if threshold:
            below_score = threshold.get("below_score")
            above_score = threshold.get("above_score")
            threshold_type = threshold.get("type", "memory").lower()
            if threshold_type not in {"memory", "summary", "any"}:
                return {"status": "error", "message": "threshold.type must be memory|summary|any"}

            if threshold_type in {"memory", "any"}:
                candidates.extend(
                    _collect_threshold_records(
                        db,
                        tier=MemoryTier.cold,
                        below_score=below_score,
                        above_score=above_score,
                        types=list(MEMORY_MODELS.keys()),
                        limit=limit,
                    )
                )
            if threshold_type in {"summary", "any"}:
                summary_records.extend(
                    _collect_summary_threshold_records(
                        db,
                        tier=MemoryTier.cold,
                        below_score=below_score,
                        above_score=above_score,
                        limit=limit,
                    )
                )

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for mem_type, record in candidates:
            key = (mem_type, record.id)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append((mem_type, record))
        candidates = unique_candidates[:limit]

        summary_records = list({summary.id: summary for summary in summary_records}.values())[:limit]

        if dry_run:
            return {
                "status": "dry_run",
                "candidate_count": len(candidates),
                "summary_candidate_count": len(summary_records),
                "candidates": [
                    {"type": mem_type, "id": record.id, "score": record.score}
                    for mem_type, record in candidates
                ],
                "summary_candidates": [
                    {"id": summary.id, "score": summary.score}
                    for summary in summary_records
                ],
            }

        rehydrated_ids = []
        tombstones_written = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.cold:
                continue
            record.tier = MemoryTier.hot
            record.archived_at = None
            record.archived_reason = None
            record.archived_by = None
            record.purge_eligible = False
            if bump_score:
                record.access_count = (record.access_count or 0) + 1
                record.last_accessed_at = datetime.utcnow()
                _apply_rehydrate_bump(record)
            rehydrated_ids.append(_serialize_memory_id(mem_type, record.id))
            _write_tombstone(
                db,
                _serialize_memory_id(mem_type, record.id),
                TombstoneAction.rehydrated,
                from_tier=MemoryTier.cold,
                to_tier=MemoryTier.hot,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        for summary in summary_records:
            if summary.tier != MemoryTier.cold:
                continue
            summary.tier = MemoryTier.hot
            summary.archived_at = None
            summary.archived_reason = None
            summary.archived_by = None
            summary.purge_eligible = False
            if bump_score:
                summary.access_count = (summary.access_count or 0) + 1
                summary.last_accessed_at = datetime.utcnow()
                _apply_rehydrate_bump(summary)
            rehydrated_ids.append(f"summary:{summary.id}")
            _write_tombstone(
                db,
                f"summary:{summary.id}",
                TombstoneAction.rehydrated,
                from_tier=MemoryTier.cold,
                to_tier=MemoryTier.hot,
                reason=reason,
                actor=actor_name,
            )
            tombstones_written += 1 if TOMBSTONES_ENABLED else 0

        db.commit()

        return {
            "status": "rehydrated",
            "rehydrated_count": len(rehydrated_ids),
            "rehydrated_ids": rehydrated_ids,
            "tombstones_written": tombstones_written,
        }
    finally:
        db.close()


@mcp.tool()
def list_archive_candidates(
    below_score: float = SUMMARY_TRIGGER_SCORE,
    limit: int = ARCHIVE_LIMIT_DEFAULT
) -> dict:
    """
    List archive candidates without mutation.
    """
    _validate_limit(limit, "limit", ARCHIVE_LIMIT_MAX)
    db = DB.SessionLocal()
    try:
        candidates = _collect_threshold_records(
            db,
            tier=MemoryTier.hot,
            below_score=below_score,
            above_score=None,
            types=list(MEMORY_MODELS.keys()),
            limit=limit,
        )
        return {
            "status": "ok",
            "candidate_count": len(candidates),
            "candidates": [
                {"type": mem_type, "id": record.id, "score": record.score}
                for mem_type, record in candidates
            ],
        }
    finally:
        db.close()


@mcp.tool()
def memory_store(
    observation: str,
    confidence: float = 0.8,
    domain: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    ai_name: str = "Unknown",
    ai_platform: str = "Unknown",
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None
) -> dict:
    """
    Store a new observation with embedding.
    
    Args:
        observation: The observation text to store
        confidence: Confidence level 0.0-1.0 (default 0.8)
        domain: Category/domain tag
        evidence: List of supporting evidence
        ai_name: Name of AI instance (e.g., "Kee", "Hexy")
        ai_platform: Platform name (e.g., "Claude", "ChatGPT")
        conversation_id: UUID of the conversation
        conversation_title: Title of the conversation
    
    Returns:
        The stored observation with its ID
    """
    _validate_required_text(observation, "observation", MAX_TEXT_LENGTH)
    _validate_confidence(confidence, "confidence")
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_string_list(evidence, "evidence", MAX_LIST_ITEMS, MAX_LIST_ITEM_LENGTH)
    _validate_required_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(conversation_title, "conversation_title", MAX_TITLE_LENGTH)

    db = DB.SessionLocal()
    try:
        # Get or create AI instance
        ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
        
        # Get or create session if conversation_id provided
        session = None
        if conversation_id:
            session = get_or_create_session(
                db, conversation_id, conversation_title, ai_instance.id
            )
        
        # Create observation
        obs = Observation(
            observation=observation,
            confidence=confidence,
            domain=domain,
            evidence=evidence or [],
            ai_instance_id=ai_instance.id,
            session_id=session.id if session else None
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)
        
        # Generate and store embedding (if enabled)
        _store_embedding(db, "observation", obs.id, observation)
        
        return {
            "status": "stored",
            "id": obs.id,
            "observation": observation,
            "confidence": confidence,
            "domain": domain,
            "ai_name": ai_name,
            "session_title": conversation_title
        }
    finally:
        db.close()


@mcp.tool()
def memory_recall(
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    ai_name: Optional[str] = None,
    include_cold: bool = False
) -> dict:
    """
    Recall observations by domain and/or confidence filter.
    
    Args:
        domain: Filter by domain/category
        min_confidence: Minimum confidence threshold
        limit: Maximum results (default 10)
        ai_name: Filter by AI instance name
        include_cold: Include cold tier records
    
    Returns:
        List of matching observations
    """
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        query = db.query(Observation).join(
            AIInstance, Observation.ai_instance_id == AIInstance.id, isouter=True
        ).join(
            Session, Observation.session_id == Session.id, isouter=True
        )
        
        if domain:
            query = query.filter(Observation.domain == domain)
        if min_confidence > 0:
            query = query.filter(Observation.confidence >= min_confidence)
        if ai_name:
            query = query.filter(AIInstance.name == ai_name)
        if not include_cold:
            query = query.filter(Observation.tier == MemoryTier.hot)
        
        results = query.order_by(desc(Observation.timestamp)).limit(limit).all()
        
        if not include_cold:
            for obs in results:
                _apply_fetch_bump(obs, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "count": len(results),
            "filters": {
                "domain": domain,
                "min_confidence": min_confidence,
                "ai_name": ai_name
            },
            "results": [
                {
                    "id": obs.id,
                    "observation": obs.observation,
                    "confidence": obs.confidence,
                    "domain": obs.domain,
                    "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
                    "evidence": obs.evidence,
                    "ai_name": obs.ai_instance.name if obs.ai_instance else None,
                    "session_title": obs.session.title if obs.session else None
                }
                for obs in results
            ]
        }
    finally:
        db.close()


@mcp.tool()
def memory_stats() -> dict:
    """
    Get memory system statistics.
    
    Returns:
        Counts and statistics about stored data
    """
    db = DB.SessionLocal()
    try:
        obs_count = db.query(func.count(Observation.id)).scalar()
        pattern_count = db.query(func.count(Pattern.id)).scalar()
        concept_count = db.query(func.count(Concept.id)).scalar()
        document_count = db.query(func.count(Document.id)).scalar()
        summary_count = db.query(func.count(MemorySummary.id)).scalar()
        session_count = db.query(func.count(Session.id)).scalar()
        ai_count = db.query(func.count(AIInstance.id)).scalar()
        embedding_count = db.query(func.count(Embedding.source_id)).scalar()
        
        # Get AI instances
        ai_instances = db.query(AIInstance).all()
        
        # Get domain distribution
        domains = db.query(
            Observation.domain, func.count(Observation.id)
        ).group_by(Observation.domain).all()
        
        hot_counts = {
            "observations": db.query(func.count(Observation.id)).filter(Observation.tier == MemoryTier.hot).scalar(),
            "patterns": db.query(func.count(Pattern.id)).filter(Pattern.tier == MemoryTier.hot).scalar(),
            "concepts": db.query(func.count(Concept.id)).filter(Concept.tier == MemoryTier.hot).scalar(),
            "documents": db.query(func.count(Document.id)).filter(Document.tier == MemoryTier.hot).scalar(),
            "summaries": db.query(func.count(MemorySummary.id)).filter(MemorySummary.tier == MemoryTier.hot).scalar(),
        }
        cold_counts = {
            "observations": db.query(func.count(Observation.id)).filter(Observation.tier == MemoryTier.cold).scalar(),
            "patterns": db.query(func.count(Pattern.id)).filter(Pattern.tier == MemoryTier.cold).scalar(),
            "concepts": db.query(func.count(Concept.id)).filter(Concept.tier == MemoryTier.cold).scalar(),
            "documents": db.query(func.count(Document.id)).filter(Document.tier == MemoryTier.cold).scalar(),
            "summaries": db.query(func.count(MemorySummary.id)).filter(MemorySummary.tier == MemoryTier.cold).scalar(),
        }

        return {
            "status": "healthy",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "counts": {
                "observations": obs_count,
                "patterns": pattern_count,
                "concepts": concept_count,
                "documents": document_count,
                "summaries": summary_count,
                "sessions": session_count,
                "ai_instances": ai_count,
                "embeddings": embedding_count
            },
            "tiers": {
                "hot": hot_counts,
                "cold": cold_counts,
            },
            "ai_instances": [
                {"name": ai.name, "platform": ai.platform}
                for ai in ai_instances
            ],
            "domains": {
                domain or "untagged": count 
                for domain, count in domains
            }
        }
    finally:
        db.close()


@mcp.tool()
def memory_init_session(
    conversation_id: str,
    title: str,
    ai_name: str,
    ai_platform: str,
    source_url: Optional[str] = None
) -> dict:
    """
    Initialize or update a session for the current conversation.
    
    Args:
        conversation_id: Unique conversation identifier (UUID)
        title: Conversation title
        ai_name: Name of AI instance (e.g., "Kee")
        ai_platform: Platform (e.g., "Claude")
        source_url: Optional URL to the conversation
    
    Returns:
        Session information
    """
    _validate_required_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(title, "title", MAX_TITLE_LENGTH)
    _validate_required_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(source_url, "source_url", MAX_URL_LENGTH)

    db = DB.SessionLocal()
    try:
        ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
        session = get_or_create_session(
            db, conversation_id, title, ai_instance.id, source_url
        )
        
        return {
            "status": "initialized",
            "session_id": session.id,
            "conversation_id": conversation_id,
            "title": title,
            "ai_name": ai_name,
            "ai_platform": ai_platform,
            "started_at": session.started_at.isoformat() if session.started_at else None
        }
    finally:
        db.close()


@mcp.tool()
def memory_store_document(
    title: str,
    doc_type: str,
    url: str,
    content_summary: str,
    key_concepts: Optional[List[str]] = None,
    publication_date: Optional[str] = None,
    metadata: Optional[dict] = None
) -> dict:
    """
    Store a document reference with summary (canonical storage: Google Drive).
    
    Documents are stored as references with summaries, not full content.
    Full content lives in canonical storage (Google Drive) and is fetched on demand.
    
    Args:
        title: Document title
        doc_type: Type of document (article, paper, book, documentation, etc.)
        url: URL to document (Google Drive share link, https://drive.google.com/...)
        content_summary: Summary or abstract (this gets embedded for search)
        key_concepts: List of key concepts/topics (optional)
        publication_date: Publication date in ISO format (optional)
        metadata: Additional metadata as dict (optional)
    
    Returns:
        The stored document with its ID
    """
    _validate_required_text(title, "title", MAX_TITLE_LENGTH)
    _validate_required_text(doc_type, "doc_type", MAX_DOC_TYPE_LENGTH)
    _validate_required_text(url, "url", MAX_URL_LENGTH)
    _validate_required_text(content_summary, "content_summary", MAX_TEXT_LENGTH)
    _validate_string_list(key_concepts, "key_concepts", MAX_LIST_ITEMS, MAX_LIST_ITEM_LENGTH)
    _validate_optional_text(publication_date, "publication_date", MAX_SHORT_TEXT_LENGTH)
    _validate_metadata(metadata, "metadata")

    db = DB.SessionLocal()
    try:
        # Parse publication date if provided
        pub_date = None
        if publication_date:
            try:
                pub_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid publication_date format: {publication_date}")
        
        # Create document
        doc = Document(
            title=title,
            doc_type=doc_type,
            url=url,
            content_summary=content_summary,
            publication_date=pub_date,
            key_concepts=key_concepts or [],
            metadata_=metadata or {}
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        # Generate and store embedding from summary (if enabled)
        _store_embedding(db, "document", doc.id, content_summary)
        
        return {
            "status": "stored",
            "id": doc.id,
            "title": title,
            "doc_type": doc_type,
            "url": url,
            "key_concepts": key_concepts,
            "publication_date": publication_date
        }
    finally:
        db.close()


@mcp.tool()
def memory_store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[dict] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None
) -> dict:
    """
    Store a new concept in the knowledge graph with embedding.
    
    Args:
        name: Concept name (case will be preserved)
        concept_type: Type of concept (project/framework/component/construct/theory)
        description: Description text (this gets embedded for semantic search)
        domain: Optional domain/category
        status: Optional status (active/archived/deprecated/etc)
        metadata: Optional metadata dict
        ai_name: Optional AI instance name
        ai_platform: Optional AI platform
    
    Returns:
        The stored concept with its ID
    """
    _validate_required_text(name, "name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(concept_type, "concept_type", MAX_CONCEPT_TYPE_LENGTH)
    _validate_required_text(description, "description", MAX_TEXT_LENGTH)
    _validate_optional_text(domain, "domain", MAX_DOMAIN_LENGTH)
    _validate_optional_text(status, "status", MAX_STATUS_LENGTH)
    _validate_metadata(metadata, "metadata")
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        # Get AI instance if provided
        ai_instance_id = None
        if ai_name and ai_platform:
            ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
            ai_instance_id = ai_instance.id
        
        # Check if concept already exists (case-insensitive)
        name_key = name.lower()
        existing = db.query(Concept).filter(Concept.name_key == name_key).first()
        if existing:
            return {
                "status": "error",
                "message": f"Concept '{name}' already exists with ID {existing.id}",
                "existing_id": existing.id
            }
        
        # Create concept
        concept = Concept(
            name=name,
            name_key=name_key,
            type=concept_type,
            description=description,
            domain=domain,
            status=status,
            metadata_=metadata or {},
            ai_instance_id=ai_instance_id
        )
        db.add(concept)
        db.commit()
        db.refresh(concept)
        
        # Generate and store embedding from description (if enabled)
        _store_embedding(db, "concept", concept.id, description)
        
        return {
            "status": "stored",
            "id": concept.id,
            "name": name,
            "type": concept_type,
            "description": description
        }
    finally:
        db.close()


@mcp.tool()
def memory_get_concept(name: str, include_cold: bool = False) -> dict:
    """
    Get a concept by name (case-insensitive, alias-aware).
    
    Args:
        name: Concept name or alias to look up
        include_cold: Include cold tier records
    
    Returns:
        Concept details or None if not found
    """
    _validate_required_text(name, "name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        name_key = name.lower()
        
        # Try direct lookup first
        concept_query = db.query(Concept).filter(Concept.name_key == name_key)
        if not include_cold:
            concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
        concept = concept_query.first()
        
        # If not found, check aliases
        if not concept:
            from models import ConceptAlias
            alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == name_key).first()
            if alias:
                concept_query = db.query(Concept).filter(Concept.id == alias.concept_id)
                if not include_cold:
                    concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
                concept = concept_query.first()
        
        if not concept:
            return {"status": "not_found", "name": name}
        
        if concept.tier == MemoryTier.hot:
            _apply_fetch_bump(concept, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "status": "found",
            "id": concept.id,
            "name": concept.name,
            "type": concept.type,
            "description": concept.description,
            "domain": concept.domain,
            "status": concept.status,
            "metadata": concept.metadata_,
            "access_count": concept.access_count
        }
    finally:
        db.close()


@mcp.tool()
def memory_add_concept_alias(concept_name: str, alias: str) -> dict:
    """
    Add an alternative name (alias) for a concept.
    
    Args:
        concept_name: Primary concept name
        alias: Alternative name to add
    
    Returns:
        Status of alias creation
    """
    _validate_required_text(concept_name, "concept_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(alias, "alias", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        from models import ConceptAlias
        
        # Find the concept
        concept_key = concept_name.lower()
        concept = db.query(Concept).filter(Concept.name_key == concept_key).first()
        if not concept:
            return {"status": "error", "message": f"Concept '{concept_name}' not found"}
        
        # Check if alias already exists
        alias_key = alias.lower()
        existing_alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == alias_key).first()
        if existing_alias:
            return {"status": "error", "message": f"Alias '{alias}' already exists"}
        
        # Check if alias conflicts with existing concept name
        existing_concept = db.query(Concept).filter(Concept.name_key == alias_key).first()
        if existing_concept:
            return {"status": "error", "message": f"Alias '{alias}' conflicts with existing concept"}
        
        # Create alias
        new_alias = ConceptAlias(
            concept_id=concept.id,
            alias=alias,
            alias_key=alias_key
        )
        db.add(new_alias)
        db.commit()
        
        return {
            "status": "created",
            "concept_id": concept.id,
            "concept_name": concept.name,
            "alias": alias
        }
    finally:
        db.close()


@mcp.tool()
def memory_add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: Optional[str] = None
) -> dict:
    """
    Create a relationship between two concepts.
    
    Args:
        from_concept: Source concept name
        to_concept: Target concept name
        rel_type: Relationship type (enables/version_of/part_of/related_to/implements/demonstrates)
        weight: Relationship strength 0.0-1.0 (default 0.5)
        description: Optional description of relationship
    
    Returns:
        Status of relationship creation
    """
    _validate_required_text(from_concept, "from_concept", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(to_concept, "to_concept", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(rel_type, "rel_type", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(description, "description", MAX_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        from models import ConceptRelationship
        
        # Valid relationship types
        valid_types = ['enables', 'version_of', 'part_of', 'related_to', 'implements', 'demonstrates']
        if rel_type not in valid_types:
            return {"status": "error", "message": f"Invalid rel_type. Must be one of: {', '.join(valid_types)}"}
        
        # Validate weight
        if not 0.0 <= weight <= 1.0:
            return {"status": "error", "message": "Weight must be between 0.0 and 1.0"}
        
        # Find both concepts (case-insensitive, alias-aware)
        from_key = from_concept.lower()
        to_key = to_concept.lower()
        
        from_c = db.query(Concept).filter(Concept.name_key == from_key).first()
        to_c = db.query(Concept).filter(Concept.name_key == to_key).first()
        
        if not from_c:
            return {"status": "error", "message": f"Source concept '{from_concept}' not found"}
        if not to_c:
            return {"status": "error", "message": f"Target concept '{to_concept}' not found"}
        
        # Check if relationship already exists
        existing = db.query(ConceptRelationship).filter(
            ConceptRelationship.from_concept_id == from_c.id,
            ConceptRelationship.to_concept_id == to_c.id,
            ConceptRelationship.rel_type == rel_type
        ).first()
        
        if existing:
            # Update existing relationship
            existing.weight = weight
            if description:
                existing.description = description
            db.commit()
            return {
                "status": "updated",
                "from": from_c.name,
                "to": to_c.name,
                "rel_type": rel_type,
                "weight": weight
            }
        
        # Create new relationship
        rel = ConceptRelationship(
            from_concept_id=from_c.id,
            to_concept_id=to_c.id,
            rel_type=rel_type,
            weight=weight,
            description=description
        )
        db.add(rel)
        db.commit()
        
        return {
            "status": "created",
            "from": from_c.name,
            "to": to_c.name,
            "rel_type": rel_type,
            "weight": weight
        }
    finally:
        db.close()


@mcp.tool()
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0,
    include_cold: bool = False
) -> dict:
    """
    Get concepts related to a given concept.
    
    Args:
        concept_name: Concept to find relationships for
        rel_type: Optional filter by relationship type
        min_weight: Minimum relationship weight (default 0.0)
        include_cold: Include cold tier concepts
    
    Returns:
        List of related concepts with relationship details
    """
    _validate_required_text(concept_name, "concept_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(rel_type, "rel_type", MAX_SHORT_TEXT_LENGTH)
    _validate_confidence(min_weight, "min_weight")

    db = DB.SessionLocal()
    try:
        from models import ConceptRelationship
        
        # Find the concept
        concept_key = concept_name.lower()
        concept_query = db.query(Concept).filter(Concept.name_key == concept_key)
        if not include_cold:
            concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
        concept = concept_query.first()
        if not concept:
            return {"status": "not_found", "concept": concept_name}
        
        # Get outgoing relationships
        query = db.query(
            ConceptRelationship, Concept
        ).join(
            Concept, ConceptRelationship.to_concept_id == Concept.id
        ).filter(
            ConceptRelationship.from_concept_id == concept.id,
            ConceptRelationship.weight >= min_weight
        )

        if not include_cold:
            query = query.filter(Concept.tier == MemoryTier.hot)
        
        if rel_type:
            query = query.filter(ConceptRelationship.rel_type == rel_type)
        
        outgoing = query.all()
        
        # Get incoming relationships
        query = db.query(
            ConceptRelationship, Concept
        ).join(
            Concept, ConceptRelationship.from_concept_id == Concept.id
        ).filter(
            ConceptRelationship.to_concept_id == concept.id,
            ConceptRelationship.weight >= min_weight
        )

        if not include_cold:
            query = query.filter(Concept.tier == MemoryTier.hot)
        
        if rel_type:
            query = query.filter(ConceptRelationship.rel_type == rel_type)
        
        incoming = query.all()
        
        return {
            "status": "found",
            "concept": concept.name,
            "outgoing": [
                {
                    "to": c.name,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description
                }
                for rel, c in outgoing
            ],
            "incoming": [
                {
                    "from": c.name,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "description": rel.description
                }
                for rel, c in incoming
            ]
        }
    finally:
        db.close()


@mcp.tool()
def memory_update_pattern(
    category: str,
    pattern_name: str,
    pattern_text: str,
    confidence: float = 0.8,
    evidence_observation_ids: Optional[List[int]] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> dict:
    """
    Create or update a pattern (synthesized understanding across observations).
    
    Patterns evolve as understanding grows. This tool performs an upsert:
    - If pattern exists (by category + pattern_name), updates it
    - If pattern doesn't exist, creates it
    
    Args:
        category: Pattern category/domain
        pattern_name: Unique name within category
        pattern_text: The synthesized pattern description (gets embedded)
        confidence: Confidence level 0.0-1.0 (default 0.8)
        evidence_observation_ids: List of observation IDs supporting this pattern
        ai_name: Optional AI instance name
        ai_platform: Optional AI platform
        conversation_id: Optional conversation UUID
    
    Returns:
        Pattern with status (created/updated)
    """
    _validate_required_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_required_text(pattern_name, "pattern_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(pattern_text, "pattern_text", MAX_TEXT_LENGTH)
    _validate_confidence(confidence, "confidence")
    _validate_list(evidence_observation_ids, "evidence_observation_ids", MAX_LIST_ITEMS)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        # Get AI instance and session if provided
        ai_instance_id = None
        session_id = None
        
        if ai_name and ai_platform:
            ai_instance = get_or_create_ai_instance(db, ai_name, ai_platform)
            ai_instance_id = ai_instance.id
            
            if conversation_id:
                session = get_or_create_session(db, conversation_id, ai_instance_id=ai_instance_id)
                session_id = session.id
        
        # Check if pattern exists
        existing = db.query(Pattern).filter(
            Pattern.category == category,
            Pattern.pattern_name == pattern_name
        ).first()
        
        if existing:
            # Update existing pattern
            existing.pattern_text = pattern_text
            existing.confidence = confidence
            existing.evidence_observation_ids = evidence_observation_ids or []
            existing.last_updated = datetime.utcnow()
            if ai_instance_id:
                existing.ai_instance_id = ai_instance_id
            if session_id:
                existing.session_id = session_id
            
            db.commit()
            db.refresh(existing)
            
            # Update embedding (if enabled)
            _store_embedding(db, "pattern", existing.id, pattern_text, replace=True)
            
            return {
                "status": "updated",
                "id": existing.id,
                "category": category,
                "pattern_name": pattern_name,
                "confidence": confidence
            }
        else:
            # Create new pattern
            pattern = Pattern(
                category=category,
                pattern_name=pattern_name,
                pattern_text=pattern_text,
                confidence=confidence,
                evidence_observation_ids=evidence_observation_ids or [],
                ai_instance_id=ai_instance_id,
                session_id=session_id
            )
            db.add(pattern)
            db.commit()
            db.refresh(pattern)
            
            # Generate and store embedding (if enabled)
            _store_embedding(db, "pattern", pattern.id, pattern_text)
            
            return {
                "status": "created",
                "id": pattern.id,
                "category": category,
                "pattern_name": pattern_name,
                "confidence": confidence
            }
    finally:
        db.close()


@mcp.tool()
def memory_get_pattern(category: str, pattern_name: str, include_cold: bool = False) -> dict:
    """
    Get a specific pattern by category and name.
    
    Args:
        category: Pattern category
        pattern_name: Pattern name within category
        include_cold: Include cold tier records
    
    Returns:
        Pattern details or not_found status
    """
    _validate_required_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_required_text(pattern_name, "pattern_name", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        pattern_query = db.query(Pattern).filter(
            Pattern.category == category,
            Pattern.pattern_name == pattern_name
        )
        if not include_cold:
            pattern_query = pattern_query.filter(Pattern.tier == MemoryTier.hot)
        pattern = pattern_query.first()
        
        if not pattern:
            return {
                "status": "not_found",
                "category": category,
                "pattern_name": pattern_name
            }
        
        if pattern.tier == MemoryTier.hot:
            _apply_fetch_bump(pattern, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "status": "found",
            "id": pattern.id,
            "category": category,
            "pattern_name": pattern_name,
            "pattern_text": pattern.pattern_text,
            "confidence": pattern.confidence,
            "evidence_observation_ids": pattern.evidence_observation_ids,
            "last_updated": pattern.last_updated.isoformat() if pattern.last_updated else None,
            "access_count": pattern.access_count
        }
    finally:
        db.close()


@mcp.tool()
def memory_patterns(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 20,
    include_cold: bool = False
) -> dict:
    """
    List patterns with optional filtering by category and confidence.
    
    Args:
        category: Optional category filter
        min_confidence: Minimum confidence threshold (default 0.0)
        limit: Maximum results (default 20)
        include_cold: Include cold tier records
    
    Returns:
        List of matching patterns
    """
    _validate_optional_text(category, "category", MAX_DOMAIN_LENGTH)
    _validate_confidence(min_confidence, "min_confidence")
    _validate_limit(limit, "limit", MAX_RESULT_LIMIT)

    db = DB.SessionLocal()
    try:
        query = db.query(Pattern)
        
        if category:
            query = query.filter(Pattern.category == category)
        if min_confidence > 0:
            query = query.filter(Pattern.confidence >= min_confidence)
        if not include_cold:
            query = query.filter(Pattern.tier == MemoryTier.hot)
        
        results = query.order_by(desc(Pattern.last_updated)).limit(limit).all()

        if not include_cold:
            for pattern in results:
                _apply_fetch_bump(pattern, SCORE_BUMP_ALPHA)
            db.commit()
        
        return {
            "count": len(results),
            "filters": {
                "category": category,
                "min_confidence": min_confidence
            },
            "results": [
                {
                    "id": p.id,
                    "category": p.category,
                    "pattern_name": p.pattern_name,
                    "pattern_text": p.pattern_text,
                    "confidence": p.confidence,
                    "evidence_count": len(p.evidence_observation_ids) if p.evidence_observation_ids else 0,
                    "last_updated": p.last_updated.isoformat() if p.last_updated else None
                }
                for p in results
            ]
        }
    finally:
        db.close()


# =============================================================================
# Self-Documentation Tools
# =============================================================================

SPEC_VERSION = "0.1.0"

RECOMMENDED_DOMAINS = [
    "technical_milestone",
    "major_milestone",
    "project_context",
    "system_architecture",
    "interaction_patterns",
    "system_behavior",
    "identity",
    "preferences",
    "decisions",
]

CONCEPT_TYPES = [
    "project",
    "framework",
    "component",
    "construct",
    "theory",
]

RELATIONSHIP_TYPES = [
    "enables",
    "version_of",
    "part_of",
    "related_to",
    "implements",
    "demonstrates",
]

CONFIDENCE_GUIDE = {
    "1.0": "Direct observation, absolute certainty",
    "0.95-0.99": "Very high confidence, strong evidence",
    "0.85-0.94": "High confidence, solid evidence",
    "0.70-0.84": "Good confidence, some uncertainty",
    "0.50-0.69": "Moderate confidence, competing interpretations",
    "<0.50": "Speculative, weak evidence",
}


@mcp.tool()
def memory_user_guide(
    format: str = "markdown",
    verbosity: str = "short"
) -> dict:
    """
    Get self-documentation for MemoryGate system.
    
    Returns usage guide, schemas, recommended practices, and examples
    so AI agents can bootstrap themselves without manual configuration.
    
    Args:
        format: Output format (markdown or json)
        verbosity: short (recommended) or verbose (comprehensive)
    
    Returns:
        Dictionary with spec_version, guide content, structured metadata
    """
    if format not in {"markdown", "json"}:
        raise ValueError("format must be 'markdown' or 'json'")
    if verbosity not in {"short", "verbose"}:
        raise ValueError("verbosity must be 'short' or 'verbose'")
    
    guide_content = """# MemoryGate User Guide

**Version:** {spec_version}

## Purpose

MemoryGate is a persistent Memory-as-a-Service system for AI agents. It provides:
- **Observations**: Discrete facts with confidence and evidence
- **Patterns**: Synthesized understanding across observations  
- **Concepts**: Canonical entities in a knowledge graph
- **Documents**: References to external content (not full copies)
- **Semantic search**: Unified vector search across all types

## Core Workflow

### 1. Initialize Session
Always start new conversations with:
```python
memory_init_session(
    conversation_id="unique-uuid",
    title="Description of conversation",
    ai_name="YourName",
    ai_platform="YourPlatform"
)
```

### 2. Search Before Answering
Use semantic search liberally (~50ms, fast):
```python
memory_search(query="relevant topic", limit=5)
```

### 3. Store New Information
**Observations** - discrete facts:
```python
memory_store(
    observation="User prefers TypeScript",
    confidence=0.9,
    domain="preferences",
    evidence=["Stated explicitly"]
)
```

**Concepts** - new frameworks/projects:
```python
memory_store_concept(
    name="MemoryGate",
    concept_type="project",
    description="Memory service for AI agents"
)
```

**Patterns** - synthesized understanding:
```python
memory_update_pattern(
    category="interaction_patterns",
    pattern_name="direct_communication",
    pattern_text="User values directness",
    confidence=0.85
)
```

## Critical Invariants

1. **Concept names are case-insensitive**
2. **Aliases prevent fragmentation**
3. **Patterns are upserts** - safe to call repeatedly
4. **Documents store references, not content**
5. **Search is primary tool** - search first, then answer

## Recommended Domains
{domains}

## Confidence Levels
{confidence}

## Concept Types
{concept_types}

## Relationship Types
{relationship_types}
""".format(
        spec_version=SPEC_VERSION,
        domains="\n".join(f"- `{d}`" for d in RECOMMENDED_DOMAINS),
        confidence="\n".join(f"- **{k}**: {v}" for k, v in CONFIDENCE_GUIDE.items()),
        concept_types="\n".join(f"- `{ct}`" for ct in CONCEPT_TYPES),
        relationship_types="\n".join(f"- `{rt}`" for rt in RELATIONSHIP_TYPES),
    )
    
    result = {
        "spec_version": SPEC_VERSION,
        "recommended_domains": RECOMMENDED_DOMAINS,
        "concept_types": CONCEPT_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "confidence_guide": CONFIDENCE_GUIDE,
    }
    
    if format == "markdown":
        result["guide"] = guide_content
    else:  # json
        result["guide"] = {
            "purpose": "Memory-as-a-Service for AI agents",
            "core_workflow": [
                "Initialize session with memory_init_session()",
                "Search with memory_search() before answering",
                "Store new info with memory_store/memory_store_concept/memory_update_pattern",
            ],
            "critical_invariants": [
                "Concept names are case-insensitive",
                "Aliases prevent fragmentation",
                "Patterns are upserts",
                "Documents store references not content",
                "Search is primary tool",
            ],
        }
    
    return result


@mcp.tool()
def memory_bootstrap(ai_name: Optional[str] = None, ai_platform: Optional[str] = None) -> dict:
    """
    Stateful bootstrap for AI agents - tells you your relationship status with MemoryGate.
    
    Returns compatibility info, connection history, and getting started guide.
    The system tells you what it already knows about you.
    
    Args:
        ai_name: Your AI instance name (e.g., "Kee", "Hexy")
        ai_platform: Your platform (e.g., "Claude", "ChatGPT")
    
    Returns:
        Relationship status, version info, and usage guide
    """
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        # Check if this AI instance has history
        connection_status = {
            "is_new_instance": True,
            "first_seen": None,
            "last_seen": None,
            "session_count": 0,
            "total_observations": 0,
        }
        
        if ai_name and ai_platform:
            # Query for this AI instance's history
            ai_instance_query = db.query(AIInstance).filter(
                AIInstance.name == ai_name,
                AIInstance.platform == ai_platform
            ).first()
            
            if ai_instance_query:
                # Found existing instance
                connection_status["is_new_instance"] = False
                connection_status["first_seen"] = ai_instance_query.created_at.isoformat()
                
                # Get last session to determine last_seen
                last_session = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id
                ).order_by(Session.started_at.desc()).first()
                
                if last_session:
                    connection_status["last_seen"] = last_session.started_at.isoformat()
                else:
                    connection_status["last_seen"] = ai_instance_query.created_at.isoformat()
                
                # Count sessions
                session_count = db.query(Session).filter(
                    Session.ai_instance_id == ai_instance_query.id
                ).count()
                connection_status["session_count"] = session_count
                
                # Count observations contributed by this AI
                obs_count = db.query(Observation).filter(
                    Observation.ai_instance_id == ai_instance_query.id
                ).count()
                connection_status["total_observations"] = obs_count
        
        return {
            "spec_version": SPEC_VERSION,
            "role": "agent_bootstrap",
            "connection_status": connection_status,
            "assumptions": [
                "You are an AI agent using MCP",
                "You may reconnect across sessions",
                "You should externalize durable memory here",
                "Memory persists between your instances",
                "Other AI agents may also use this memory"
            ],
            "compatibility": {
                "spec_version": SPEC_VERSION,
                "breaking_changes_since": None,  # Future: track version bumps
                "recommended_action": "Use normally" if not connection_status["is_new_instance"] else "Initialize new session"
            },
            "first_steps": [
                {
                    "step": 1,
                    "action": "Initialize session" if connection_status["is_new_instance"] else "Initialize this session",
                    "tool": "memory_init_session",
                    "required_params": ["conversation_id", "title", "ai_name", "ai_platform"],
                    "note": "Creates session record and updates last_seen" if not connection_status["is_new_instance"] else "Registers you as new AI instance"
                },
                {
                    "step": 2,
                    "action": "Search for relevant context",
                    "tool": "memory_search",
                    "params": {"query": "topic keywords", "limit": 5},
                    "note": f"You have {connection_status['total_observations']} observations in the system" if connection_status["total_observations"] > 0 else "System is empty - you'll build memory as you go"
                },
                {
                    "step": 3,
                    "action": "Store new observations",
                    "tool": "memory_store",
                    "params": {
                        "observation": "What you learned",
                        "confidence": 0.8,
                        "domain": "appropriate_domain",
                        "evidence": ["supporting facts"],
                    },
                },
            ],
            "critical_rules": [
                "ALWAYS call memory_init_session() at conversation start",
                "Search liberally - it's fast (~50ms)",
                "Concept names are case-insensitive",
                "Use confidence weights honestly (0.0-1.0)",
                "Documents are references only (Google Drive = canonical storage)"
            ],
            "recommended_domains": RECOMMENDED_DOMAINS,
            "confidence_guide": CONFIDENCE_GUIDE,
            "next_step": "Call memory_user_guide() for full documentation",
        }
    finally:
        db.close()


# =============================================================================
# FastAPI App
# =============================================================================

# Create MCP ASGI app
mcp_app = mcp.http_app(
    path="/",
    transport="sse",
    stateless_http=True,
    json_response=True,
)


# =============================================================================
# Pure ASGI wrapper: Normalize /mcp to /mcp/ without buffering
# =============================================================================

class SlashNormalizerASGI:
    """Pure ASGI middleware - no response buffering, SSE-safe."""
    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = dict(scope)  # Make mutable copy
            scope["path"] = "/mcp/"
        await self.wrapped_app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    global cleanup_task, retention_task
    init_db()
    init_http_client()
    if CLEANUP_INTERVAL_SECONDS > 0:
        await _run_cleanup_once()
        cleanup_task = asyncio.create_task(_cleanup_loop())
    if RETENTION_TICK_SECONDS > 0:
        retention_task = asyncio.create_task(_retention_loop())
    yield
    if retention_task:
        retention_task.cancel()
        try:
            await retention_task
        except asyncio.CancelledError:
            pass
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    cleanup_http_client()
    if rate_limiter:
        await rate_limiter.close()
    if DB.engine:
        DB.engine.dispose()


app = FastAPI(title="MemoryGate", redirect_slashes=False, lifespan=lifespan)

# Request size limits (keep CORS outermost to add headers on 413 responses)
app.add_middleware(
    RequestSizeLimitMiddleware,
    config=request_size_config,
)

# Rate limiting middleware (outer CORS will still add headers on 429 responses)
app.add_middleware(
    RateLimitMiddleware,
    limiter=rate_limiter,
    config=rate_limit_config,
)

# Security headers (applies to all responses)
app.add_middleware(
    SecurityHeadersMiddleware,
    config=security_headers_config,
)

# Optional host allowlist for production deployments
trusted_hosts_env = os.environ.get("TRUSTED_HOSTS", "")
trusted_hosts = [host.strip() for host in trusted_hosts_env.split(",") if host.strip()]
if trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts,
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("FRONTEND_URL", "http://localhost:3000"),
        "http://localhost:3000",
        "https://memorygate.ai",
        "https://www.memorygate.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_db_health() -> dict:
    if DB.engine is None:
        return {"ok": False, "error": "db_not_initialized"}

    try:
        with DB.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            ext_version = None
            pgvector_installed = True
            if DB_BACKEND == "postgres" and VECTOR_BACKEND_EFFECTIVE == "pgvector":
                ext_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
                pgvector_installed = bool(ext_version)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    current_rev, head_rev = _get_schema_revisions(DB.engine)
    schema_ok = head_rev is None or current_rev == head_rev
    return {
        "ok": schema_ok,
        "pgvector_installed": pgvector_installed,
        "pgvector_version": ext_version,
        "schema_revision": current_rev,
        "schema_expected": head_rev,
        "schema_up_to_date": schema_ok,
    }

# Mount OAuth discovery and authorization routes (for Claude Desktop MCP)
from oauth_discovery import router as oauth_discovery_router
app.include_router(oauth_discovery_router)

# Mount OAuth user management routes
from oauth_routes import router as auth_router
from mcp_auth_gate import MCPAuthGateASGI
app.include_router(auth_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    db_health = _check_db_health()
    vector_required = DB_BACKEND == "postgres" and VECTOR_BACKEND_EFFECTIVE == "pgvector"
    if not db_health.get("ok") or (vector_required and not db_health.get("pgvector_installed")):
        raise HTTPException(status_code=503, detail=db_health)

    return {
        "status": "healthy",
        "service": "MemoryGate",
        "version": "0.1.0",
        "instance_id": os.environ.get("MEMORYGATE_INSTANCE_ID", "memorygate-1"),
        "tenant_mode": TENANCY_MODE,
        "database": db_health,
    }


@app.get("/health/deps")
async def health_deps():
    """Dependency health checks (optional embedding provider probe)."""
    db_health = _check_db_health()
    vector_required = DB_BACKEND == "postgres" and VECTOR_BACKEND_EFFECTIVE == "pgvector"
    if not db_health.get("ok") or (vector_required and not db_health.get("pgvector_installed")):
        raise HTTPException(status_code=503, detail={"database": db_health})

    breaker_status = embedding_circuit_breaker.status()
    embedding_status = {
        "status": "unknown",
        "provider": EMBEDDING_PROVIDER,
        "circuit_breaker": breaker_status,
        "checked": False,
    }

    if EMBEDDING_PROVIDER == "none":
        embedding_status["status"] = "disabled"
    elif breaker_status.get("open"):
        embedding_status["status"] = "cooldown"
    elif EMBEDDING_HEALTHCHECK_ENABLED:
        embedding_status["checked"] = True
        start = time.time()
        try:
            embed_text_sync("healthcheck")
            embedding_status["status"] = "ok"
            embedding_status["latency_ms"] = int((time.time() - start) * 1000)
        except EmbeddingProviderError as exc:
            embedding_status["status"] = "error"
            embedding_status["error"] = str(exc)
    else:
        embedding_status["status"] = "skipped"

    return {
        "status": "healthy",
        "service": "MemoryGate",
        "tenant_mode": TENANCY_MODE,
        "database": db_health,
        "embedding_provider": embedding_status,
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "MemoryGate",
        "version": "0.1.0",
        "description": "Persistent Memory-as-a-Service for AI Agents",
        "tenant_mode": TENANCY_MODE,
        "embedding_model": EMBEDDING_MODEL,
        "endpoints": {
            "health": "/health",
            "health_deps": "/health/deps",
            "mcp": "/mcp",
            "auth": {
                "client_credentials": "/auth/client",
                "login_google": "/auth/login/google",
                "login_github": "/auth/login/github",
                "me": "/auth/me",
                "api_keys": "/auth/api-keys"
            }
        }
    }


# Mount MCP app at /mcp/ with auth gate (pass DB class for dynamic lookup)
app.mount("/mcp/", MCPAuthGateASGI(mcp_app, lambda: DB.SessionLocal))


# =============================================================================
# ASGI Application (module-level for production deployment)
# =============================================================================

# Wrap entire app with slash normalizer to handle /mcp -> /mcp/
asgi_app = SlashNormalizerASGI(app)


# =============================================================================
# Main (for local development only)
# =============================================================================

if __name__ == "__main__":
    print("MemoryGate starting...")
    uvicorn.run(asgi_app, host="0.0.0.0", port=8080)
