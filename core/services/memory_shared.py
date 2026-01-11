"""
Shared helpers and configuration for memory services.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from functools import wraps
from datetime import datetime
from typing import Optional, List, Callable

import httpx
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

import core.config as config
from core.errors import EmbeddingProviderError, ValidationIssue
from core.models import (
    Embedding,
    MemorySummary,
    MemoryTombstone,
    MemoryTier,
    TombstoneAction,
    MEMORY_MODELS,
)
from core.services.retention_service import (
    apply_fetch_bump,
    clamp_score,
    apply_floor,
)
from core.validators import (
    validate_required_text as _validate_required_text,
    validate_optional_text as _validate_optional_text,
    validate_limit as _validate_limit,
    validate_confidence as _validate_confidence,
    validate_list as _validate_list,
    validate_string_list as _validate_string_list,
    validate_metadata as _validate_metadata,
    validate_embedding_text as _validate_embedding_text,
    validate_evidence_observation_ids as _validate_evidence_observation_ids,
)

# =============================================================================
# Configuration
# =============================================================================

logger = config.logger

OPENAI_API_KEY = config.OPENAI_API_KEY
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM

MAX_RESULT_LIMIT = config.MAX_RESULT_LIMIT
MAX_QUERY_LENGTH = config.MAX_QUERY_LENGTH
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
MAX_SHORT_TEXT_LENGTH = config.MAX_SHORT_TEXT_LENGTH
MAX_DOMAIN_LENGTH = config.MAX_DOMAIN_LENGTH
MAX_TITLE_LENGTH = config.MAX_TITLE_LENGTH
MAX_URL_LENGTH = config.MAX_URL_LENGTH
MAX_DOC_TYPE_LENGTH = config.MAX_DOC_TYPE_LENGTH
MAX_CONCEPT_TYPE_LENGTH = config.MAX_CONCEPT_TYPE_LENGTH
MAX_STATUS_LENGTH = config.MAX_STATUS_LENGTH
MAX_METADATA_BYTES = config.MAX_METADATA_BYTES
MAX_LIST_ITEMS = config.MAX_LIST_ITEMS
MAX_LIST_ITEM_LENGTH = config.MAX_LIST_ITEM_LENGTH
MAX_TAG_ITEMS = config.MAX_TAG_ITEMS
MAX_RELATIONSHIP_ITEMS = config.MAX_RELATIONSHIP_ITEMS

EMBEDDING_TIMEOUT_SECONDS = config.EMBEDDING_TIMEOUT_SECONDS
EMBEDDING_RETRY_MAX = config.EMBEDDING_RETRY_MAX
EMBEDDING_RETRY_BACKOFF_SECONDS = config.EMBEDDING_RETRY_BACKOFF_SECONDS
EMBEDDING_RETRY_JITTER_SECONDS = config.EMBEDDING_RETRY_JITTER_SECONDS
EMBEDDING_FAILURE_THRESHOLD = config.EMBEDDING_FAILURE_THRESHOLD
EMBEDDING_COOLDOWN_SECONDS = config.EMBEDDING_COOLDOWN_SECONDS
EMBEDDING_PROVIDER = config.EMBEDDING_PROVIDER
EMBEDDING_BACKFILL_INTERVAL_SECONDS = config.EMBEDDING_BACKFILL_INTERVAL_SECONDS
EMBEDDING_BACKFILL_BATCH_LIMIT = config.EMBEDDING_BACKFILL_BATCH_LIMIT

SCORE_BUMP_ALPHA = config.SCORE_BUMP_ALPHA
REHYDRATE_BUMP_ALPHA = config.REHYDRATE_BUMP_ALPHA
SCORE_CLAMP_MIN = config.SCORE_CLAMP_MIN
SCORE_CLAMP_MAX = config.SCORE_CLAMP_MAX
SUMMARY_TRIGGER_SCORE = config.SUMMARY_TRIGGER_SCORE
COLD_SEARCH_ENABLED = config.COLD_SEARCH_ENABLED
ARCHIVE_LIMIT_DEFAULT = config.ARCHIVE_LIMIT_DEFAULT
ARCHIVE_LIMIT_MAX = config.ARCHIVE_LIMIT_MAX
REHYDRATE_LIMIT_MAX = config.REHYDRATE_LIMIT_MAX
TOMBSTONES_ENABLED = config.TOMBSTONES_ENABLED
SUMMARY_MAX_LENGTH = config.SUMMARY_MAX_LENGTH

http_client = None  # Reusable HTTP client for OpenAI API


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
    return config.DB_BACKEND == "postgres" and config.VECTOR_BACKEND_EFFECTIVE == "pgvector"


def _store_embedding(
    db,
    source_type: str,
    source_id: int,
    text_value: str,
    replace: bool = False,
) -> bool:
    if not _vector_search_enabled():
        return False
    try:
        embedding_vector = embed_text_sync(text_value)
    except EmbeddingProviderError:
        logger.warning("Embedding unavailable; skipping embedding store")
        return False
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
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return False
    return True


# =============================================================================
# Helper Functions
# =============================================================================

def _tool_error_payload(tool_name: str, exc: ValidationIssue) -> dict:
    return {
        "status": "error",
        "error_type": "validation_error",
        "tool": tool_name,
        "field": exc.field,
        "message": str(exc),
    }


def _log_validation_issue(tool_name: str, exc: ValidationIssue, warn: bool = False) -> None:
    payload = {
        "tool": tool_name,
        "field": exc.field,
        "error_type": exc.error_type,
        "detail": str(exc),
    }
    if warn:
        logger.warning("tool_validation_error", extra=payload)
    else:
        logger.info("tool_validation_error", extra=payload)


def _tool_error_handler(fn: Callable[..., dict]) -> Callable[..., dict]:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValidationIssue as exc:
            _log_validation_issue(fn.__name__, exc, warn=False)
            return _tool_error_payload(fn.__name__, exc)
        except ValueError as exc:
            issue = ValidationIssue(str(exc), field="unknown", error_type="value_error")
            _log_validation_issue(fn.__name__, issue, warn=True)
            return _tool_error_payload(fn.__name__, issue)
    return wrapper


def service_tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    return _tool_error_handler(fn)


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
    logger.warning("Embedding provider unavailable")
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
            try:
                return mem_type.strip().lower(), int(mem_id)
            except ValueError as exc:
                raise ValidationIssue(
                    "memory_ids must include a numeric id",
                    field="memory_ids",
                    error_type="invalid_id",
                ) from exc
        try:
            return "observation", int(value)
        except ValueError as exc:
            raise ValidationIssue(
                "memory_ids must include a numeric id",
                field="memory_ids",
                error_type="invalid_id",
            ) from exc
    raise ValidationIssue(
        "memory_ids must be int or 'type:id' string",
        field="memory_ids",
        error_type="invalid_type",
    )


def _collect_records_by_refs(db, refs: list[tuple[str, int]]) -> list[tuple[str, object]]:
    records = []
    for mem_type, mem_id in refs:
        model = MEMORY_MODELS.get(mem_type)
        if not model:
            raise ValidationIssue(
                f"Unknown memory type: {mem_type}",
                field="memory_ids",
                error_type="invalid_type",
            )
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


