"""
Memory CRUD/search services and embedding helpers.
"""

import asyncio
import json
import random
import threading
import time
from functools import wraps
from datetime import datetime
from typing import Optional, List, Callable

import httpx
from fastapi import HTTPException
from sqlalchemy import text, func, desc, and_, or_
from sqlalchemy.exc import IntegrityError

import core.config as config
from core.context import RequestContext
from core.db import DB
from core.errors import EmbeddingProviderError, ValidationIssue
from core.models import (
    AIInstance,
    Session,
    Observation,
    Pattern,
    Concept,
    ConceptAlias,
    ConceptRelationship,
    Document,
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


def get_or_create_ai_instance(db, name: str, platform: str) -> AIInstance:
    """Get or create an AI instance by name."""
    instance = db.query(AIInstance).filter(AIInstance.name == name).first()
    if not instance:
        instance = AIInstance(name=name, platform=platform)
        db.add(instance)
        try:
            db.commit()
            db.refresh(instance)
        except IntegrityError as exc:
            db.rollback()
            instance = db.query(AIInstance).filter(AIInstance.name == name).first()
            if instance is None:
                raise exc
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
        try:
            db.commit()
            db.refresh(session)
        except IntegrityError as exc:
            db.rollback()
            session = db.query(Session).filter(Session.conversation_id == conversation_id).first()
            if session is None:
                raise exc
    elif title and session.title != title:
        session.title = title
        session.last_active = datetime.utcnow()
        db.commit()
    return session


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


def _run_embedding_backfill() -> dict:
    if DB.SessionLocal is None:
        return {"status": "skipped", "reason": "db_not_initialized"}
    if not _vector_search_enabled():
        return {"status": "skipped", "reason": "vector_disabled"}
    if EMBEDDING_PROVIDER == "none":
        return {"status": "skipped", "reason": "embedding_disabled"}
    if embedding_circuit_breaker.is_open():
        return {"status": "skipped", "reason": "circuit_open"}
    if EMBEDDING_BACKFILL_BATCH_LIMIT <= 0:
        return {"status": "skipped", "reason": "batch_limit_disabled"}

    db = DB.SessionLocal()
    processed = 0
    backfilled = 0
    skipped = 0
    try:
        candidates = [
            ("observation", Observation, "observation"),
            ("pattern", Pattern, "pattern_text"),
            ("concept", Concept, "description"),
            ("document", Document, "content_summary"),
        ]
        for source_type, model, field in candidates:
            if processed >= EMBEDDING_BACKFILL_BATCH_LIMIT:
                break
            missing = (
                db.query(model)
                .outerjoin(
                    Embedding,
                    and_(
                        Embedding.source_type == source_type,
                        Embedding.source_id == model.id,
                    ),
                )
                .filter(Embedding.source_id.is_(None))
                .order_by(model.id.asc())
                .limit(EMBEDDING_BACKFILL_BATCH_LIMIT - processed)
                .all()
            )
            for record in missing:
                if processed >= EMBEDDING_BACKFILL_BATCH_LIMIT:
                    break
                if embedding_circuit_breaker.is_open():
                    return {
                        "status": "skipped",
                        "reason": "circuit_open",
                        "processed": processed,
                        "backfilled": backfilled,
                        "skipped_count": skipped,
                    }
                processed += 1
                text_value = getattr(record, field, None)
                if not text_value:
                    skipped += 1
                    continue
                if _store_embedding(db, source_type, record.id, text_value):
                    backfilled += 1
                else:
                    skipped += 1
        return {
            "status": "ok",
            "processed": processed,
            "backfilled": backfilled,
            "skipped_count": skipped,
        }
    finally:
        db.close()


async def _embedding_backfill_loop() -> None:
    if EMBEDDING_BACKFILL_INTERVAL_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(EMBEDDING_BACKFILL_INTERVAL_SECONDS)
        try:
            stats = await asyncio.to_thread(_run_embedding_backfill)
            if stats.get("status") == "ok" and stats.get("backfilled", 0) > 0:
                logger.info("Embedding backfill complete", extra=stats)
        except Exception as exc:
            logger.warning(f"Embedding backfill error: {exc}")


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

@service_tool
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
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


@service_tool
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
    bump_score: bool = False,
    context: Optional[RequestContext] = None,
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
    _validate_string_list(tags, "tags", MAX_TAG_ITEMS, MAX_LIST_ITEM_LENGTH)

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


@service_tool
def archive_memory(
    memory_ids: Optional[List[str]] = None,
    summary_ids: Optional[List[int]] = None,
    cluster_ids: Optional[List[str]] = None,
    threshold: Optional[dict] = None,
    mode: str = "archive_and_tombstone",
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    dry_run: bool = True,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    context: Optional[RequestContext] = None,
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
        already_archived_ids = []
        tombstones_written = 0
        summaries_created = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.hot:
                already_archived_ids.append(_serialize_memory_id(mem_type, record.id))
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
                already_archived_ids.append(f"summary:{summary.id}")
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
            "already_archived_count": len(already_archived_ids),
            "already_archived_ids": already_archived_ids,
            "tombstones_written": tombstones_written,
            "summaries_created": summaries_created,
        }
    finally:
        db.close()


@service_tool
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
    bump_score: bool = True,
    context: Optional[RequestContext] = None,
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
        already_hot_ids = []
        tombstones_written = 0

        for mem_type, record in candidates:
            if record.tier != MemoryTier.cold:
                already_hot_ids.append(_serialize_memory_id(mem_type, record.id))
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
                already_hot_ids.append(f"summary:{summary.id}")
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
            "already_hot_count": len(already_hot_ids),
            "already_hot_ids": already_hot_ids,
            "tombstones_written": tombstones_written,
        }
    finally:
        db.close()


@service_tool
def list_archive_candidates(
    below_score: float = SUMMARY_TRIGGER_SCORE,
    limit: int = ARCHIVE_LIMIT_DEFAULT,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_store(
    observation: str,
    confidence: float = 0.8,
    domain: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    ai_name: str = "Unknown",
    ai_platform: str = "Unknown",
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_recall(
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    ai_name: Optional[str] = None,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_stats(context: Optional[RequestContext] = None) -> dict:
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


@service_tool
def memory_init_session(
    conversation_id: str,
    title: str,
    ai_name: str,
    ai_platform: str,
    source_url: Optional[str] = None,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_store_document(
    title: str,
    doc_type: str,
    url: str,
    content_summary: str,
    key_concepts: Optional[List[str]] = None,
    publication_date: Optional[str] = None,
    metadata: Optional[dict] = None,
    context: Optional[RequestContext] = None,
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
                logger.warning("Invalid publication_date format")
        
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


@service_tool
def memory_store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[dict] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    context: Optional[RequestContext] = None,
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


def _resolve_concept_by_name(db, name: str, include_cold: bool) -> Optional[Concept]:
    name_key = name.lower()
    concept_query = db.query(Concept).filter(Concept.name_key == name_key)
    if not include_cold:
        concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
    concept = concept_query.first()

    if not concept:
        alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == name_key).first()
        if alias:
            concept_query = db.query(Concept).filter(Concept.id == alias.concept_id)
            if not include_cold:
                concept_query = concept_query.filter(Concept.tier == MemoryTier.hot)
            concept = concept_query.first()

    return concept


@service_tool
def memory_get_concept(
    name: str,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
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
        concept = _resolve_concept_by_name(db, name, include_cold)
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


@service_tool
def memory_add_concept_alias(
    concept_name: str,
    alias: str,
    context: Optional[RequestContext] = None,
) -> dict:
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


@service_tool
def memory_add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: Optional[str] = None,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
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
        
        # Find the concept (alias-aware)
        concept = _resolve_concept_by_name(db, concept_name, include_cold)
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


@service_tool
def memory_update_pattern(
    category: str,
    pattern_name: str,
    pattern_text: str,
    confidence: float = 0.8,
    evidence_observation_ids: Optional[List[int]] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    conversation_id: Optional[str] = None,
    context: Optional[RequestContext] = None,
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
    _validate_list(evidence_observation_ids, "evidence_observation_ids", MAX_RELATIONSHIP_ITEMS)
    _validate_optional_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        _validate_evidence_observation_ids(db, evidence_observation_ids)
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


@service_tool
def memory_get_pattern(
    category: str,
    pattern_name: str,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
) -> dict:
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


@service_tool
def memory_patterns(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 20,
    include_cold: bool = False,
    context: Optional[RequestContext] = None,
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


@service_tool
def memory_user_guide(
    format: str = "markdown",
    verbosity: str = "short",
    context: Optional[RequestContext] = None,
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
        raise ValidationIssue(
            "format must be 'markdown' or 'json'",
            field="format",
            error_type="invalid_value",
        )
    if verbosity not in {"short", "verbose"}:
        raise ValidationIssue(
            "verbosity must be 'short' or 'verbose'",
            field="verbosity",
            error_type="invalid_value",
        )
    
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

## Limits (defaults)
- **Search result limit**: {max_result_limit} (`MEMORYGATE_MAX_RESULT_LIMIT`)
- **Query length**: {max_query_length} chars (`MEMORYGATE_MAX_QUERY_LENGTH`)
- **Text length**: {max_text_length} chars (`MEMORYGATE_MAX_TEXT_LENGTH`)
- **Short text length**: {max_short_text_length} chars (`MEMORYGATE_MAX_SHORT_TEXT_LENGTH`)
- **Evidence list size**: {max_relationship_items} observation IDs (`MEMORYGATE_MAX_RELATIONSHIP_ITEMS`)
- **List sizes**: {max_list_items} items (`MEMORYGATE_MAX_LIST_ITEMS`)
- **Metadata size**: {max_metadata_bytes} bytes (`MEMORYGATE_MAX_METADATA_BYTES`)

Limits can be configured per deployment via the environment variables above.
""".format(
        spec_version=SPEC_VERSION,
        domains="\n".join(f"- `{d}`" for d in RECOMMENDED_DOMAINS),
        confidence="\n".join(f"- **{k}**: {v}" for k, v in CONFIDENCE_GUIDE.items()),
        concept_types="\n".join(f"- `{ct}`" for ct in CONCEPT_TYPES),
        relationship_types="\n".join(f"- `{rt}`" for rt in RELATIONSHIP_TYPES),
        max_result_limit=MAX_RESULT_LIMIT,
        max_query_length=MAX_QUERY_LENGTH,
        max_text_length=MAX_TEXT_LENGTH,
        max_short_text_length=MAX_SHORT_TEXT_LENGTH,
        max_relationship_items=MAX_RELATIONSHIP_ITEMS,
        max_list_items=MAX_LIST_ITEMS,
        max_metadata_bytes=MAX_METADATA_BYTES,
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
            "limits": {
                "max_result_limit": MAX_RESULT_LIMIT,
                "max_query_length": MAX_QUERY_LENGTH,
                "max_text_length": MAX_TEXT_LENGTH,
                "max_short_text_length": MAX_SHORT_TEXT_LENGTH,
                "max_relationship_items": MAX_RELATIONSHIP_ITEMS,
                "max_list_items": MAX_LIST_ITEMS,
                "max_metadata_bytes": MAX_METADATA_BYTES,
            },
            "limit_env_vars": {
                "MEMORYGATE_MAX_RESULT_LIMIT": MAX_RESULT_LIMIT,
                "MEMORYGATE_MAX_QUERY_LENGTH": MAX_QUERY_LENGTH,
                "MEMORYGATE_MAX_TEXT_LENGTH": MAX_TEXT_LENGTH,
                "MEMORYGATE_MAX_SHORT_TEXT_LENGTH": MAX_SHORT_TEXT_LENGTH,
                "MEMORYGATE_MAX_RELATIONSHIP_ITEMS": MAX_RELATIONSHIP_ITEMS,
                "MEMORYGATE_MAX_LIST_ITEMS": MAX_LIST_ITEMS,
                "MEMORYGATE_MAX_METADATA_BYTES": MAX_METADATA_BYTES,
            },
        }
    
    return result


@service_tool
def memory_bootstrap(
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    context: Optional[RequestContext] = None,
) -> dict:
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

