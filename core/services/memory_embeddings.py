"""
Embedding maintenance tasks.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import and_

from core.db import DB
from core.models import Observation, Pattern, Concept, Document, Embedding
from core.services.memory_shared import (
    EMBEDDING_PROVIDER,
    EMBEDDING_BACKFILL_BATCH_LIMIT,
    EMBEDDING_BACKFILL_INTERVAL_SECONDS,
    embedding_circuit_breaker,
    _store_embedding,
    _vector_search_enabled,
    logger,
)

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
                if _store_embedding(
                    db,
                    source_type,
                    record.id,
                    text_value,
                    tenant_id=getattr(record, "tenant_id", None),
                ):
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
                logger.debug("embedding_backfill_complete", extra=stats)
        except Exception as exc:
            logger.warning(f"Embedding backfill error: {exc}")
