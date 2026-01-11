"""
Retention and pruning logic for MemoryGate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import case, func, or_

import core.config as config
from core.db import DB
from core.models import (
    Concept,
    Document,
    MemorySummary,
    MemoryTombstone,
    MemoryTier,
    Observation,
    Pattern,
    TombstoneAction,
    MEMORY_MODELS,
)


def clamp_score(score: float, min_value: float, max_value: float) -> float:
    if score < min_value:
        return min_value
    if score > max_value:
        return max_value
    return score


def apply_floor(score: float, floor_score: float) -> float:
    return max(score, floor_score)


def apply_fetch_bump(
    score: float,
    alpha: float,
    bump_clamp_min: float = -2.0,
    bump_clamp_max: float = 1.0,
) -> float:
    clamped = clamp_score(score, bump_clamp_min, bump_clamp_max)
    return score + alpha * (1 - clamped)


def apply_decay_tick(score: float, beta: float, pressure_multiplier: float) -> float:
    return score - beta * pressure_multiplier


def serialize_memory_id(memory_type: str, memory_id: int) -> str:
    return f"{memory_type}:{memory_id}"


def write_tombstone(
    db,
    memory_id: str,
    action: TombstoneAction,
    from_tier: Optional[MemoryTier],
    to_tier: Optional[MemoryTier],
    reason: Optional[str],
    actor: Optional[str],
    metadata: Optional[dict] = None,
) -> None:
    if not config.TOMBSTONES_ENABLED:
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


def summary_text_for_record(memory_type: str, record) -> str:
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
    return source[: config.SUMMARY_MAX_LENGTH]


def find_summary_for_source(db, memory_type: str, source_id: int) -> Optional[MemorySummary]:
    return db.query(MemorySummary).filter(
        MemorySummary.source_type == memory_type,
        MemorySummary.source_id == source_id,
    ).first()


def calculate_pressure_multiplier(db) -> float:
    total = (
        db.query(func.count(Observation.id)).scalar()
        + db.query(func.count(Pattern.id)).scalar()
        + db.query(func.count(Concept.id)).scalar()
        + db.query(func.count(Document.id)).scalar()
        + db.query(func.count(MemorySummary.id)).scalar()
    )
    budget = max(1, config.RETENTION_BUDGET)
    multiplier = config.RETENTION_PRESSURE * (total / budget)
    return max(1.0, multiplier)


def apply_decay_to_model(
    db,
    model,
    tier: MemoryTier,
    pressure_multiplier: float,
    decay_multiplier: float,
) -> int:
    beta = config.SCORE_DECAY_BETA * decay_multiplier
    if beta <= 0:
        return 0
    score_expr = model.score - beta * pressure_multiplier
    clamped = case(
        (score_expr < config.SCORE_CLAMP_MIN, config.SCORE_CLAMP_MIN),
        (score_expr > config.SCORE_CLAMP_MAX, config.SCORE_CLAMP_MAX),
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


def summarize_and_archive(db) -> dict:
    archived = 0
    summaries_created = 0
    reason = "auto_summarize"
    for mem_type, model in MEMORY_MODELS.items():
        records = (
            db.query(model)
            .filter(model.tier == MemoryTier.hot)
            .filter(model.score <= config.SUMMARY_TRIGGER_SCORE)
            .order_by(model.score.asc())
            .limit(config.SUMMARY_BATCH_LIMIT)
            .all()
        )
        for record in records:
            summary_text = summary_text_for_record(mem_type, record)
            if not summary_text:
                continue
            summary = find_summary_for_source(db, mem_type, record.id)
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
            write_tombstone(
                db,
                serialize_memory_id(mem_type, record.id),
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
            write_tombstone(
                db,
                serialize_memory_id(mem_type, record.id),
                TombstoneAction.archived,
                from_tier=MemoryTier.hot,
                to_tier=MemoryTier.cold,
                reason=reason,
                actor="system",
            )
            archived += 1

    db.commit()
    return {"archived": archived, "summaries_created": summaries_created}


def purge_cold_records(db) -> dict:
    purged = 0
    marked = 0
    skipped = 0
    reason = "retention_purge"

    from core.services.memory_archive import _archive_cold_record_to_store, _enforce_archive_quota

    for mem_type, model in MEMORY_MODELS.items():
        records = (
            db.query(model)
            .filter(model.tier == MemoryTier.cold)
            .filter(
                or_(
                    model.score <= config.PURGE_TRIGGER_SCORE,
                    model.purge_eligible.is_(True),
                )
            )
            .order_by(model.score.asc())
            .limit(config.RETENTION_PURGE_LIMIT)
            .all()
        )
        for record in records:
            if config.FORGET_MODE == "soft" and not record.purge_eligible:
                record.purge_eligible = True
                write_tombstone(
                    db,
                    serialize_memory_id(mem_type, record.id),
                    TombstoneAction.purged,
                    from_tier=MemoryTier.cold,
                    to_tier=MemoryTier.cold,
                    reason="soft_purge_marked",
                    actor="system",
                    metadata={"mode": "soft"},
                )
                marked += 1
                continue

            summary = find_summary_for_source(db, mem_type, record.id)
            if summary is None and not config.ALLOW_HARD_PURGE_WITHOUT_SUMMARY:
                skipped += 1
                continue

            _archive_cold_record_to_store(db, mem_type, record, reason, "system")
            write_tombstone(
                db,
                serialize_memory_id(mem_type, record.id),
                TombstoneAction.purged,
                from_tier=MemoryTier.cold,
                to_tier=None,
                reason=reason,
                actor="system",
                metadata={"mode": "archive"},
            )
            purged += 1

    # Purge summaries into archive store
    summaries = (
        db.query(MemorySummary)
        .filter(MemorySummary.tier == MemoryTier.cold)
        .filter(
            or_(
                MemorySummary.score <= config.PURGE_TRIGGER_SCORE,
                MemorySummary.purge_eligible.is_(True),
            )
        )
        .order_by(MemorySummary.score.asc())
        .limit(config.RETENTION_PURGE_LIMIT)
        .all()
    )
    for summary in summaries:
        if config.FORGET_MODE == "soft" and not summary.purge_eligible:
            summary.purge_eligible = True
            write_tombstone(
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

        _archive_cold_record_to_store(db, "summary", summary, reason, "system")
        write_tombstone(
            db,
            f"summary:{summary.id}",
            TombstoneAction.purged,
            from_tier=MemoryTier.cold,
            to_tier=None,
            reason=reason,
            actor="system",
            metadata={"mode": "archive"},
        )
        purged += 1

    db.commit()
    quota_stats = _enforce_archive_quota(db) if purged else {"evicted": 0}
    return {
        "purged": purged,
        "marked": marked,
        "skipped": skipped,
        "archive_evicted": quota_stats.get("evicted", 0),
    }


def run_retention_tick() -> None:
    if DB.SessionLocal is None:
        return
    db = DB.SessionLocal()
    try:
        pressure = calculate_pressure_multiplier(db)
        hot_updates = 0
        cold_updates = 0
        for model in MEMORY_MODELS.values():
            hot_updates += apply_decay_to_model(db, model, MemoryTier.hot, pressure, 1.0)
            cold_updates += apply_decay_to_model(
                db,
                model,
                MemoryTier.cold,
                pressure,
                config.COLD_DECAY_MULTIPLIER,
            )
        hot_updates += apply_decay_to_model(db, MemorySummary, MemoryTier.hot, pressure, 1.0)
        cold_updates += apply_decay_to_model(
            db,
            MemorySummary,
            MemoryTier.cold,
            pressure,
            config.COLD_DECAY_MULTIPLIER,
        )
        db.commit()

        summary_stats = summarize_and_archive(db)
        purge_stats = purge_cold_records(db)
        config.logger.info(
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
