"""
Retention and pruning logic for MemoryGate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import case, func, or_

import core.config as config
from core.context import require_tenant_id_value
from core.audit import log_event
from core.audit_constants import (
    EVENT_MEMORY_ARCHIVED,
    EVENT_MEMORY_PURGED_TO_ARCHIVE,
)
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
    tenant_id: Optional[str] = None,
) -> None:
    if not config.TOMBSTONES_ENABLED:
        return
    effective_tenant_id = require_tenant_id_value(tenant_id)
    tombstone = MemoryTombstone(
        memory_id=memory_id,
        action=action,
        from_tier=from_tier,
        to_tier=to_tier,
        reason=reason,
        actor=actor,
        metadata_=metadata or {},
        tenant_id=effective_tenant_id,
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


def find_summary_for_source(
    db,
    memory_type: str,
    source_id: int,
    tenant_id: Optional[str] = None,
) -> Optional[MemorySummary]:
    query = db.query(MemorySummary).filter(
        MemorySummary.source_type == memory_type,
        MemorySummary.source_id == source_id,
    )
    if tenant_id:
        query = query.filter(MemorySummary.tenant_id == tenant_id)
    return query.first()


def calculate_pressure_multiplier(db, tenant_id: Optional[str] = None) -> float:
    obs_query = db.query(func.count(Observation.id))
    pattern_query = db.query(func.count(Pattern.id))
    concept_query = db.query(func.count(Concept.id))
    doc_query = db.query(func.count(Document.id))
    summary_query = db.query(func.count(MemorySummary.id))
    if tenant_id:
        obs_query = obs_query.filter(Observation.tenant_id == tenant_id)
        pattern_query = pattern_query.filter(Pattern.tenant_id == tenant_id)
        concept_query = concept_query.filter(Concept.tenant_id == tenant_id)
        doc_query = doc_query.filter(Document.tenant_id == tenant_id)
        summary_query = summary_query.filter(MemorySummary.tenant_id == tenant_id)
    total = (
        obs_query.scalar()
        + pattern_query.scalar()
        + concept_query.scalar()
        + doc_query.scalar()
        + summary_query.scalar()
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
    tenant_id: Optional[str] = None,
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
    query = db.query(model).filter(model.tier == tier)
    if tenant_id:
        query = query.filter(model.tenant_id == tenant_id)
    result = query.update({model.score: floored}, synchronize_session=False)
    return result or 0


def summarize_and_archive(db, tenant_id: Optional[str] = None) -> dict:
    archived = 0
    archived_ids: list[str] = []
    summaries_created = 0
    reason = "auto_summarize"
    for mem_type, model in MEMORY_MODELS.items():
        records_query = (
            db.query(model)
            .filter(model.tier == MemoryTier.hot)
            .filter(model.score <= config.SUMMARY_TRIGGER_SCORE)
            .order_by(model.score.asc())
            .limit(config.SUMMARY_BATCH_LIMIT)
        )
        if tenant_id:
            records_query = records_query.filter(model.tenant_id == tenant_id)
        records = records_query.all()
        for record in records:
            summary_text = summary_text_for_record(mem_type, record)
            if not summary_text:
                continue
            summary = find_summary_for_source(db, mem_type, record.id, tenant_id=tenant_id)
            if summary:
                summary.summary_text = summary_text
            else:
                payload = {
                    "source_type": mem_type,
                    "source_id": record.id,
                    "source_ids": [record.id],
                    "summary_text": summary_text,
                    "metadata_": {"reason": reason},
                }
                if tenant_id:
                    payload["tenant_id"] = tenant_id
                summary = MemorySummary(**payload)
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
                tenant_id=tenant_id,
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
                tenant_id=tenant_id,
            )
            archived += 1
            archived_ids.append(serialize_memory_id(mem_type, record.id))

    if archived_ids:
        log_event(
            db,
            event_type=EVENT_MEMORY_ARCHIVED,
            actor_type="system",
            org_id=tenant_id,
            target_type="memory",
            target_ids=archived_ids,
            count_affected=archived,
            reason=reason,
            metadata={"source": "retention_tick", "mode": "summarize"},
        )

    db.commit()
    return {"archived": archived, "summaries_created": summaries_created}


def purge_cold_records(db, tenant_id: Optional[str] = None) -> dict:
    purged = 0
    marked = 0
    skipped = 0
    reason = "retention_purge"
    purged_ids: list[str] = []
    purged_summary_ids: list[int] = []

    from core.services.memory_archive import _archive_cold_record_to_store, _enforce_archive_quota

    for mem_type, model in MEMORY_MODELS.items():
        records_query = (
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
        )
        if tenant_id:
            records_query = records_query.filter(model.tenant_id == tenant_id)
        records = records_query.all()
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
                    tenant_id=tenant_id,
                )
                marked += 1
                continue

            summary = find_summary_for_source(db, mem_type, record.id, tenant_id=tenant_id)
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
                tenant_id=tenant_id,
            )
            purged += 1
            purged_ids.append(serialize_memory_id(mem_type, record.id))

    summaries_query = (
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
    )
    if tenant_id:
        summaries_query = summaries_query.filter(MemorySummary.tenant_id == tenant_id)
    summaries = summaries_query.all()
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
                tenant_id=tenant_id,
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
            tenant_id=tenant_id,
        )
        purged += 1
        purged_summary_ids.append(summary.id)

    if purged_ids:
        log_event(
            db,
            event_type=EVENT_MEMORY_PURGED_TO_ARCHIVE,
            actor_type="system",
            org_id=tenant_id,
            target_type="memory",
            target_ids=purged_ids,
            count_affected=len(purged_ids),
            reason=reason,
            metadata={"source": "retention_tick"},
        )
    if purged_summary_ids:
        log_event(
            db,
            event_type=EVENT_MEMORY_PURGED_TO_ARCHIVE,
            actor_type="system",
            org_id=tenant_id,
            target_type="summary",
            target_ids=purged_summary_ids,
            count_affected=len(purged_summary_ids),
            reason=reason,
            metadata={"source": "retention_tick"},
        )

    db.commit()
    quota_stats = _enforce_archive_quota(db, tenant_id=tenant_id) if purged else {"evicted": 0}
    return {
        "purged": purged,
        "marked": marked,
        "skipped": skipped,
        "archive_evicted": quota_stats.get("evicted", 0),
    }


def _tenant_ids_for_retention(db) -> list[str]:
    tenant_ids: set[str] = set()
    for model in list(MEMORY_MODELS.values()) + [MemorySummary]:
        rows = db.query(model.tenant_id).distinct().all()
        for value in rows:
            if value and value[0]:
                tenant_ids.add(value[0])
    return sorted(tenant_ids)


def _run_retention_for_tenant(db, tenant_id: Optional[str]) -> None:
    pressure = calculate_pressure_multiplier(db, tenant_id=tenant_id)
    hot_updates = 0
    cold_updates = 0
    for model in MEMORY_MODELS.values():
        hot_updates += apply_decay_to_model(db, model, MemoryTier.hot, pressure, 1.0, tenant_id=tenant_id)
        cold_updates += apply_decay_to_model(
            db,
            model,
            MemoryTier.cold,
            pressure,
            config.COLD_DECAY_MULTIPLIER,
            tenant_id=tenant_id,
        )
    hot_updates += apply_decay_to_model(db, MemorySummary, MemoryTier.hot, pressure, 1.0, tenant_id=tenant_id)
    cold_updates += apply_decay_to_model(
        db,
        MemorySummary,
        MemoryTier.cold,
        pressure,
        config.COLD_DECAY_MULTIPLIER,
        tenant_id=tenant_id,
    )
    db.commit()

    summary_stats = summarize_and_archive(db, tenant_id=tenant_id)
    purge_stats = purge_cold_records(db, tenant_id=tenant_id)
    config.logger.debug(
        "retention_tick_complete",
        extra={
            "tenant_id": tenant_id,
            "decayed_hot": hot_updates,
            "decayed_cold": cold_updates,
            **summary_stats,
            **purge_stats,
        },
    )


def run_retention_tick(tenant_id: Optional[str] = None) -> None:
    if DB.SessionLocal is None:
        return
    db = DB.SessionLocal()
    try:
        if tenant_id:
            _run_retention_for_tenant(db, tenant_id)
            return
        if config.TENANCY_MODE == config.TENANCY_REQUIRED:
            if tenant_id:
                _run_retention_for_tenant(db, tenant_id)
                return
            for tenant in _tenant_ids_for_retention(db):
                _run_retention_for_tenant(db, tenant)
            return
        _run_retention_for_tenant(db, tenant_id)
    finally:
        db.close()
