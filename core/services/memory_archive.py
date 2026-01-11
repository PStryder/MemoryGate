"""
Archive and rehydration services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from core.context import RequestContext
from core.db import DB
from core.models import MemorySummary, MemoryTier, TombstoneAction, MEMORY_MODELS
from core.services.memory_search import search_cold_memory
from core.services.memory_shared import (
    _apply_rehydrate_bump,
    _collect_records_by_refs,
    _collect_summary_threshold_records,
    _collect_threshold_records,
    _find_summary_for_source,
    _parse_memory_ref,
    _serialize_memory_id,
    _summary_text_for_record,
    _validate_limit,
    _write_tombstone,
    ARCHIVE_LIMIT_DEFAULT,
    ARCHIVE_LIMIT_MAX,
    REHYDRATE_LIMIT_MAX,
    SUMMARY_TRIGGER_SCORE,
    TOMBSTONES_ENABLED,
    service_tool,
)

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
