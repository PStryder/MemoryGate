"""
Shared validation helpers for MemoryGate services.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from core.config import (
    MAX_EMBEDDING_TEXT_LENGTH,
    MAX_METADATA_BYTES,
)
from core.errors import ValidationIssue
from core.models import Observation


def validate_required_text(value: str, field: str, max_len: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationIssue(f"{field} must be a non-empty string", field=field, error_type="required")
    if len(value) > max_len:
        raise ValidationIssue(f"{field} exceeds max length {max_len}", field=field, error_type="max_length")


def validate_optional_text(value: Optional[str], field: str, max_len: int) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValidationIssue(f"{field} must be a string", field=field, error_type="invalid_type")
    if len(value) > max_len:
        raise ValidationIssue(f"{field} exceeds max length {max_len}", field=field, error_type="max_length")


def validate_limit(value: int, field: str, max_value: int) -> None:
    if value <= 0 or value > max_value:
        raise ValidationIssue(f"{field} must be between 1 and {max_value}", field=field, error_type="out_of_range")


def validate_confidence(value: float, field: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValidationIssue(f"{field} must be between 0.0 and 1.0", field=field, error_type="out_of_range")


def validate_list(values: Optional[Sequence], field: str, max_items: int) -> None:
    if values is None:
        return
    if len(values) > max_items:
        raise ValidationIssue(f"{field} exceeds max items {max_items}", field=field, error_type="max_items")


def validate_string_list(
    values: Optional[Sequence[str]],
    field: str,
    max_items: int,
    max_item_length: int,
) -> None:
    if values is None:
        return
    if len(values) > max_items:
        raise ValidationIssue(f"{field} exceeds max items {max_items}", field=field, error_type="max_items")
    for item in values:
        if not isinstance(item, str):
            raise ValidationIssue(f"{field} must contain only strings", field=field, error_type="invalid_type")
        if len(item) > max_item_length:
            raise ValidationIssue(
                f"{field} item exceeds max length {max_item_length}",
                field=field,
                error_type="max_length",
            )


def validate_metadata(metadata: Optional[dict], field: str) -> None:
    if metadata is None:
        return
    try:
        size = len(json.dumps(metadata))
    except (TypeError, ValueError) as exc:
        raise ValidationIssue(f"{field} must be JSON-serializable", field=field, error_type="invalid_type") from exc
    if size > MAX_METADATA_BYTES:
        raise ValidationIssue(
            f"{field} exceeds max size {MAX_METADATA_BYTES} bytes",
            field=field,
            error_type="max_bytes",
        )


def validate_embedding_text(text: str) -> None:
    validate_required_text(text, "text", MAX_EMBEDDING_TEXT_LENGTH)


def validate_evidence_observation_ids(db, evidence_ids: Optional[Sequence[int]]) -> None:
    if not evidence_ids:
        return
    invalid_types = [
        value for value in evidence_ids
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid_types:
        raise ValidationIssue(
            "evidence_observation_ids must contain integer IDs",
            field="evidence_observation_ids",
            error_type="invalid_type",
        )
    invalid_values = [value for value in evidence_ids if value <= 0]
    if invalid_values:
        raise ValidationIssue(
            f"evidence_observation_ids must be positive integers: {invalid_values}",
            field="evidence_observation_ids",
            error_type="invalid_id",
        )
    existing_rows = (
        db.query(Observation.id)
        .filter(Observation.id.in_(set(evidence_ids)))
        .all()
    )
    existing_ids = {row[0] for row in existing_rows}
    missing_ids = sorted({value for value in evidence_ids if value not in existing_ids})
    if missing_ids:
        raise ValidationIssue(
            f"evidence_observation_ids not found: {missing_ids}",
            field="evidence_observation_ids",
            error_type="not_found",
        )
