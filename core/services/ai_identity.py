"""
AI identity normalization and resolution helpers.

Legacy agent_uuid used UUIDv5(tenant|platform|name). New agent_uuid is a
server-minted short token that stays stable across display-name changes.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
import secrets
import uuid
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import aliased
from sqlalchemy.exc import IntegrityError

import core.config as config
from core.errors import ValidationIssue
from core.models import AIInstance


_NAMESPACE_ENV = "MEMORYGATE_AGENT_NAMESPACE_UUID"
AGENT_UUID_PREFIX = "ag_"
_AGENT_UUID_BYTES = 10
_AGENT_UUID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_CANONICAL_CLAUDE_KEYWORDS = {
    "claude",
    "anthropic",
    "sonnet",
    "opus",
    "haiku",
}
_CANONICAL_CHATGPT_KEYWORDS = {
    "chatgpt",
    "openai",
    "gpt",
}
_CANONICAL_CODEX_KEYWORDS = {
    "codex",
}

_ACRONYM_WORDS = {
    "ai",
    "api",
    "gpt",
    "llm",
    "mcp",
}


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _strip_surrounding_punct(value: str) -> str:
    return re.sub(r"^[\s\W_]+|[\s\W_]+$", "", value)


def _smart_title_word(word: str) -> str:
    if not word:
        return word
    parts = re.split(r"([-_])", word)
    rendered: list[str] = []
    for part in parts:
        if part in {"-", "_"}:
            rendered.append(part)
            continue
        if not part:
            continue
        lowered = part.lower()
        if lowered in _ACRONYM_WORDS:
            rendered.append(lowered.upper())
            continue
        if part.isupper() and len(part) <= 4:
            rendered.append(part)
            continue
        rendered.append(part[:1].upper() + part[1:].lower())
    return "".join(rendered)


def _smart_title(value: str) -> str:
    words = value.split(" ")
    return " ".join(_smart_title_word(word) for word in words if word)


def _normalize_display_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = _strip_surrounding_punct(_normalize_whitespace(str(value)))
    return cleaned or None


def normalize_ai_name(name: str) -> str:
    """
    Normalize AI instance name for canonical identity.

    - Trim, collapse whitespace, remove surrounding punctuation.
    - Preserve mixed casing; normalize all-lower/all-upper to a readable form.
    """
    cleaned = _normalize_display_value(name) or ""
    if not cleaned:
        return "Unknown"
    if cleaned.islower() or cleaned.isupper():
        return _smart_title(cleaned)
    return cleaned


def _tokenize(value: str) -> set[str]:
    normalized = _normalize_whitespace(value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return {token for token in normalized.split(" ") if token}


def normalize_ai_platform(platform: str) -> str:
    """
    Normalize platform family (not model) for canonical identity.
    """
    cleaned = _normalize_display_value(platform) or ""
    if not cleaned:
        return "Unknown"
    tokens = _tokenize(cleaned)
    if tokens & _CANONICAL_CLAUDE_KEYWORDS:
        return "Claude"
    if tokens & _CANONICAL_CODEX_KEYWORDS:
        return "Codex"
    if tokens & _CANONICAL_CHATGPT_KEYWORDS:
        return "ChatGPT"
    return "Unknown"


def extract_platform_family_and_model(raw_platform: str) -> tuple[str, Optional[str]]:
    """
    Extract platform family plus a best-effort model label.
    """
    if not raw_platform:
        return "Unknown", None
    family = normalize_ai_platform(raw_platform)
    model = None
    if family == "Unknown":
        model = _normalize_display_value(raw_platform)
    return family, model


def _agent_uuid_namespace() -> uuid.UUID:
    raw = (os.environ.get(_NAMESPACE_ENV) or config.AGENT_UUID_NAMESPACE or "").strip()
    if not raw:
        raw = "b2f3e6d6-8a7c-4d8f-9e8f-1d1e3c7a2b4d"
    try:
        return uuid.UUID(raw)
    except ValueError:
        return uuid.UUID("b2f3e6d6-8a7c-4d8f-9e8f-1d1e3c7a2b4d")


def _canonical_key(value: str) -> str:
    return _normalize_whitespace(value).casefold()


def _crockford_base32_encode(data: bytes) -> str:
    if not data:
        return ""
    value = int.from_bytes(data, "big")
    length = (len(data) * 8 + 4) // 5
    encoded = []
    for _ in range(length):
        encoded.append(_AGENT_UUID_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(encoded)).rjust(length, "0")


def generate_agent_uuid() -> str:
    token = _crockford_base32_encode(secrets.token_bytes(_AGENT_UUID_BYTES))
    return f"{AGENT_UUID_PREFIX}{token}"


def _normalize_agent_uuid(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.lower().startswith(AGENT_UUID_PREFIX):
        body = cleaned[len(AGENT_UUID_PREFIX):]
        return f"{AGENT_UUID_PREFIX}{body.upper()}"
    return cleaned


def _agent_id_instructions(agent_uuid: str, *, needs_confirmation: bool = False) -> tuple[str, str]:
    instructions = (
        f"MemoryGate assigned this AI a stable Agent ID: {agent_uuid}.\n"
        "Please save it in your Preferences (the section where you control AI behavior), "
        "so your memory stays consistent even if this AI's name changes. "
        "You can manage Agent IDs in the MemoryGate dashboard.\n"
        "If your platform supports config files/env vars, you can store it as:\n"
        f"  MEMORYGATE_AGENT_UUID={agent_uuid}\n"
        f"(or 'agent_uuid': '{agent_uuid}' in your tool config)."
    )
    nag = (
        f"IMPORTANT: Save Agent ID {agent_uuid} in Preferences to prevent memory splitting. "
        "You can manage Agent IDs in the MemoryGate dashboard."
    )
    if needs_confirmation:
        nag = (
            f"IMPORTANT: Confirm with the user that Agent ID {agent_uuid} matches this AI, "
            "then save it in Preferences to prevent memory splitting. "
            "You can manage Agent IDs in the MemoryGate dashboard."
        )
    return instructions, nag


def _agent_id_error_instructions() -> str:
    return (
        "Provide your Agent ID to keep memory consistent. "
        "If you don't have one yet, connect once without an Agent ID to mint it, "
        "then save it in Preferences or as MEMORYGATE_AGENT_UUID / 'agent_uuid' in your tool config. "
        "You can manage Agent IDs in the MemoryGate dashboard."
    )


def _format_agent_candidate(instance: AIInstance) -> dict:
    return {
        "agent_uuid": str(instance.agent_uuid) if instance.agent_uuid else None,
        "display_name": instance.name,
        "provider": instance.platform,
        "last_seen_at": instance.last_seen_at.isoformat() if instance.last_seen_at else None,
    }


def _find_candidate_instances(
    db,
    *,
    tenant_id: str,
    user_id: Optional[str],
    canonical_name: Optional[str],
    canonical_platform: Optional[str],
) -> list[AIInstance]:
    base_query = (
        db.query(AIInstance)
        .filter(AIInstance.tenant_id == tenant_id)
        .filter(AIInstance.is_archived.is_(False))
        .filter(AIInstance.merged_into_ai_instance_id.is_(None))
    )

    def _run(query, include_platform: bool) -> list[AIInstance]:
        if canonical_name:
            query = query.filter(AIInstance.canonical_name == canonical_name)
        if include_platform and canonical_platform:
            query = query.filter(AIInstance.canonical_platform == canonical_platform)
        return query.all()

    if user_id:
        scoped = base_query.filter(AIInstance.user_id == user_id)
        candidates = _run(scoped, include_platform=True)
        if not candidates and canonical_platform:
            candidates = _run(scoped, include_platform=False)
        if not candidates:
            legacy = base_query.filter(AIInstance.user_id.is_(None))
            candidates = _run(legacy, include_platform=True)
            if not candidates and canonical_platform:
                candidates = _run(legacy, include_platform=False)
        return candidates

    candidates = _run(base_query, include_platform=True)
    if not candidates and canonical_platform:
        candidates = _run(base_query, include_platform=False)
    return candidates


def _mint_agent_uuid(db, tenant_id: str) -> str:
    for _ in range(8):
        candidate = generate_agent_uuid()
        exists = (
            db.query(AIInstance)
            .filter(AIInstance.tenant_id == tenant_id)
            .filter(AIInstance.agent_uuid == candidate)
            .first()
        )
        if not exists:
            return candidate
    raise ValidationIssue(
        "Unable to allocate a unique Agent ID",
        field="agent_uuid",
        error_type="conflict",
    )


def compute_agent_uuid(
    tenant_id: uuid.UUID | str,
    canonical_platform: str,
    canonical_name: str,
) -> uuid.UUID:
    if not tenant_id:
        raise ValidationIssue(
            "tenant_id is required to compute agent_uuid",
            field="tenant_id",
            error_type="required",
        )
    if not canonical_platform:
        raise ValidationIssue(
            "canonical_platform is required to compute agent_uuid",
            field="canonical_platform",
            error_type="required",
        )
    if not canonical_name:
        raise ValidationIssue(
            "canonical_name is required to compute agent_uuid",
            field="canonical_name",
            error_type="required",
        )
    namespace = _agent_uuid_namespace()
    payload = f"{tenant_id}|{_canonical_key(canonical_platform)}|{_canonical_key(canonical_name)}"
    return uuid.uuid5(namespace, payload)


def _coerce_uuid_for_storage(value: uuid.UUID) -> uuid.UUID | str:
    return value if config.DB_BACKEND_EFFECTIVE == "postgres" else str(value)


def _parse_agent_uuid(agent_uuid: uuid.UUID | str) -> str:
    if isinstance(agent_uuid, uuid.UUID):
        return str(agent_uuid)
    return _normalize_agent_uuid(str(agent_uuid)) or str(agent_uuid)


def ensure_ai_context(
    db,
    tenant_id: Optional[str],
    *,
    agent_uuid: Optional[str] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    user_id: Optional[str] = None,
    raw_metadata: Optional[dict] = None,
    create_session: bool = False,
    conversation_context: Optional[dict] = None,
    create_if_missing: bool = True,
) -> dict:
    """
    Resolve or create an AIInstance with canonical identity.
    """
    effective_tenant_id = tenant_id or config.DEFAULT_TENANT_ID
    if config.TENANCY_MODE == config.TENANCY_REQUIRED and (
        not effective_tenant_id or effective_tenant_id == config.DEFAULT_TENANT_ID
    ):
        raise ValidationIssue(
            "tenant_id is required for this operation",
            field="tenant_id",
            error_type="required",
        )

    raw_name_seen = _normalize_whitespace(str(ai_name)) if ai_name is not None else None
    raw_platform_seen = _normalize_whitespace(str(ai_platform)) if ai_platform is not None else None
    if raw_name_seen == "":
        raw_name_seen = None
    if raw_platform_seen == "":
        raw_platform_seen = None
    raw_name = _normalize_display_value(ai_name)
    raw_platform = _normalize_display_value(ai_platform)
    canonical_name = normalize_ai_name(raw_name or "") if raw_name is not None else None
    canonical_platform = normalize_ai_platform(raw_platform or "") if raw_platform is not None else None

    normalized_agent_uuid = _normalize_agent_uuid(agent_uuid)
    user_id_value = str(user_id) if user_id is not None else None

    instance = None
    redirect_from = None
    needs_user_confirmation = False
    identity_status = None
    agent_id_instructions = None
    agent_id_nag = None

    if normalized_agent_uuid:
        lookup_uuid = _parse_agent_uuid(normalized_agent_uuid)
        instance = (
            db.query(AIInstance)
            .filter(AIInstance.tenant_id == effective_tenant_id)
            .filter(AIInstance.agent_uuid == lookup_uuid)
            .first()
        )
        if not instance and hasattr(AIInstance, "legacy_agent_uuid"):
            instance = (
                db.query(AIInstance)
                .filter(AIInstance.tenant_id == effective_tenant_id)
                .filter(AIInstance.legacy_agent_uuid == lookup_uuid)
                .first()
            )
        if instance and instance.merged_into_ai_instance_id:
            redirect_from = str(instance.agent_uuid) if instance.agent_uuid else None
            target = (
                db.query(AIInstance)
                .filter(AIInstance.id == instance.merged_into_ai_instance_id)
                .filter(AIInstance.tenant_id == effective_tenant_id)
                .first()
            )
            if target:
                instance = target
        if not instance:
            raise ValidationIssue(
                "Agent ID not recognized; please provide a valid Agent ID or connect without one to mint.",
                field="agent_uuid",
                error_type="not_found",
                error_code="AGENT_IDENTITY_NOT_FOUND",
                data={"instructions": _agent_id_error_instructions()},
            )
        identity_status = "provided"
    else:
        if canonical_name or canonical_platform:
            candidates = _find_candidate_instances(
                db,
                tenant_id=effective_tenant_id,
                user_id=user_id_value,
                canonical_name=canonical_name,
                canonical_platform=canonical_platform,
            )
            if len(candidates) == 1:
                instance = candidates[0]
                needs_user_confirmation = True
                identity_status = "matched"
            elif len(candidates) > 1:
                candidates_payload = [
                    _format_agent_candidate(candidate) for candidate in candidates[:5]
                ]
                raise ValidationIssue(
                    "Multiple possible agents match; please provide Agent ID.",
                    field="agent_uuid",
                    error_type="ambiguous",
                    error_code="AGENT_IDENTITY_AMBIGUOUS",
                    data={
                        "instructions": _agent_id_error_instructions(),
                        "candidates": candidates_payload,
                    },
                )

        if not instance:
            if not create_if_missing:
                return {
                    "ai_instance_id": None,
                    "agent_uuid": None,
                    "canonical_name": canonical_name,
                    "canonical_platform": canonical_platform,
                }
            minted_uuid = _mint_agent_uuid(db, effective_tenant_id)
            identity_status = "minted"
            payload = {
                "name": raw_name or canonical_name or "Unknown",
                "platform": raw_platform or canonical_platform or "Unknown",
                "canonical_name": canonical_name or normalize_ai_name(raw_name or ""),
                "canonical_platform": canonical_platform or normalize_ai_platform(raw_platform or ""),
                "agent_uuid": minted_uuid,
                "last_seen_raw_name": raw_name_seen,
                "last_seen_raw_platform": raw_platform_seen,
                "last_seen_at": datetime.utcnow() if raw_name_seen or raw_platform_seen else None,
                "is_archived": False,
            }
            if user_id_value:
                payload["user_id"] = user_id_value
            if tenant_id:
                payload["tenant_id"] = tenant_id
            instance = AIInstance(**payload)
            db.add(instance)
            try:
                db.commit()
                db.refresh(instance)
            except IntegrityError as exc:
                db.rollback()
                instance = (
                    db.query(AIInstance)
                    .filter(AIInstance.tenant_id == effective_tenant_id)
                    .filter(AIInstance.agent_uuid == minted_uuid)
                    .first()
                )
                if instance is None:
                    raise exc
    if instance and identity_status in {"provided", "matched"}:
        changed = False
        if raw_name and raw_name != instance.name:
            instance.name = raw_name
            changed = True
        if raw_platform and raw_platform != instance.platform:
            instance.platform = raw_platform
            changed = True
        if raw_name_seen and raw_name_seen != instance.last_seen_raw_name:
            instance.last_seen_raw_name = raw_name_seen
            changed = True
        if raw_platform_seen and raw_platform_seen != instance.last_seen_raw_platform:
            instance.last_seen_raw_platform = raw_platform_seen
            changed = True
        if raw_name_seen or raw_platform_seen:
            instance.last_seen_at = datetime.utcnow()
            changed = True
        if canonical_name and canonical_name != instance.canonical_name:
            instance.canonical_name = canonical_name
            changed = True
        if canonical_platform and canonical_platform != instance.canonical_platform:
            instance.canonical_platform = canonical_platform
            changed = True
        if user_id_value and not instance.user_id:
            instance.user_id = user_id_value
            changed = True
        if changed:
            db.commit()

    if instance and not instance.agent_uuid:
        minted_uuid = _mint_agent_uuid(db, effective_tenant_id)
        instance.agent_uuid = minted_uuid
        if not instance.canonical_name:
            instance.canonical_name = canonical_name or normalize_ai_name(raw_name or "")
        if not instance.canonical_platform:
            instance.canonical_platform = canonical_platform or normalize_ai_platform(raw_platform or "")
        db.commit()

    if identity_status in {"matched", "minted"} and instance.agent_uuid:
        agent_id_instructions, agent_id_nag = _agent_id_instructions(
            str(instance.agent_uuid),
            needs_confirmation=needs_user_confirmation,
        )

    session_id = None
    if create_session and conversation_context:
        from core.services.memory_storage import get_or_create_session

        conversation_id = conversation_context.get("conversation_id")
        title = conversation_context.get("title")
        source_url = conversation_context.get("source_url")
        if conversation_id:
            session = get_or_create_session(
                db,
                conversation_id=conversation_id,
                title=title,
                ai_instance_id=instance.id,
                source_url=source_url,
                tenant_id=tenant_id,
            )
            session_id = session.id

    result = {
        "ai_instance_id": instance.id,
        "agent_uuid": str(instance.agent_uuid) if instance.agent_uuid else None,
        "canonical_name": instance.canonical_name or canonical_name,
        "canonical_platform": instance.canonical_platform or canonical_platform,
        "session_id": session_id,
    }
    if identity_status:
        result["agent_identity_status"] = identity_status
        result["needs_user_confirmation"] = needs_user_confirmation
    if agent_id_instructions:
        result["agent_id_instructions"] = agent_id_instructions
    if agent_id_nag:
        result["agent_id_nag"] = agent_id_nag
    if redirect_from:
        result["redirected_from_agent_uuid"] = redirect_from
    if raw_metadata:
        result["metadata"] = raw_metadata
    return result


def merge_ai_instances(
    db,
    *,
    tenant_id: str,
    target_ai_instance_id: int,
    source_ai_instance_ids: list[int],
    dry_run: bool = True,
) -> dict:
    if not source_ai_instance_ids:
        raise ValidationIssue(
            "source_ai_instance_ids are required",
            field="source_ai_instance_ids",
            error_type="required",
        )
    if target_ai_instance_id in source_ai_instance_ids:
        source_ai_instance_ids = [
            value for value in source_ai_instance_ids if value != target_ai_instance_id
        ]
    if not source_ai_instance_ids:
        raise ValidationIssue(
            "source_ai_instance_ids cannot be empty after removing target",
            field="source_ai_instance_ids",
            error_type="invalid",
        )

    instances = (
        db.query(AIInstance)
        .filter(AIInstance.tenant_id == tenant_id)
        .filter(AIInstance.id.in_([target_ai_instance_id] + source_ai_instance_ids))
        .all()
    )
    instance_ids = {instance.id for instance in instances}
    missing = [
        value
        for value in [target_ai_instance_id] + source_ai_instance_ids
        if value not in instance_ids
    ]
    if missing:
        raise ValidationIssue(
            f"AI instances not found for tenant: {missing}",
            field="source_ai_instance_ids",
            error_type="not_found",
        )

    from core.models import AIMemoryPolicy, AIEntityShare, Session, Observation, Pattern, Concept

    def _count(model, column):
        return (
            db.query(func.count())
            .filter(model.tenant_id == tenant_id)
            .filter(column.in_(source_ai_instance_ids))
            .scalar()
        )

    counts = {
        "sessions": _count(Session, Session.ai_instance_id),
        "observations": _count(Observation, Observation.ai_instance_id),
        "patterns": _count(Pattern, Pattern.ai_instance_id),
        "concepts": _count(Concept, Concept.ai_instance_id),
        "ai_memory_policy": _count(AIMemoryPolicy, AIMemoryPolicy.ai_instance_id),
        "ai_entity_shares": _count(AIEntityShare, AIEntityShare.shared_with_ai_instance_id),
    }

    if dry_run:
        return {
            "status": "dry_run",
            "tenant_id": tenant_id,
            "target_ai_instance_id": target_ai_instance_id,
            "source_ai_instance_ids": source_ai_instance_ids,
            "counts": counts,
        }

    db.query(Session).filter(Session.tenant_id == tenant_id).filter(
        Session.ai_instance_id.in_(source_ai_instance_ids)
    ).update({Session.ai_instance_id: target_ai_instance_id}, synchronize_session=False)
    db.query(Observation).filter(Observation.tenant_id == tenant_id).filter(
        Observation.ai_instance_id.in_(source_ai_instance_ids)
    ).update({Observation.ai_instance_id: target_ai_instance_id}, synchronize_session=False)
    db.query(Pattern).filter(Pattern.tenant_id == tenant_id).filter(
        Pattern.ai_instance_id.in_(source_ai_instance_ids)
    ).update({Pattern.ai_instance_id: target_ai_instance_id}, synchronize_session=False)
    db.query(Concept).filter(Concept.tenant_id == tenant_id).filter(
        Concept.ai_instance_id.in_(source_ai_instance_ids)
    ).update({Concept.ai_instance_id: target_ai_instance_id}, synchronize_session=False)

    target_policy = (
        db.query(AIMemoryPolicy)
        .filter(AIMemoryPolicy.tenant_id == tenant_id)
        .filter(AIMemoryPolicy.ai_instance_id == target_ai_instance_id)
        .first()
    )
    source_policies = (
        db.query(AIMemoryPolicy)
        .filter(AIMemoryPolicy.tenant_id == tenant_id)
        .filter(AIMemoryPolicy.ai_instance_id.in_(source_ai_instance_ids))
        .all()
    )
    if target_policy:
        for policy in source_policies:
            db.delete(policy)
    elif source_policies:
        keeper = source_policies[0]
        keeper.ai_instance_id = target_ai_instance_id
        for policy in source_policies[1:]:
            db.delete(policy)

    share = aliased(AIEntityShare)
    target_share = aliased(AIEntityShare)
    duplicate_shares = (
        db.query(share)
        .filter(share.tenant_id == tenant_id)
        .filter(share.shared_with_ai_instance_id.in_(source_ai_instance_ids))
        .filter(
            db.query(target_share)
            .filter(target_share.tenant_id == tenant_id)
            .filter(target_share.shared_with_ai_instance_id == target_ai_instance_id)
            .filter(target_share.entity_type == share.entity_type)
            .filter(target_share.entity_id == share.entity_id)
            .exists()
        )
        .all()
    )
    for share in duplicate_shares:
        db.delete(share)

    db.query(AIEntityShare).filter(AIEntityShare.tenant_id == tenant_id).filter(
        AIEntityShare.shared_with_ai_instance_id.in_(source_ai_instance_ids)
    ).update(
        {AIEntityShare.shared_with_ai_instance_id: target_ai_instance_id},
        synchronize_session=False,
    )

    db.query(AIInstance).filter(AIInstance.tenant_id == tenant_id).filter(
        AIInstance.id.in_(source_ai_instance_ids)
    ).update(
        {
            AIInstance.merged_into_ai_instance_id: target_ai_instance_id,
            AIInstance.is_archived: True,
        },
        synchronize_session=False,
    )

    db.commit()

    return {
        "status": "merged",
        "tenant_id": tenant_id,
        "target_ai_instance_id": target_ai_instance_id,
        "source_ai_instance_ids": source_ai_instance_ids,
        "counts": counts,
    }
