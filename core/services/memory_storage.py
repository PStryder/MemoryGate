"""
Memory storage and session services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy.exc import IntegrityError

from core.context import RequestContext, resolve_tenant_id
import core.config as config
from core.errors import ValidationIssue
from core.db import DB
from core.models import AIInstance, Session, Observation, Document
from core.services.ai_identity import ensure_ai_context
from core.services.memory_shared import (
    _store_embedding,
    _validate_required_text,
    _validate_optional_text,
    _validate_confidence,
    _validate_string_list,
    _validate_metadata,
    _validate_evidence_observation_ids,
    MAX_TEXT_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_LIST_ITEMS,
    MAX_LIST_ITEM_LENGTH,
    MAX_SHORT_TEXT_LENGTH,
    MAX_TITLE_LENGTH,
    MAX_DOC_TYPE_LENGTH,
    MAX_URL_LENGTH,
    service_tool,
    logger,
)
from core.services.memory_quota import (
    enforce_quota_or_raise,
    estimate_document_bytes,
    estimate_observation_bytes,
    estimate_embedding_bytes,
    record_storage_usage,
)


def _context_values(context: Optional[RequestContext]) -> tuple[Optional[str], Optional[str]]:
    tenant_id = resolve_tenant_id(context)
    user_pk = None
    if context and context.auth and context.auth.user_id is not None:
        user_pk = str(context.auth.user_id)
    return tenant_id, user_pk


def get_or_create_ai_instance(
    db,
    name: str,
    platform: str,
    tenant_id: Optional[str] = None,
) -> AIInstance:
    """Get or create an AI instance by name."""
    ai_context = ensure_ai_context(
        db,
        tenant_id,
        ai_name=name,
        ai_platform=platform,
    )
    instance = (
        db.query(AIInstance)
        .filter(AIInstance.id == ai_context["ai_instance_id"])
        .first()
    )
    if instance is None:
        raise RuntimeError("AIInstance create failed")
    return instance


def get_or_create_session(
    db,
    conversation_id: str,
    title: Optional[str] = None,
    ai_instance_id: Optional[int] = None,
    source_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Session:
    """Get or create a session by conversation_id."""
    if config.TENANCY_MODE == config.TENANCY_REQUIRED and (
        not tenant_id or tenant_id == config.DEFAULT_TENANT_ID
    ):
        raise ValidationIssue(
            "tenant_id is required for this operation",
            field="tenant_id",
            error_type="required",
        )
    query = db.query(Session).filter(Session.conversation_id == conversation_id)
    if tenant_id:
        query = query.filter(Session.tenant_id == tenant_id)
    session = query.first()
    if not session:
        payload = {
            "conversation_id": conversation_id,
            "title": title,
            "ai_instance_id": ai_instance_id,
            "source_url": source_url,
        }
        if tenant_id:
            payload["tenant_id"] = tenant_id
        session = Session(**payload)
        db.add(session)
        try:
            db.commit()
            db.refresh(session)
        except IntegrityError as exc:
            db.rollback()
            query = db.query(Session).filter(Session.conversation_id == conversation_id)
            if tenant_id:
                query = query.filter(Session.tenant_id == tenant_id)
            session = query.first()
            if session is None:
                raise exc
    elif title and session.title != title:
        session.title = title
        session.last_active = datetime.utcnow()
        db.commit()
    return session


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
    agent_uuid: Optional[str] = None,
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
        agent_uuid: Stable AI identity token (optional)
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
    effective_agent_uuid = agent_uuid or (context.agent_uuid if context else None)
    _validate_optional_text(effective_agent_uuid, "agent_uuid", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        tenant_id, user_pk = _context_values(context)
        evidence_ids: list[int] = []
        if evidence:
            for item in evidence:
                if isinstance(item, str) and item.strip().isdigit():
                    evidence_ids.append(int(item.strip()))
        if evidence_ids and len(evidence_ids) == len(evidence):
            _validate_evidence_observation_ids(db, evidence_ids, tenant_id=tenant_id)

        bytes_to_write = estimate_observation_bytes(
            observation,
            domain=domain,
            evidence=evidence,
        )
        bytes_to_write += estimate_embedding_bytes()
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)

        # Get or create AI instance
        effective_ai_name = ai_name
        effective_ai_platform = ai_platform
        if effective_agent_uuid and ai_name == "Unknown" and ai_platform == "Unknown":
            effective_ai_name = None
            effective_ai_platform = None
        ai_context = ensure_ai_context(
            db,
            tenant_id,
            agent_uuid=effective_agent_uuid,
            ai_name=effective_ai_name,
            ai_platform=effective_ai_platform,
            user_id=user_pk,
        )

        # Get or create session if conversation_id provided
        session = None
        if conversation_id:
            session = get_or_create_session(
                db,
                conversation_id,
                conversation_title,
                ai_context["ai_instance_id"],
                tenant_id=tenant_id,
            )

        # Create observation
        obs_payload = {
            "observation": observation,
            "confidence": confidence,
            "domain": domain,
            "evidence": evidence or [],
            "ai_instance_id": ai_context["ai_instance_id"],
            "session_id": session.id if session else None,
            "created_by_user_pk": user_pk,
        }
        if tenant_id:
            obs_payload["tenant_id"] = tenant_id
        obs = Observation(**obs_payload)
        db.add(obs)
        db.commit()
        db.refresh(obs)

        # Generate and store embedding (if enabled)
        _store_embedding(db, "observation", obs.id, observation, tenant_id=tenant_id)

        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)

        result = {
            "ack": "SUCCESS",
            "status": "stored",
            "message": f"Observation stored successfully (id={obs.id})",
            "id": obs.id,
            "ref": f"observation:{obs.id}",
            "observation": observation,
            "confidence": confidence,
            "domain": domain,
            "ai_name": ai_context.get("canonical_name") or ai_name,
            "session_title": conversation_title,
        }
        result.update(
            {
                key: ai_context[key]
                for key in (
                    "agent_uuid",
                    "canonical_name",
                    "canonical_platform",
                    "agent_id_instructions",
                    "agent_id_nag",
                    "needs_user_confirmation",
                    "agent_identity_status",
                )
                if ai_context.get(key) is not None
            }
        )
        return result
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
    agent_uuid: Optional[str] = None,
) -> dict:
    """
    Initialize or update a session for the current conversation.

    Args:
        conversation_id: Unique conversation identifier (UUID)
        title: Conversation title
        ai_name: Name of AI instance (e.g., "Kee")
        ai_platform: Platform (e.g., "Claude")
        source_url: Optional URL to the conversation
        agent_uuid: Stable AI identity token (optional)

    Returns:
        Session information
    """
    _validate_required_text(conversation_id, "conversation_id", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(title, "title", MAX_TITLE_LENGTH)
    _validate_required_text(ai_name, "ai_name", MAX_SHORT_TEXT_LENGTH)
    _validate_required_text(ai_platform, "ai_platform", MAX_SHORT_TEXT_LENGTH)
    _validate_optional_text(source_url, "source_url", MAX_URL_LENGTH)
    effective_agent_uuid = agent_uuid or (context.agent_uuid if context else None)
    _validate_optional_text(effective_agent_uuid, "agent_uuid", MAX_SHORT_TEXT_LENGTH)

    db = DB.SessionLocal()
    try:
        tenant_id, user_pk = _context_values(context)
        ai_context = ensure_ai_context(
            db,
            tenant_id,
            agent_uuid=effective_agent_uuid,
            ai_name=ai_name,
            ai_platform=ai_platform,
            user_id=user_pk,
        )
        session = get_or_create_session(
            db,
            conversation_id,
            title,
            ai_context["ai_instance_id"],
            source_url,
            tenant_id=tenant_id,
        )

        result = {
            "status": "initialized",
            "session_id": session.id,
            "conversation_id": conversation_id,
            "title": title,
            "ai_name": ai_context.get("canonical_name") or ai_name,
            "ai_platform": ai_context.get("canonical_platform") or ai_platform,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "context": {"tenant_id": tenant_id},
        }
        result.update(
            {
                key: ai_context[key]
                for key in (
                    "agent_uuid",
                    "canonical_name",
                    "canonical_platform",
                    "agent_id_instructions",
                    "agent_id_nag",
                    "needs_user_confirmation",
                    "agent_identity_status",
                )
                if ai_context.get(key) is not None
            }
        )
        return result
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
        tenant_id, user_pk = _context_values(context)
        bytes_to_write = estimate_document_bytes(
            title=title,
            doc_type=doc_type,
            url=url,
            content_summary=content_summary,
            key_concepts=key_concepts,
            metadata=metadata,
        )
        bytes_to_write += estimate_embedding_bytes()
        enforce_quota_or_raise(db, tenant_id=tenant_id, bytes_to_write=bytes_to_write)
        # Parse publication date if provided
        pub_date = None
        if publication_date:
            try:
                pub_date = datetime.fromisoformat(publication_date.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Invalid publication_date format")

        # Create document
        doc_payload = {
            "title": title,
            "doc_type": doc_type,
            "url": url,
            "content_summary": content_summary,
            "publication_date": pub_date,
            "key_concepts": key_concepts or [],
            "metadata_": metadata or {},
            "created_by_user_pk": user_pk,
        }
        if tenant_id:
            doc_payload["tenant_id"] = tenant_id
        doc = Document(**doc_payload)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Generate and store embedding from summary (if enabled)
        _store_embedding(db, "document", doc.id, content_summary, tenant_id=tenant_id)

        record_storage_usage(db, tenant_id=tenant_id, bytes_delta=bytes_to_write)

        return {
            "ack": "SUCCESS",
            "status": "stored",
            "message": f"Document stored successfully (id={doc.id})",
            "id": doc.id,
            "ref": f"document:{doc.id}",
            "title": title,
            "doc_type": doc_type,
            "url": url,
            "key_concepts": key_concepts,
            "publication_date": publication_date,
        }
    finally:
        db.close()
