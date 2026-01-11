"""
Memory storage and session services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy.exc import IntegrityError

from core.context import RequestContext
from core.db import DB
from core.models import AIInstance, Session, Observation, Document
from core.services.memory_shared import (
    _store_embedding,
    _validate_required_text,
    _validate_optional_text,
    _validate_confidence,
    _validate_string_list,
    _validate_metadata,
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
