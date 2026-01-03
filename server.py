"""
MemoryGate - Persistent Memory-as-a-Service for AI Agents
MCP Server with PostgreSQL + pgvector backend
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from sqlalchemy import create_engine, text, func, desc
from sqlalchemy.orm import sessionmaker
import numpy as np

from models import (
    Base, AIInstance, Session, Observation, Pattern, 
    Concept, ConceptAlias, ConceptRelationship, Document, Embedding
)

# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memorygate")

# =============================================================================
# Global State
# =============================================================================

# Database state holder (avoids global scoping issues)
class DB:
    engine = None
    SessionLocal = None

http_client = None  # Reusable HTTP client for OpenAI API


def init_http_client():
    """Initialize HTTP client for OpenAI API calls."""
    global http_client
    http_client = httpx.Client(
        timeout=30.0,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    logger.info("HTTP client initialized")


def cleanup_http_client():
    """Clean up HTTP client on shutdown."""
    global http_client
    if http_client:
        http_client.close()
        logger.info("HTTP client closed")


def init_db():
    """Initialize database connection and create tables."""
    
    logger.info("Connecting to database...")
    DB.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    DB.SessionLocal = sessionmaker(bind=DB.engine)
    
    # FIRST: Ensure pgvector extension exists
    logger.info("Ensuring pgvector extension...")
    with DB.engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Import OAuth models to register tables with Base
    import oauth_models  # noqa: F401
    
    # THEN: Create tables (which depend on vector type)
    logger.info("Creating tables...")
    Base.metadata.create_all(DB.engine)
    
    # Create HNSW index for fast vector search (non-fatal if fails)
    try:
        with DB.engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw 
                ON embeddings USING hnsw (embedding vector_cosine_ops)
            """))
            conn.commit()
        logger.info("HNSW index ready")
    except Exception as e:
        logger.warning(f"Could not create HNSW index (non-fatal): {e}")
    
    logger.info("Database initialized")


async def embed_text(text: str) -> List[float]:
    """Generate embedding using OpenAI API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


def embed_text_sync(text: str) -> List[float]:
    """Synchronous version of embed_text using pooled HTTP client."""
    response = http_client.post(
        "https://api.openai.com/v1/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


# =============================================================================
# Helper Functions
# =============================================================================

def get_or_create_ai_instance(db, name: str, platform: str) -> AIInstance:
    """Get or create an AI instance by name."""
    instance = db.query(AIInstance).filter(AIInstance.name == name).first()
    if not instance:
        instance = AIInstance(name=name, platform=platform)
        db.add(instance)
        db.commit()
        db.refresh(instance)
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
        db.commit()
        db.refresh(session)
    elif title and session.title != title:
        session.title = title
        session.last_active = datetime.utcnow()
        db.commit()
    return session


# =============================================================================
# FastMCP Server
# =============================================================================

mcp = FastMCP("MemoryGate")


@mcp.tool()
def memory_search(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.0,
    domain: Optional[str] = None
) -> dict:
    """
    Unified semantic search across all memory types (observations, patterns, concepts, documents).
    
    Args:
        query: Search query text
        limit: Maximum results to return (default 5)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        domain: Optional domain filter (applies to observations only)
    
    Returns:
        List of matching items from all sources with similarity scores and source_type
    """
    db = SessionLocal()
    try:
        # Generate query embedding
        query_embedding = embed_text_sync(query)
        
        # Unified search across all embedded types
        # Note: pgvector's cast() handles vector conversion natively
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
            ORDER BY e.embedding <=> cast(:embedding as vector)
            LIMIT :limit
        """)
        
        results = db.execute(sql, {
            "embedding": str(query_embedding),  # pgvector handles list conversion
            "min_confidence": min_confidence,
            "domain": domain,
            "limit": limit
        }).fetchall()
        
        # Update access counts for each source type
        for row in results:
            if row.source_type == 'observation':
                db.execute(
                    text("UPDATE observations SET access_count = access_count + 1, last_accessed = NOW() WHERE id = :id"),
                    {"id": row.source_id}
                )
            elif row.source_type == 'pattern':
                db.execute(
                    text("UPDATE patterns SET access_count = access_count + 1, last_accessed = NOW() WHERE id = :id"),
                    {"id": row.source_id}
                )
            elif row.source_type == 'concept':
                db.execute(
                    text("UPDATE concepts SET access_count = access_count + 1, last_accessed = NOW() WHERE id = :id"),
                    {"id": row.source_id}
                )
            elif row.source_type == 'document':
                db.execute(
                    text("UPDATE documents SET access_count = access_count + 1, last_accessed = NOW() WHERE id = :id"),
                    {"id": row.source_id}
                )
        db.commit()
        
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "source_type": row.source_type,
                    "id": row.source_id,
                    "content": row.content,
                    "name": row.item_name,
                    "confidence": row.confidence,
                    "domain": row.domain_or_category,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "metadata": row.metadata,
                    "ai_name": row.ai_name,
                    "session_title": row.session_title,
                    "similarity": float(row.similarity)
                }
                for row in results
            ]
        }
    finally:
        db.close()


@mcp.tool()
def memory_store(
    observation: str,
    confidence: float = 0.8,
    domain: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    ai_name: str = "Unknown",
    ai_platform: str = "Unknown",
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None
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
    db = SessionLocal()
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
        
        # Generate and store embedding
        embedding_vector = embed_text_sync(observation)
        emb = Embedding(
            source_type="observation",
            source_id=obs.id,
            model_version=EMBEDDING_MODEL,
            embedding=embedding_vector,
            normalized=True
        )
        db.add(emb)
        db.commit()
        
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


@mcp.tool()
def memory_recall(
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    ai_name: Optional[str] = None
) -> dict:
    """
    Recall observations by domain and/or confidence filter.
    
    Args:
        domain: Filter by domain/category
        min_confidence: Minimum confidence threshold
        limit: Maximum results (default 10)
        ai_name: Filter by AI instance name
    
    Returns:
        List of matching observations
    """
    db = SessionLocal()
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
        
        results = query.order_by(desc(Observation.timestamp)).limit(limit).all()
        
        # Update access counts
        for obs in results:
            obs.access_count += 1
            obs.last_accessed = datetime.utcnow()
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


@mcp.tool()
def memory_stats() -> dict:
    """
    Get memory system statistics.
    
    Returns:
        Counts and statistics about stored data
    """
    db = SessionLocal()
    try:
        obs_count = db.query(func.count(Observation.id)).scalar()
        pattern_count = db.query(func.count(Pattern.id)).scalar()
        concept_count = db.query(func.count(Concept.id)).scalar()
        document_count = db.query(func.count(Document.id)).scalar()
        session_count = db.query(func.count(Session.id)).scalar()
        ai_count = db.query(func.count(AIInstance.id)).scalar()
        embedding_count = db.query(func.count(Embedding.source_id)).scalar()
        
        # Get AI instances
        ai_instances = db.query(AIInstance).all()
        
        # Get domain distribution
        domains = db.query(
            Observation.domain, func.count(Observation.id)
        ).group_by(Observation.domain).all()
        
        return {
            "status": "healthy",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "counts": {
                "observations": obs_count,
                "patterns": pattern_count,
                "concepts": concept_count,
                "documents": document_count,
                "sessions": session_count,
                "ai_instances": ai_count,
                "embeddings": embedding_count
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


@mcp.tool()
def memory_init_session(
    conversation_id: str,
    title: str,
    ai_name: str,
    ai_platform: str,
    source_url: Optional[str] = None
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
    db = SessionLocal()
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


@mcp.tool()
def memory_store_document(
    title: str,
    doc_type: str,
    url: str,
    content_summary: str,
    key_concepts: Optional[List[str]] = None,
    publication_date: Optional[str] = None,
    metadata: Optional[dict] = None
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
    db = SessionLocal()
    try:
        # Parse publication date if provided
        pub_date = None
        if publication_date:
            try:
                pub_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid publication_date format: {publication_date}")
        
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
        
        # Generate and store embedding from summary
        embedding_vector = embed_text_sync(content_summary)
        emb = Embedding(
            source_type="document",
            source_id=doc.id,
            model_version=EMBEDDING_MODEL,
            embedding=embedding_vector,
            normalized=True
        )
        db.add(emb)
        db.commit()
        
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


@mcp.tool()
def memory_store_concept(
    name: str,
    concept_type: str,
    description: str,
    domain: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[dict] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None
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
    db = SessionLocal()
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
        
        # Generate and store embedding from description
        embedding_vector = embed_text_sync(description)
        emb = Embedding(
            source_type="concept",
            source_id=concept.id,
            model_version=EMBEDDING_MODEL,
            embedding=embedding_vector,
            normalized=True
        )
        db.add(emb)
        db.commit()
        
        return {
            "status": "stored",
            "id": concept.id,
            "name": name,
            "type": concept_type,
            "description": description
        }
    finally:
        db.close()


@mcp.tool()
def memory_get_concept(name: str) -> dict:
    """
    Get a concept by name (case-insensitive, alias-aware).
    
    Args:
        name: Concept name or alias to look up
    
    Returns:
        Concept details or None if not found
    """
    db = SessionLocal()
    try:
        name_key = name.lower()
        
        # Try direct lookup first
        concept = db.query(Concept).filter(Concept.name_key == name_key).first()
        
        # If not found, check aliases
        if not concept:
            from models import ConceptAlias
            alias = db.query(ConceptAlias).filter(ConceptAlias.alias_key == name_key).first()
            if alias:
                concept = db.query(Concept).filter(Concept.id == alias.concept_id).first()
        
        if not concept:
            return {"status": "not_found", "name": name}
        
        # Update access tracking
        concept.access_count += 1
        concept.last_accessed = datetime.utcnow()
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


@mcp.tool()
def memory_add_concept_alias(concept_name: str, alias: str) -> dict:
    """
    Add an alternative name (alias) for a concept.
    
    Args:
        concept_name: Primary concept name
        alias: Alternative name to add
    
    Returns:
        Status of alias creation
    """
    db = SessionLocal()
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


@mcp.tool()
def memory_add_concept_relationship(
    from_concept: str,
    to_concept: str,
    rel_type: str,
    weight: float = 0.5,
    description: Optional[str] = None
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
    db = SessionLocal()
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


@mcp.tool()
def memory_related_concepts(
    concept_name: str,
    rel_type: Optional[str] = None,
    min_weight: float = 0.0
) -> dict:
    """
    Get concepts related to a given concept.
    
    Args:
        concept_name: Concept to find relationships for
        rel_type: Optional filter by relationship type
        min_weight: Minimum relationship weight (default 0.0)
    
    Returns:
        List of related concepts with relationship details
    """
    db = SessionLocal()
    try:
        from models import ConceptRelationship
        
        # Find the concept
        concept_key = concept_name.lower()
        concept = db.query(Concept).filter(Concept.name_key == concept_key).first()
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


@mcp.tool()
def memory_update_pattern(
    category: str,
    pattern_name: str,
    pattern_text: str,
    confidence: float = 0.8,
    evidence_observation_ids: Optional[List[int]] = None,
    ai_name: Optional[str] = None,
    ai_platform: Optional[str] = None,
    conversation_id: Optional[str] = None
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
    db = SessionLocal()
    try:
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
            
            # Update embedding
            embedding_vector = embed_text_sync(pattern_text)
            
            # Delete old embedding
            db.query(Embedding).filter(
                Embedding.source_type == 'pattern',
                Embedding.source_id == existing.id
            ).delete()
            
            # Create new embedding
            emb = Embedding(
                source_type="pattern",
                source_id=existing.id,
                model_version=EMBEDDING_MODEL,
                embedding=embedding_vector,
                normalized=True
            )
            db.add(emb)
            db.commit()
            
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
            
            # Generate and store embedding
            embedding_vector = embed_text_sync(pattern_text)
            emb = Embedding(
                source_type="pattern",
                source_id=pattern.id,
                model_version=EMBEDDING_MODEL,
                embedding=embedding_vector,
                normalized=True
            )
            db.add(emb)
            db.commit()
            
            return {
                "status": "created",
                "id": pattern.id,
                "category": category,
                "pattern_name": pattern_name,
                "confidence": confidence
            }
    finally:
        db.close()


@mcp.tool()
def memory_get_pattern(category: str, pattern_name: str) -> dict:
    """
    Get a specific pattern by category and name.
    
    Args:
        category: Pattern category
        pattern_name: Pattern name within category
    
    Returns:
        Pattern details or not_found status
    """
    db = SessionLocal()
    try:
        pattern = db.query(Pattern).filter(
            Pattern.category == category,
            Pattern.pattern_name == pattern_name
        ).first()
        
        if not pattern:
            return {
                "status": "not_found",
                "category": category,
                "pattern_name": pattern_name
            }
        
        # Update access tracking
        pattern.access_count += 1
        pattern.last_accessed = datetime.utcnow()
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


@mcp.tool()
def memory_patterns(
    category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 20
) -> dict:
    """
    List patterns with optional filtering by category and confidence.
    
    Args:
        category: Optional category filter
        min_confidence: Minimum confidence threshold (default 0.0)
        limit: Maximum results (default 20)
    
    Returns:
        List of matching patterns
    """
    db = SessionLocal()
    try:
        query = db.query(Pattern)
        
        if category:
            query = query.filter(Pattern.category == category)
        if min_confidence > 0:
            query = query.filter(Pattern.confidence >= min_confidence)
        
        results = query.order_by(desc(Pattern.last_updated)).limit(limit).all()
        
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
# FastAPI App
# =============================================================================

# Create MCP ASGI app
mcp_app = mcp.http_app(
    path="/",
    transport="sse",
    stateless_http=True,
    json_response=True,
)


# =============================================================================
# Pure ASGI wrapper: Normalize /mcp to /mcp/ without buffering
# =============================================================================

class SlashNormalizerASGI:
    """Pure ASGI middleware - no response buffering, SSE-safe."""
    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = dict(scope)  # Make mutable copy
            scope["path"] = "/mcp/"
        await self.wrapped_app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown."""
    init_db()
    init_http_client()
    yield
    cleanup_http_client()


app = FastAPI(title="MemoryGate", redirect_slashes=False, lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("FRONTEND_URL", "http://localhost:3000"),
        "http://localhost:3000",
        "https://memorygate.ai",
        "https://www.memorygate.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount OAuth routes
from oauth_routes import router as auth_router
from mcp_auth_gate import MCPAuthGateASGI
app.include_router(auth_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "memorygate"}


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "MemoryGate",
        "version": "0.1.0",
        "description": "Persistent Memory-as-a-Service for AI Agents",
        "embedding_model": EMBEDDING_MODEL,
        "endpoints": {
            "health": "/health",
            "mcp": "/mcp",
            "auth": {
                "client_credentials": "/auth/client",
                "login_google": "/auth/login/google",
                "login_github": "/auth/login/github",
                "me": "/auth/me",
                "api_keys": "/auth/api-keys"
            }
        }
    }


# Mount MCP app at /mcp/ with auth gate (pass DB class for dynamic lookup)
app.mount("/mcp/", MCPAuthGateASGI(mcp_app, lambda: DB.SessionLocal))


# =============================================================================
# ASGI Application (module-level for production deployment)
# =============================================================================

# Wrap entire app with slash normalizer to handle /mcp -> /mcp/
asgi_app = SlashNormalizerASGI(app)


# =============================================================================
# Main (for local development only)
# =============================================================================

if __name__ == "__main__":
    print("MemoryGate starting...")
    uvicorn.run(asgi_app, host="0.0.0.0", port=8080)
