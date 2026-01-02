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
from fastmcp import FastMCP
from sqlalchemy import create_engine, text, func, desc
from sqlalchemy.orm import sessionmaker
import numpy as np

from models import (
    Base, AIInstance, Session, Observation, Pattern, 
    Concept, ConceptAlias, Document, Embedding
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

engine = None
SessionLocal = None


def init_db():
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    logger.info("Connecting to database...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)
    
    # FIRST: Ensure pgvector extension exists
    logger.info("Ensuring pgvector extension...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # THEN: Create tables (which depend on vector type)
    logger.info("Creating tables...")
    Base.metadata.create_all(engine)
    
    # Create HNSW index for fast vector search (non-fatal if fails)
    try:
        with engine.connect() as conn:
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
    """Synchronous version of embed_text."""
    with httpx.Client() as client:
        response = client.post(
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
    Semantic search across observations.
    
    Args:
        query: Search query text
        limit: Maximum results to return (default 5)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        domain: Optional domain filter
    
    Returns:
        List of matching observations with similarity scores
    """
    db = SessionLocal()
    try:
        # Generate query embedding
        query_embedding = embed_text_sync(query)
        
        # Format embedding for pgvector
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Build query with vector similarity using explicit cast
        sql = text("""
            SELECT 
                o.id,
                o.observation,
                o.confidence,
                o.domain,
                o.timestamp,
                o.evidence,
                ai.name as ai_name,
                s.title as session_title,
                1 - (e.embedding <=> (:embedding)::vector) as similarity
            FROM observations o
            JOIN embeddings e ON e.source_type = 'observation' AND e.source_id = o.id
            LEFT JOIN ai_instances ai ON o.ai_instance_id = ai.id
            LEFT JOIN sessions s ON o.session_id = s.id
            WHERE o.confidence >= :min_confidence
            AND (:domain IS NULL OR o.domain = :domain)
            ORDER BY e.embedding <=> (:embedding)::vector
            LIMIT :limit
        """)
        
        results = db.execute(sql, {
            "embedding": embedding_str,
            "min_confidence": min_confidence,
            "domain": domain,
            "limit": limit
        }).fetchall()
        
        # Update access counts
        for row in results:
            db.execute(
                text("UPDATE observations SET access_count = access_count + 1, last_accessed = NOW() WHERE id = :id"),
                {"id": row.id}
            )
        db.commit()
        
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "id": row.id,
                    "observation": row.observation,
                    "confidence": row.confidence,
                    "domain": row.domain,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "evidence": row.evidence,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    init_db()
    yield


app = FastAPI(title="MemoryGate", redirect_slashes=False, lifespan=lifespan)


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
            "mcp": "/mcp"
        }
    }


# Mount MCP app at /mcp (FastMCP will handle SSE at this path)
app.mount("/mcp", mcp_app)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("MemoryGate starting...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
