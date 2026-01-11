"""
MemoryGate Database Models
PostgreSQL + pgvector schema
"""

from datetime import datetime
from enum import Enum as PyEnum
import os
import uuid
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Float, Boolean,
    DateTime, ForeignKey, CheckConstraint, Index, UniqueConstraint, Enum, JSON
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, declarative_base

DB_BACKEND = os.environ.get("DB_BACKEND", "postgres").strip().lower()
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "pgvector").strip().lower()

DB_BACKEND_EFFECTIVE = DB_BACKEND if DB_BACKEND in {"postgres", "sqlite"} else "postgres"
VECTOR_BACKEND_EFFECTIVE = (
    VECTOR_BACKEND if VECTOR_BACKEND in {"pgvector", "sqlite_vss", "none"} else "none"
)
if DB_BACKEND_EFFECTIVE == "sqlite" and VECTOR_BACKEND_EFFECTIVE == "pgvector":
    VECTOR_BACKEND_EFFECTIVE = "none"
if DB_BACKEND_EFFECTIVE == "postgres" and VECTOR_BACKEND_EFFECTIVE == "sqlite_vss":
    VECTOR_BACKEND_EFFECTIVE = "none"

try:
    from pgvector.sqlalchemy import Vector as PgVector
    PGVECTOR_AVAILABLE = True
except Exception:
    PgVector = None
    PGVECTOR_AVAILABLE = False

if (
    DB_BACKEND_EFFECTIVE == "postgres"
    and VECTOR_BACKEND_EFFECTIVE == "pgvector"
    and PGVECTOR_AVAILABLE
):
    EMBEDDING_COLUMN_TYPE = PgVector(1536)
else:
    EMBEDDING_COLUMN_TYPE = JSON

JSON_TYPE = JSONB if DB_BACKEND_EFFECTIVE == "postgres" else JSON
UUID_TYPE = UUID(as_uuid=True) if DB_BACKEND_EFFECTIVE == "postgres" else String(36)


def _uuid_default() -> str | uuid.UUID:
    value = uuid.uuid4()
    return value if DB_BACKEND_EFFECTIVE == "postgres" else str(value)

Base = declarative_base()

# =============================================================================
# Enums
# =============================================================================

class MemoryTier(str, PyEnum):
    hot = "hot"
    cold = "cold"


class TombstoneAction(str, PyEnum):
    archived = "archived"
    rehydrated = "rehydrated"
    purged = "purged"
    summarized = "summarized"


# =============================================================================
# AI Instances (Kee, Hexy, etc.)
# =============================================================================

class AIInstance(Base):
    __tablename__ = "ai_instances"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # "Kee", "Hexy"
    platform = Column(String(100), nullable=False)  # "Claude", "ChatGPT"
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="ai_instance")
    observations = relationship("Observation", back_populates="ai_instance")
    patterns = relationship("Pattern", back_populates="ai_instance")
    concepts = relationship("Concept", back_populates="ai_instance")


# =============================================================================
# Sessions (Conversations)
# =============================================================================

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(255), unique=True)  # UUID from chat URL
    title = Column(String(500))
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    source_url = Column(String(1000))
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_active = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = Column(Text)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    
    # Relationships
    ai_instance = relationship("AIInstance", back_populates="sessions")
    observations = relationship("Observation", back_populates="session")
    patterns = relationship("Pattern", back_populates="session")


# =============================================================================
# Observations
# =============================================================================

class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    observation = Column(Text, nullable=False)
    confidence = Column(Float, default=0.8)
    domain = Column(String(100))
    evidence = Column(JSON_TYPE, default=list)
    
    # Provenance
    session_id = Column(Integer, ForeignKey("sessions.id"))
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    
    # Access tracking
    access_count = Column(BigInteger, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))

    # Tiering & retention
    tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False), default=MemoryTier.hot, nullable=False)
    archived_at = Column(DateTime(timezone=True))
    archived_reason = Column(Text)
    archived_by = Column(String(100))
    score = Column(Float, default=0.0, nullable=False)
    floor_score = Column(Float, default=-9999.0, nullable=False)
    purge_eligible = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="observations")
    ai_instance = relationship("AIInstance", back_populates="observations")
    
    __table_args__ = (
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence'),
        Index('ix_observations_domain', 'domain'),
        Index('ix_observations_confidence', 'confidence'),
        Index('ix_observations_tier', 'tier'),
        Index('ix_observations_score', 'score'),
    )


# =============================================================================
# Patterns
# =============================================================================

class Pattern(Base):
    __tablename__ = "patterns"
    
    id = Column(Integer, primary_key=True)
    category = Column(String(100), nullable=False)
    pattern_name = Column(String(255), nullable=False)
    pattern_text = Column(Text, nullable=False)
    confidence = Column(Float, default=0.8)
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    evidence_observation_ids = Column(JSON_TYPE, default=list)
    
    # Provenance
    session_id = Column(Integer, ForeignKey("sessions.id"))
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    
    # Access tracking
    access_count = Column(BigInteger, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))

    # Tiering & retention
    tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False), default=MemoryTier.hot, nullable=False)
    archived_at = Column(DateTime(timezone=True))
    archived_reason = Column(Text)
    archived_by = Column(String(100))
    score = Column(Float, default=0.0, nullable=False)
    floor_score = Column(Float, default=-9999.0, nullable=False)
    purge_eligible = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="patterns")
    ai_instance = relationship("AIInstance", back_populates="patterns")
    
    __table_args__ = (
        UniqueConstraint('category', 'pattern_name', name='uq_pattern_category_name'),
        Index('ix_patterns_category', 'category'),
        Index('ix_patterns_tier', 'tier'),
        Index('ix_patterns_score', 'score'),
    )


# =============================================================================
# Concepts (Knowledge Graph)
# =============================================================================

class Concept(Base):
    __tablename__ = "concepts"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # Original case preserved
    name_key = Column(String(255), nullable=False)  # Lowercase for lookups
    type = Column(String(50), nullable=False)  # project/framework/component/construct/theory
    status = Column(String(50))
    domain = Column(String(100))
    description = Column(Text)  # Used for embedding
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    
    # Provenance
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Access tracking
    access_count = Column(BigInteger, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))

    # Tiering & retention
    tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False), default=MemoryTier.hot, nullable=False)
    archived_at = Column(DateTime(timezone=True))
    archived_reason = Column(Text)
    archived_by = Column(String(100))
    score = Column(Float, default=0.0, nullable=False)
    floor_score = Column(Float, default=-9999.0, nullable=False)
    purge_eligible = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    ai_instance = relationship("AIInstance", back_populates="concepts")
    aliases = relationship("ConceptAlias", back_populates="concept", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_concepts_name_key', 'name_key'),
        Index('ix_concepts_type', 'type'),
        Index('ix_concepts_tier', 'tier'),
        Index('ix_concepts_score', 'score'),
    )


class ConceptAlias(Base):
    __tablename__ = "concept_aliases"
    
    alias = Column(String(255), primary_key=True)
    alias_key = Column(String(255), nullable=False)  # Lowercase for lookups
    concept_id = Column(Integer, ForeignKey("concepts.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    concept = relationship("Concept", back_populates="aliases")
    
    __table_args__ = (
        Index('ix_concept_aliases_alias_key', 'alias_key'),
    )


class ConceptRelationship(Base):
    __tablename__ = "concept_relationships"
    
    from_concept_id = Column(Integer, ForeignKey("concepts.id"), primary_key=True)
    to_concept_id = Column(Integer, ForeignKey("concepts.id"), primary_key=True)
    rel_type = Column(String(50), primary_key=True)  # enables/version_of/part_of/related_to/implements/demonstrates
    weight = Column(Float, default=0.5)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint('weight >= 0 AND weight <= 1', name='check_rel_weight'),
    )


# =============================================================================
# Documents
# =============================================================================

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    doc_type = Column(String(50), nullable=False)
    content_summary = Column(Text)
    url = Column(String(1000))
    publication_date = Column(DateTime(timezone=True))
    key_concepts = Column(JSON_TYPE, default=list)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    
    # Access tracking
    access_count = Column(BigInteger, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))

    # Tiering & retention
    tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False), default=MemoryTier.hot, nullable=False)
    archived_at = Column(DateTime(timezone=True))
    archived_reason = Column(Text)
    archived_by = Column(String(100))
    score = Column(Float, default=0.0, nullable=False)
    floor_score = Column(Float, default=-9999.0, nullable=False)
    purge_eligible = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index('ix_documents_tier', 'tier'),
        Index('ix_documents_score', 'score'),
    )


# =============================================================================
# Summaries
# =============================================================================

class MemorySummary(Base):
    __tablename__ = "memory_summaries"

    id = Column(Integer, primary_key=True)
    source_type = Column(String(50), nullable=False)
    source_id = Column(Integer, nullable=True)
    source_ids = Column(JSON_TYPE, default=list)
    summary_text = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Access tracking
    access_count = Column(BigInteger, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))

    # Tiering & retention
    tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False), default=MemoryTier.hot, nullable=False)
    archived_at = Column(DateTime(timezone=True))
    archived_reason = Column(Text)
    archived_by = Column(String(100))
    score = Column(Float, default=0.0, nullable=False)
    floor_score = Column(Float, default=-9999.0, nullable=False)
    purge_eligible = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index('ix_memory_summaries_source', 'source_type', 'source_id'),
        Index('ix_memory_summaries_tier', 'tier'),
        Index('ix_memory_summaries_score', 'score'),
    )


# =============================================================================
# Tombstones
# =============================================================================

class MemoryTombstone(Base):
    __tablename__ = "memory_tombstones"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    memory_id = Column(String(255), nullable=False)
    action = Column(Enum(TombstoneAction, name="tombstone_action", create_type=False), nullable=False)
    from_tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False))
    to_tier = Column(Enum(MemoryTier, name="memory_tier", create_type=False))
    reason = Column(Text)
    actor = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)

    __table_args__ = (
        Index('ix_memory_tombstones_memory_id', 'memory_id'),
        Index('ix_memory_tombstones_action', 'action'),
    )


# =============================================================================
# Embeddings (Unified)
# =============================================================================

class Embedding(Base):
    __tablename__ = "embeddings"
    
    source_type = Column(String(50), primary_key=True)  # observation/pattern/concept/document
    source_id = Column(Integer, primary_key=True)
    model_version = Column(String(100), primary_key=True, default="all-MiniLM-L6-v2")
    embedding = Column(EMBEDDING_COLUMN_TYPE, nullable=False)  # OpenAI text-embedding-3-small
    normalized = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_embeddings_source', 'source_type', 'source_id'),
    )


# =============================================================================
# Memory model registry
# =============================================================================

MEMORY_MODELS = {
    "observation": Observation,
    "pattern": Pattern,
    "concept": Concept,
    "document": Document,
}
