"""
MemoryGate Database Models
PostgreSQL + pgvector schema
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, CheckConstraint, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

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
    metadata_ = Column("metadata", JSONB, default={})
    
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
    evidence = Column(JSONB, default=[])
    
    # Provenance
    session_id = Column(Integer, ForeignKey("sessions.id"))
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    
    # Access tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("Session", back_populates="observations")
    ai_instance = relationship("AIInstance", back_populates="observations")
    
    __table_args__ = (
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence'),
        Index('ix_observations_domain', 'domain'),
        Index('ix_observations_confidence', 'confidence'),
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
    evidence_observation_ids = Column(JSONB, default=[])
    
    # Provenance
    session_id = Column(Integer, ForeignKey("sessions.id"))
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    
    # Access tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("Session", back_populates="patterns")
    ai_instance = relationship("AIInstance", back_populates="patterns")
    
    __table_args__ = (
        UniqueConstraint('category', 'pattern_name', name='uq_pattern_category_name'),
        Index('ix_patterns_category', 'category'),
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
    metadata_ = Column("metadata", JSONB, default={})
    
    # Provenance
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Access tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    ai_instance = relationship("AIInstance", back_populates="concepts")
    aliases = relationship("ConceptAlias", back_populates="concept", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_concepts_name_key', 'name_key'),
        Index('ix_concepts_type', 'type'),
    )


class ConceptAlias(Base):
    __tablename__ = "concept_aliases"
    
    alias = Column(String(255), primary_key=True)
    concept_id = Column(Integer, ForeignKey("concepts.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    concept = relationship("Concept", back_populates="aliases")


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
    key_concepts = Column(JSONB, default=[])
    metadata_ = Column("metadata", JSONB, default={})
    
    # Access tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


# =============================================================================
# Embeddings (Unified)
# =============================================================================

class Embedding(Base):
    __tablename__ = "embeddings"
    
    source_type = Column(String(50), primary_key=True)  # observation/pattern/concept/document
    source_id = Column(Integer, primary_key=True)
    model_version = Column(String(100), primary_key=True, default="all-MiniLM-L6-v2")
    embedding = Column(Vector(384), nullable=False)
    normalized = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_embeddings_source', 'source_type', 'source_id'),
    )
