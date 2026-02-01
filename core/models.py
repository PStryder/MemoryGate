"""
MemoryGate Database Models
PostgreSQL + pgvector schema
"""

from datetime import datetime
from enum import Enum as PyEnum
import uuid
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Float, Boolean,
    DateTime, ForeignKey, CheckConstraint, Index, UniqueConstraint, Enum, JSON, event
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, declarative_base

import core.config as config
from core.errors import ValidationIssue

DB_BACKEND_EFFECTIVE = config.DB_BACKEND_EFFECTIVE
VECTOR_BACKEND_EFFECTIVE = config.VECTOR_BACKEND_EFFECTIVE
DEFAULT_TENANT_ID = config.DEFAULT_TENANT_ID

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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    name = Column(String(100), nullable=False)  # "Kee", "Hexy"
    platform = Column(String(100), nullable=False)  # "Claude", "ChatGPT"
    canonical_name = Column(String(100))
    canonical_platform = Column(String(100))
    agent_uuid = Column(String(40))
    legacy_agent_uuid = Column(String(36))
    user_id = Column(String(100))
    description = Column(Text)
    tags = Column(JSON_TYPE, default=list)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen_raw_name = Column(String(100))
    last_seen_raw_platform = Column(String(100))
    last_seen_at = Column(DateTime(timezone=True))
    is_archived = Column(Boolean, default=False, nullable=False)
    merged_into_ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"))
    
    # Relationships
    sessions = relationship("Session", back_populates="ai_instance")
    observations = relationship("Observation", back_populates="ai_instance")
    patterns = relationship("Pattern", back_populates="ai_instance")
    concepts = relationship("Concept", back_populates="ai_instance")

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "platform", name="uq_ai_instances_tenant_name_platform"),
        UniqueConstraint("tenant_id", "agent_uuid", name="uq_ai_instances_tenant_agent_uuid"),
    )


# =============================================================================
# AI Memory Topology
# =============================================================================

class AIMemoryPolicy(Base):
    __tablename__ = "ai_memory_policy"

    tenant_id = Column(
        String(100),
        primary_key=True,
        nullable=False,
        default=DEFAULT_TENANT_ID,
        server_default=DEFAULT_TENANT_ID,
    )
    ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"), primary_key=True, nullable=False)
    mode = Column(String(20), nullable=False)


class AIEntityShare(Base):
    __tablename__ = "ai_entity_shares"

    tenant_id = Column(
        String(100),
        primary_key=True,
        nullable=False,
        default=DEFAULT_TENANT_ID,
        server_default=DEFAULT_TENANT_ID,
    )
    entity_type = Column(String(50), primary_key=True, nullable=False)
    entity_id = Column(Integer, primary_key=True, nullable=False)
    shared_with_ai_instance_id = Column(Integer, ForeignKey("ai_instances.id"), primary_key=True, nullable=False)
    shared_by_user_id = Column(UUID_TYPE)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ai_entity_shares_tenant_ai", "tenant_id", "shared_with_ai_instance_id"),
    )


# =============================================================================
# Sessions (Conversations)
# =============================================================================

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    conversation_id = Column(String(255))  # UUID from chat URL
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

    __table_args__ = (
        UniqueConstraint("tenant_id", "conversation_id", name="uq_sessions_tenant_conversation_id"),
    )


# =============================================================================
# Observations
# =============================================================================

class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_pk = Column(UUID_TYPE)
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_pk = Column(UUID_TYPE)
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
        UniqueConstraint('tenant_id', 'category', 'pattern_name', name='uq_patterns_tenant_category_name'),
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_pk = Column(UUID_TYPE)
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
        UniqueConstraint('tenant_id', 'name_key', name='uq_concepts_tenant_name_key'),
        Index('ix_concepts_name_key', 'name_key'),
        Index('ix_concepts_type', 'type'),
        Index('ix_concepts_tier', 'tier'),
        Index('ix_concepts_score', 'score'),
    )


class ConceptAlias(Base):
    __tablename__ = "concept_aliases"
    
    alias = Column(String(255), primary_key=True)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_pk = Column(UUID_TYPE)
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_pk = Column(UUID_TYPE)
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
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
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
# Audit Events
# =============================================================================

class AuditEvent(Base):
    __tablename__ = "audit_events"

    event_id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    event_type = Column(String(100), nullable=False)
    event_version = Column(Integer, default=1, nullable=False)
    actor_type = Column(String(50), nullable=False)
    actor_id = Column(String(255))
    org_id = Column(String(255))
    user_id = Column(String(255))
    target_type = Column(String(50), nullable=False)
    target_ids = Column(JSON_TYPE, nullable=False)
    count_affected = Column(Integer)
    reason = Column(Text)
    request_id = Column(String(255))
    metadata_ = Column("metadata", JSON_TYPE)

    __table_args__ = (
        Index("ix_audit_events_created_at", "created_at"),
        Index("ix_audit_events_event_type", "event_type"),
        Index("ix_audit_events_org_id", "org_id"),
        Index("ix_audit_events_user_id", "user_id"),
    )


# =============================================================================
# Archived Memories
# =============================================================================

class ArchivedMemory(Base):
    __tablename__ = "archived_memories"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    source_type = Column(String(50), nullable=False)  # observation/pattern/concept/document/summary
    source_id = Column(Integer, nullable=True)
    payload = Column(JSON_TYPE, nullable=False)
    archived_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    purge_reason = Column(Text)
    purge_actor = Column(String(100))
    expires_at = Column(DateTime(timezone=True))
    size_bytes_estimate = Column(BigInteger, default=0, nullable=False)
    original_embedding = Column(JSON_TYPE)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_archived_memories_source"),
        Index("ix_archived_memories_source_type", "source_type"),
        Index("ix_archived_memories_archived_at", "archived_at"),
        Index("ix_archived_memories_expires_at", "expires_at"),
    )


# =============================================================================
# Embeddings (Unified)
# =============================================================================

class Embedding(Base):
    __tablename__ = "embeddings"
    
    source_type = Column(String(50), primary_key=True)  # observation/pattern/concept/document
    source_id = Column(Integer, primary_key=True)
    model_version = Column(String(100), primary_key=True, default="all-MiniLM-L6-v2")
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    embedding = Column(EMBEDDING_COLUMN_TYPE, nullable=False)  # OpenAI text-embedding-3-small
    normalized = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_embeddings_source', 'source_type', 'source_id'),
    )


# =============================================================================
# Memory model registry
# =============================================================================

# =============================================================================
# Memory Chains
# =============================================================================

class MemoryChain(Base):
    __tablename__ = "memory_chains"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    created_by_user_id = Column(UUID_TYPE, nullable=True)
    created_by_type = Column(String(50), nullable=True)
    created_by_id = Column(String(255), nullable=True)
    title = Column(Text, nullable=True)
    kind = Column(String(100), nullable=True)  # anchor, decision, investigation, workflow
    meta = Column(JSON_TYPE, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    items = relationship(
        "MemoryChainItem",
        back_populates="chain",
        cascade="all, delete-orphan",
        order_by="MemoryChainItem.seq",
    )

    __table_args__ = (
        Index("ix_memory_chains_tenant_id", "tenant_id"),
        Index("ix_memory_chains_kind", "kind"),
    )


class MemoryChainItem(Base):
    __tablename__ = "memory_chain_items"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    chain_id = Column(UUID_TYPE, ForeignKey("memory_chains.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(String(255), nullable=False)  # "type:id" format
    observation_id = Column(Integer, ForeignKey("observations.id"), nullable=True)
    seq = Column(Integer, nullable=False)
    role = Column(String(100), nullable=True)  # discovery, analysis, resolution, etc.
    linked_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    chain = relationship("MemoryChain", back_populates="items")

    __table_args__ = (
        UniqueConstraint("tenant_id", "chain_id", "seq", name="uq_memory_chain_items_seq"),
        UniqueConstraint("tenant_id", "chain_id", "memory_id", name="uq_memory_chain_items_memory"),
        Index("ix_memory_chain_items_chain_seq", "tenant_id", "chain_id", "seq"),
        Index("ix_memory_chain_items_memory", "tenant_id", "memory_id"),
    )


# =============================================================================
# Anchor Pointers (for agent profiles)
# =============================================================================

class AnchorPointer(Base):
    __tablename__ = "anchor_pointers"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    scope_type = Column(String(50), nullable=False)  # "agent", "user", etc.
    scope_key = Column(String(255), nullable=False)  # "platform::name" for agents
    anchor_kind = Column(String(50), nullable=False)  # "agent_profile"
    chain_id = Column(UUID_TYPE, nullable=False)
    chain_head_seq = Column(Integer, nullable=True)
    meta = Column(JSON_TYPE, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "scope_type",
            "scope_key",
            "anchor_kind",
            name="uq_anchor_pointers_scope",
        ),
        Index("ix_anchor_pointers_tenant_scope", "tenant_id", "scope_type", "scope_key"),
    )


# =============================================================================
# Memory Relationships (generic edges between any memory types)
# =============================================================================

class MemoryRelationship(Base):
    __tablename__ = "memory_relationships"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    from_type = Column(Text, nullable=False)  # observation, pattern, concept, etc.
    from_id = Column(Text, nullable=False)
    to_type = Column(Text, nullable=False)
    to_id = Column(Text, nullable=False)
    rel_type = Column(Text, nullable=False)  # supersedes, enables, related_to, etc.
    weight = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON_TYPE, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    created_by_type = Column(Text, nullable=True)
    created_by_id = Column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("from_type != '' AND to_type != ''", name="ck_memory_relationships_types"),
        CheckConstraint(
            "from_id != to_id OR from_type != to_type",
            name="ck_memory_relationships_distinct",
        ),
        UniqueConstraint(
            "tenant_id",
            "from_type",
            "from_id",
            "to_type",
            "to_id",
            "rel_type",
            name="uq_memory_relationships_unique",
        ),
        Index("ix_memory_relationships_from", "tenant_id", "from_type", "from_id"),
        Index("ix_memory_relationships_to", "tenant_id", "to_type", "to_id"),
        Index("ix_memory_relationships_type", "tenant_id", "rel_type"),
    )


# =============================================================================
# Relationship Residue (compression friction tracking)
# =============================================================================

class RelationshipResidue(Base):
    __tablename__ = "relationship_residue"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    edge_id = Column(UUID_TYPE, ForeignKey("memory_relationships.id"), nullable=False)
    event_type = Column(Text, nullable=False)  # created, revised, divergent, confidence_update, merged
    actor = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    encoded_rel_type = Column(Text, nullable=False)
    encoded_weight = Column(Float, nullable=True)
    alternatives_considered = Column(JSON_TYPE, nullable=True)
    alternatives_ruled_out = Column(JSON_TYPE, nullable=True)
    friction_metrics = Column(JSON_TYPE, nullable=True)
    compression_texture = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_relationship_residue_edge_created_at", "edge_id", "created_at"),
        Index("ix_relationship_residue_event_type", "event_type"),
        Index("ix_relationship_residue_actor", "actor"),
    )


# =============================================================================
# Memory Supersession (immutable correction tracking)
# =============================================================================

class MemorySupersede(Base):
    __tablename__ = "memory_supersedes"

    id = Column(UUID_TYPE, primary_key=True, default=_uuid_default)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    from_memory_id = Column(String(255), nullable=False)  # "type:id" being superseded
    to_memory_id = Column(String(255), nullable=False)  # "type:id" replacement
    relation_type = Column(String(50), nullable=False, default="supersedes")
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("from_memory_id != to_memory_id", name="ck_memory_supersedes_distinct"),
        UniqueConstraint(
            "tenant_id",
            "from_memory_id",
            "relation_type",
            name="uq_memory_supersedes_from_relation",
        ),
        Index("ix_memory_supersedes_from", "tenant_id", "from_memory_id"),
        Index("ix_memory_supersedes_to", "tenant_id", "to_memory_id"),
    )


# =============================================================================
# Store Management
# =============================================================================

class UserActiveStore(Base):
    """Tracks which store a user has currently active."""
    __tablename__ = "user_active_stores"

    user_id = Column(UUID_TYPE, primary_key=True)
    tenant_id = Column(String(100), nullable=False, default=DEFAULT_TENANT_ID, server_default=DEFAULT_TENANT_ID)
    active_store_id = Column(String(255), nullable=True)  # None means personal/default store
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# =============================================================================
# Tenant Storage Usage (standalone quota tracking)
# =============================================================================

class TenantStorageUsage(Base):
    __tablename__ = "tenant_storage_usage"

    tenant_id = Column(
        String(100),
        primary_key=True,
        nullable=False,
        default=DEFAULT_TENANT_ID,
        server_default=DEFAULT_TENANT_ID,
    )
    storage_used_bytes = Column(BigInteger, default=0, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


MEMORY_MODELS = {
    "observation": Observation,
    "pattern": Pattern,
    "concept": Concept,
    "document": Document,
}

MEMORY_RELATIONSHIP_MODELS = {
    "observation": Observation,
    "pattern": Pattern,
    "concept": Concept,
    "document": Document,
    "summary": MemorySummary,
    "relationship": MemoryRelationship,
    "residue": RelationshipResidue,
}

@event.listens_for(Base, "before_insert", propagate=True)
def _validate_tenant_id_before_insert(mapper, connection, target) -> None:
    if not hasattr(target, "tenant_id"):
        return
    if config.TENANCY_MODE != config.TENANCY_REQUIRED:
        return
    tenant_id = getattr(target, "tenant_id", None)
    if not tenant_id or tenant_id == DEFAULT_TENANT_ID:
        raise ValidationIssue(
            "tenant_id is required for this operation",
            field="tenant_id",
            error_type="required",
        )

__all__ = [
    "Base",
    "MemoryTier",
    "TombstoneAction",
    "AIInstance",
    "AIMemoryPolicy",
    "AIEntityShare",
    "Session",
    "Observation",
    "Pattern",
    "Concept",
    "ConceptAlias",
    "ConceptRelationship",
    "Document",
    "MemorySummary",
    "MemoryTombstone",
    "AuditEvent",
    "ArchivedMemory",
    "Embedding",
    "MemoryChain",
    "MemoryChainItem",
    "AnchorPointer",
    "MemoryRelationship",
    "RelationshipResidue",
    "MemorySupersede",
    "UserActiveStore",
    "TenantStorageUsage",
    "MEMORY_MODELS",
    "MEMORY_RELATIONSHIP_MODELS",
]
