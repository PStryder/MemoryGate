"""Initial schema.

Revision ID: 0001_initial
Revises: 
Create Date: 2026-01-09
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    vector_backend = os.environ.get("VECTOR_BACKEND", "pgvector").strip().lower()
    json_type = postgresql.JSONB if is_postgres else sa.JSON
    uuid_type = postgresql.UUID(as_uuid=True) if is_postgres else sa.String(length=36)
    embedding_type = sa.JSON
    if is_postgres and vector_backend == "pgvector":
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError as exc:
            raise RuntimeError("pgvector is required when VECTOR_BACKEND=pgvector") from exc
        embedding_type = Vector(1536)
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "ai_instances",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("platform", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.String(length=255), nullable=True, unique=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("ai_instance_id", sa.Integer(), sa.ForeignKey("ai_instances.id")),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("metadata", json_type, nullable=True),
    )

    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("domain", sa.String(length=100), nullable=True),
        sa.Column("evidence", json_type, nullable=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id")),
        sa.Column("ai_instance_id", sa.Integer(), sa.ForeignKey("ai_instances.id")),
        sa.Column("access_count", sa.Integer(), nullable=True),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="check_confidence"
        ),
    )
    op.create_index("ix_observations_domain", "observations", ["domain"])
    op.create_index("ix_observations_confidence", "observations", ["confidence"])

    op.create_table(
        "patterns",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("pattern_name", sa.String(length=255), nullable=False),
        sa.Column("pattern_text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evidence_observation_ids", json_type, nullable=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id")),
        sa.Column("ai_instance_id", sa.Integer(), sa.ForeignKey("ai_instances.id")),
        sa.Column("access_count", sa.Integer(), nullable=True),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("category", "pattern_name", name="uq_pattern_category_name"),
    )
    op.create_index("ix_patterns_category", "patterns", ["category"])

    op.create_table(
        "concepts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("name_key", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("domain", sa.String(length=100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata", json_type, nullable=True),
        sa.Column("ai_instance_id", sa.Integer(), sa.ForeignKey("ai_instances.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=True),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_concepts_name_key", "concepts", ["name_key"])
    op.create_index("ix_concepts_type", "concepts", ["type"])

    op.create_table(
        "concept_aliases",
        sa.Column("alias", sa.String(length=255), primary_key=True),
        sa.Column("alias_key", sa.String(length=255), nullable=False),
        sa.Column("concept_id", sa.Integer(), sa.ForeignKey("concepts.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_concept_aliases_alias_key", "concept_aliases", ["alias_key"])

    op.create_table(
        "concept_relationships",
        sa.Column(
            "from_concept_id",
            sa.Integer(),
            sa.ForeignKey("concepts.id"),
            primary_key=True,
        ),
        sa.Column(
            "to_concept_id",
            sa.Integer(),
            sa.ForeignKey("concepts.id"),
            primary_key=True,
        ),
        sa.Column("rel_type", sa.String(length=50), primary_key=True),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("weight >= 0 AND weight <= 1", name="check_rel_weight"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("doc_type", sa.String(length=50), nullable=False),
        sa.Column("content_summary", sa.Text(), nullable=True),
        sa.Column("url", sa.String(length=1000), nullable=True),
        sa.Column("publication_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("key_concepts", json_type, nullable=True),
        sa.Column("metadata", json_type, nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=True),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "embeddings",
        sa.Column("source_type", sa.String(length=50), primary_key=True),
        sa.Column("source_id", sa.Integer(), primary_key=True),
        sa.Column(
            "model_version",
            sa.String(length=100),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.Column("normalized", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_embeddings_source", "embeddings", ["source_type", "source_id"])
    if is_postgres and vector_backend == "pgvector":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw "
            "ON embeddings USING hnsw (embedding vector_cosine_ops)"
        )

    op.create_table(
        "users",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("oauth_provider", sa.String(), nullable=False),
        sa.Column("oauth_subject", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_login", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index(
        "idx_oauth_provider_subject",
        "users",
        ["oauth_provider", "oauth_subject"],
        unique=True,
    )

    op.create_table(
        "oauth_states",
        sa.Column("state", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("redirect_uri", sa.String(), nullable=True),
        sa.Column("code_verifier", sa.String(), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "oauth_authorization_codes",
        sa.Column("code", sa.String(), primary_key=True),
        sa.Column("client_id", sa.String(), nullable=False),
        sa.Column("redirect_uri", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("code_challenge", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "user_sessions",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "user_id",
            uuid_type,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token", sa.String(), nullable=False, unique=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("last_activity", sa.DateTime(), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_token", "user_sessions", ["token"])

    op.create_table(
        "api_keys",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "user_id",
            uuid_type,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key_prefix", sa.String(), nullable=False),
        sa.Column("key_hash", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("scopes", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_used", sa.DateTime(), nullable=True),
        sa.Column("usage_count", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_revoked", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    vector_backend = os.environ.get("VECTOR_BACKEND", "pgvector").strip().lower()
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_user_sessions_token", table_name="user_sessions")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_table("user_sessions")
    op.drop_table("oauth_authorization_codes")
    op.drop_table("oauth_states")
    op.drop_index("idx_oauth_provider_subject", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    if is_postgres and vector_backend == "pgvector":
        op.execute("DROP INDEX IF EXISTS ix_embeddings_vector_hnsw")
    op.drop_index("ix_embeddings_source", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_table("documents")
    op.drop_table("concept_relationships")
    op.drop_index("ix_concept_aliases_alias_key", table_name="concept_aliases")
    op.drop_table("concept_aliases")
    op.drop_index("ix_concepts_type", table_name="concepts")
    op.drop_index("ix_concepts_name_key", table_name="concepts")
    op.drop_table("concepts")
    op.drop_index("ix_patterns_category", table_name="patterns")
    op.drop_table("patterns")
    op.drop_index("ix_observations_confidence", table_name="observations")
    op.drop_index("ix_observations_domain", table_name="observations")
    op.drop_table("observations")
    op.drop_table("sessions")
    op.drop_table("ai_instances")
