"""Add chains, relationships, residue, supersession, anchors, and stores.

Revision ID: 0006_add_chains_relationships
Revises: 0005_add_tenant_id
Create Date: 2026-01-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0006_add_chains_relationships"
down_revision = "0005_add_tenant_id"
branch_labels = None
depends_on = None


TENANT_ID_DEFAULT = "system"


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    is_postgres = bind.dialect.name == "postgresql"

    # Use appropriate UUID type
    if is_postgres:
        uuid_type = postgresql.UUID(as_uuid=True)
        json_type = postgresql.JSONB()
    else:
        uuid_type = sa.String(36)
        json_type = sa.JSON()

    # =============================================================================
    # Memory Chains
    # =============================================================================
    op.create_table(
        "memory_chains",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("created_by_user_id", uuid_type, nullable=True),
        sa.Column("created_by_type", sa.String(50), nullable=True),
        sa.Column("created_by_id", sa.String(255), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(100), nullable=True),
        sa.Column("meta", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_chains_tenant_id", "memory_chains", ["tenant_id"])
    op.create_index("ix_memory_chains_kind", "memory_chains", ["kind"])

    # =============================================================================
    # Memory Chain Items
    # =============================================================================
    op.create_table(
        "memory_chain_items",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("chain_id", uuid_type, nullable=False),
        sa.Column("memory_id", sa.String(255), nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(100), nullable=True),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["chain_id"],
            ["memory_chains.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["observation_id"],
            ["observations.id"],
        ),
    )
    op.create_unique_constraint(
        "uq_memory_chain_items_seq",
        "memory_chain_items",
        ["tenant_id", "chain_id", "seq"],
    )
    op.create_unique_constraint(
        "uq_memory_chain_items_memory",
        "memory_chain_items",
        ["tenant_id", "chain_id", "memory_id"],
    )
    op.create_index(
        "ix_memory_chain_items_chain_seq",
        "memory_chain_items",
        ["tenant_id", "chain_id", "seq"],
    )
    op.create_index(
        "ix_memory_chain_items_memory",
        "memory_chain_items",
        ["tenant_id", "memory_id"],
    )

    # =============================================================================
    # Anchor Pointers
    # =============================================================================
    op.create_table(
        "anchor_pointers",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("scope_type", sa.String(50), nullable=False),
        sa.Column("scope_key", sa.String(255), nullable=False),
        sa.Column("anchor_kind", sa.String(50), nullable=False),
        sa.Column("chain_id", uuid_type, nullable=False),
        sa.Column("chain_head_seq", sa.Integer(), nullable=True),
        sa.Column("meta", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint(
        "uq_anchor_pointers_scope",
        "anchor_pointers",
        ["tenant_id", "scope_type", "scope_key", "anchor_kind"],
    )
    op.create_index(
        "ix_anchor_pointers_tenant_scope",
        "anchor_pointers",
        ["tenant_id", "scope_type", "scope_key"],
    )

    # =============================================================================
    # Memory Relationships
    # =============================================================================
    op.create_table(
        "memory_relationships",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("from_type", sa.Text(), nullable=False),
        sa.Column("from_id", sa.Text(), nullable=False),
        sa.Column("to_type", sa.Text(), nullable=False),
        sa.Column("to_id", sa.Text(), nullable=False),
        sa.Column("rel_type", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_type", sa.Text(), nullable=True),
        sa.Column("created_by_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "from_type != '' AND to_type != ''",
            name="ck_memory_relationships_types",
        ),
        sa.CheckConstraint(
            "from_id != to_id OR from_type != to_type",
            name="ck_memory_relationships_distinct",
        ),
    )
    op.create_unique_constraint(
        "uq_memory_relationships_unique",
        "memory_relationships",
        ["tenant_id", "from_type", "from_id", "to_type", "to_id", "rel_type"],
    )
    op.create_index(
        "ix_memory_relationships_from",
        "memory_relationships",
        ["tenant_id", "from_type", "from_id"],
    )
    op.create_index(
        "ix_memory_relationships_to",
        "memory_relationships",
        ["tenant_id", "to_type", "to_id"],
    )
    op.create_index(
        "ix_memory_relationships_type",
        "memory_relationships",
        ["tenant_id", "rel_type"],
    )

    # =============================================================================
    # Relationship Residue
    # =============================================================================
    op.create_table(
        "relationship_residue",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("edge_id", uuid_type, nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("encoded_rel_type", sa.Text(), nullable=False),
        sa.Column("encoded_weight", sa.Float(), nullable=True),
        sa.Column("alternatives_considered", json_type, nullable=True),
        sa.Column("alternatives_ruled_out", json_type, nullable=True),
        sa.Column("friction_metrics", json_type, nullable=True),
        sa.Column("compression_texture", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["edge_id"],
            ["memory_relationships.id"],
        ),
    )
    op.create_index(
        "ix_relationship_residue_edge_created_at",
        "relationship_residue",
        ["edge_id", "created_at"],
    )
    op.create_index(
        "ix_relationship_residue_event_type",
        "relationship_residue",
        ["event_type"],
    )
    op.create_index(
        "ix_relationship_residue_actor",
        "relationship_residue",
        ["actor"],
    )

    # =============================================================================
    # Memory Supersedes
    # =============================================================================
    op.create_table(
        "memory_supersedes",
        sa.Column("id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("from_memory_id", sa.String(255), nullable=False),
        sa.Column("to_memory_id", sa.String(255), nullable=False),
        sa.Column("relation_type", sa.String(50), nullable=False, server_default="supersedes"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "from_memory_id != to_memory_id",
            name="ck_memory_supersedes_distinct",
        ),
    )
    op.create_unique_constraint(
        "uq_memory_supersedes_from_relation",
        "memory_supersedes",
        ["tenant_id", "from_memory_id", "relation_type"],
    )
    op.create_index(
        "ix_memory_supersedes_from",
        "memory_supersedes",
        ["tenant_id", "from_memory_id"],
    )
    op.create_index(
        "ix_memory_supersedes_to",
        "memory_supersedes",
        ["tenant_id", "to_memory_id"],
    )

    # =============================================================================
    # User Active Stores
    # =============================================================================
    op.create_table(
        "user_active_stores",
        sa.Column("user_id", uuid_type, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("active_store_id", sa.String(255), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_active_stores")
    op.drop_table("memory_supersedes")
    op.drop_table("relationship_residue")
    op.drop_table("memory_relationships")
    op.drop_table("anchor_pointers")
    op.drop_table("memory_chain_items")
    op.drop_table("memory_chains")
