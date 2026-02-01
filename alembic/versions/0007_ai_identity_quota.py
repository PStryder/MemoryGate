"""Add AI identity fields, sharing policies, and storage quota tracking.

Revision ID: 0007_ai_identity_quota
Revises: 0006_add_chains_relationships
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0007_ai_identity_quota"
down_revision = "0006_add_chains_relationships"
branch_labels = None
depends_on = None


TENANT_ID_DEFAULT = "system"


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        uuid_type = postgresql.UUID(as_uuid=True)
        json_type = postgresql.JSONB()
    else:
        uuid_type = sa.String(36)
        json_type = sa.JSON()

    # -------------------------------------------------------------------------
    # AI identity columns + constraints
    # -------------------------------------------------------------------------
    with op.batch_alter_table("ai_instances", recreate="always" if is_sqlite else "auto") as batch:
        batch.add_column(sa.Column("canonical_name", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("canonical_platform", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("agent_uuid", sa.String(length=40), nullable=True))
        batch.add_column(sa.Column("legacy_agent_uuid", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("user_id", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("tags", json_type, nullable=True))
        batch.add_column(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("last_seen_raw_name", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("last_seen_raw_platform", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("is_archived", sa.Boolean(), nullable=False, server_default=sa.text("false"))
        )
        batch.add_column(sa.Column("merged_into_ai_instance_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_ai_instances_merged_into",
            "ai_instances",
            ["merged_into_ai_instance_id"],
            ["id"],
        )

        # Drop legacy unique constraint on name (if present) and add tenant-aware uniques
        if not is_sqlite:
            batch.drop_constraint("ai_instances_name_key", type_="unique")
        batch.create_unique_constraint(
            "uq_ai_instances_tenant_name_platform",
            ["tenant_id", "name", "platform"],
        )
        batch.create_unique_constraint(
            "uq_ai_instances_tenant_agent_uuid",
            ["tenant_id", "agent_uuid"],
        )

    # -------------------------------------------------------------------------
    # Session uniqueness scoped by tenant
    # -------------------------------------------------------------------------
    with op.batch_alter_table("sessions", recreate="always" if is_sqlite else "auto") as batch:
        if not is_sqlite:
            batch.drop_constraint("sessions_conversation_id_key", type_="unique")
        batch.create_unique_constraint(
            "uq_sessions_tenant_conversation_id",
            ["tenant_id", "conversation_id"],
        )

    # -------------------------------------------------------------------------
    # created_by_user_pk provenance columns
    # -------------------------------------------------------------------------
    op.add_column("observations", sa.Column("created_by_user_pk", uuid_type, nullable=True))
    op.add_column("patterns", sa.Column("created_by_user_pk", uuid_type, nullable=True))
    op.add_column("concepts", sa.Column("created_by_user_pk", uuid_type, nullable=True))
    op.add_column("documents", sa.Column("created_by_user_pk", uuid_type, nullable=True))
    op.add_column("memory_summaries", sa.Column("created_by_user_pk", uuid_type, nullable=True))

    # -------------------------------------------------------------------------
    # AI memory policy + entity shares
    # -------------------------------------------------------------------------
    op.create_table(
        "ai_memory_policy",
        sa.Column(
            "tenant_id",
            sa.String(length=100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("ai_instance_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "ai_instance_id"),
        sa.ForeignKeyConstraint(["ai_instance_id"], ["ai_instances.id"]),
    )

    op.create_table(
        "ai_entity_shares",
        sa.Column(
            "tenant_id",
            sa.String(length=100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("shared_with_ai_instance_id", sa.Integer(), nullable=False),
        sa.Column("shared_by_user_id", uuid_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint(
            "tenant_id",
            "entity_type",
            "entity_id",
            "shared_with_ai_instance_id",
        ),
        sa.ForeignKeyConstraint(
            ["shared_with_ai_instance_id"],
            ["ai_instances.id"],
        ),
    )
    op.create_index(
        "ix_ai_entity_shares_tenant_ai",
        "ai_entity_shares",
        ["tenant_id", "shared_with_ai_instance_id"],
    )

    # -------------------------------------------------------------------------
    # Tenant storage usage tracking
    # -------------------------------------------------------------------------
    op.create_table(
        "tenant_storage_usage",
        sa.Column(
            "tenant_id",
            sa.String(length=100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
        sa.Column("storage_used_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("tenant_id"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.drop_table("tenant_storage_usage")
    op.drop_index("ix_ai_entity_shares_tenant_ai", table_name="ai_entity_shares")
    op.drop_table("ai_entity_shares")
    op.drop_table("ai_memory_policy")

    op.drop_column("memory_summaries", "created_by_user_pk")
    op.drop_column("documents", "created_by_user_pk")
    op.drop_column("concepts", "created_by_user_pk")
    op.drop_column("patterns", "created_by_user_pk")
    op.drop_column("observations", "created_by_user_pk")

    with op.batch_alter_table("sessions", recreate="always" if is_sqlite else "auto") as batch:
        batch.drop_constraint("uq_sessions_tenant_conversation_id", type_="unique")
        batch.create_unique_constraint(
            "sessions_conversation_id_key",
            ["conversation_id"],
        )

    with op.batch_alter_table("ai_instances", recreate="always" if is_sqlite else "auto") as batch:
        batch.drop_constraint("uq_ai_instances_tenant_agent_uuid", type_="unique")
        batch.drop_constraint("uq_ai_instances_tenant_name_platform", type_="unique")
        if not is_sqlite:
            batch.create_unique_constraint("ai_instances_name_key", ["name"])
        batch.drop_constraint("fk_ai_instances_merged_into", type_="foreignkey")
        batch.drop_column("merged_into_ai_instance_id")
        batch.drop_column("is_archived")
        batch.drop_column("last_seen_at")
        batch.drop_column("last_seen_raw_platform")
        batch.drop_column("last_seen_raw_name")
        batch.drop_column("updated_at")
        batch.drop_column("tags")
        batch.drop_column("user_id")
        batch.drop_column("legacy_agent_uuid")
        batch.drop_column("agent_uuid")
        batch.drop_column("canonical_platform")
        batch.drop_column("canonical_name")
