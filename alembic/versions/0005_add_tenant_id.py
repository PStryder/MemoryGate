"""Add tenant_id to core tables.

Revision ID: 0005_add_tenant_id
Revises: 0004_audit_events
Create Date: 2026-01-11
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_add_tenant_id"
down_revision = "0004_audit_events"
branch_labels = None
depends_on = None


TENANT_ID_DEFAULT = "system"


def _add_tenant_id(table_name: str) -> None:
    op.add_column(
        table_name,
        sa.Column(
            "tenant_id",
            sa.String(length=100),
            nullable=False,
            server_default=TENANT_ID_DEFAULT,
        ),
    )


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    _add_tenant_id("ai_instances")
    _add_tenant_id("sessions")
    _add_tenant_id("observations")
    _add_tenant_id("documents")
    _add_tenant_id("memory_summaries")
    _add_tenant_id("memory_tombstones")
    _add_tenant_id("archived_memories")
    _add_tenant_id("embeddings")
    _add_tenant_id("audit_events")
    _add_tenant_id("concept_aliases")
    _add_tenant_id("concept_relationships")

    with op.batch_alter_table("patterns", recreate="always" if is_sqlite else "auto") as batch:
        batch.add_column(
            sa.Column(
                "tenant_id",
                sa.String(length=100),
                nullable=False,
                server_default=TENANT_ID_DEFAULT,
            )
        )
        batch.drop_constraint("uq_pattern_category_name", type_="unique")
        batch.create_unique_constraint(
            "uq_patterns_tenant_category_name",
            ["tenant_id", "category", "pattern_name"],
        )

    with op.batch_alter_table("concepts", recreate="always" if is_sqlite else "auto") as batch:
        batch.add_column(
            sa.Column(
                "tenant_id",
                sa.String(length=100),
                nullable=False,
                server_default=TENANT_ID_DEFAULT,
            )
        )
        batch.create_unique_constraint(
            "uq_concepts_tenant_name_key",
            ["tenant_id", "name_key"],
        )

    if not is_sqlite:
        op.execute("UPDATE patterns SET tenant_id = 'system' WHERE tenant_id IS NULL")
        op.execute("UPDATE concepts SET tenant_id = 'system' WHERE tenant_id IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    with op.batch_alter_table("concepts", recreate="always" if is_sqlite else "auto") as batch:
        batch.drop_constraint("uq_concepts_tenant_name_key", type_="unique")
        batch.drop_column("tenant_id")

    with op.batch_alter_table("patterns", recreate="always" if is_sqlite else "auto") as batch:
        batch.drop_constraint("uq_patterns_tenant_category_name", type_="unique")
        batch.create_unique_constraint(
            "uq_pattern_category_name",
            ["category", "pattern_name"],
        )
        batch.drop_column("tenant_id")

    op.drop_column("concept_relationships", "tenant_id")
    op.drop_column("concept_aliases", "tenant_id")
    op.drop_column("audit_events", "tenant_id")
    op.drop_column("embeddings", "tenant_id")
    op.drop_column("archived_memories", "tenant_id")
    op.drop_column("memory_tombstones", "tenant_id")
    op.drop_column("memory_summaries", "tenant_id")
    op.drop_column("documents", "tenant_id")
    op.drop_column("observations", "tenant_id")
    op.drop_column("sessions", "tenant_id")
    op.drop_column("ai_instances", "tenant_id")
