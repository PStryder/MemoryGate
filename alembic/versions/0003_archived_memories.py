"""Add archived memories table.

Revision ID: 0003_archived_memories
Revises: 8eaa0966fb0f_add_oauth_tables
Create Date: 2026-01-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_archived_memories"
down_revision = "8eaa0966fb0f_add_oauth_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB if is_postgres else sa.JSON

    op.create_table(
        "archived_memories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("purge_reason", sa.Text()),
        sa.Column("purge_actor", sa.String(length=100)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "size_bytes_estimate",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("original_embedding", json_type),
        sa.Column("metadata", json_type),
        sa.UniqueConstraint(
            "source_type",
            "source_id",
            name="uq_archived_memories_source",
        ),
    )
    op.create_index(
        "ix_archived_memories_source_type",
        "archived_memories",
        ["source_type"],
    )
    op.create_index(
        "ix_archived_memories_archived_at",
        "archived_memories",
        ["archived_at"],
    )
    op.create_index(
        "ix_archived_memories_expires_at",
        "archived_memories",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_archived_memories_expires_at", table_name="archived_memories")
    op.drop_index("ix_archived_memories_archived_at", table_name="archived_memories")
    op.drop_index("ix_archived_memories_source_type", table_name="archived_memories")
    op.drop_table("archived_memories")
