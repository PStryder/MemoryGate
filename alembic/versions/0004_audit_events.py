"""Add audit events table.

Revision ID: 0004_audit_events
Revises: 0003_archived_memories
Create Date: 2026-01-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004_audit_events"
down_revision = "0003_archived_memories"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB if is_postgres else sa.JSON
    uuid_type = postgresql.UUID(as_uuid=True) if is_postgres else sa.String(length=36)

    op.create_table(
        "audit_events",
        sa.Column("event_id", uuid_type, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("event_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("actor_type", sa.String(length=50), nullable=False),
        sa.Column("actor_id", sa.String(length=255)),
        sa.Column("org_id", sa.String(length=255)),
        sa.Column("user_id", sa.String(length=255)),
        sa.Column("target_type", sa.String(length=50), nullable=False),
        sa.Column("target_ids", json_type, nullable=False),
        sa.Column("count_affected", sa.Integer()),
        sa.Column("reason", sa.Text()),
        sa.Column("request_id", sa.String(length=255)),
        sa.Column("metadata", json_type),
    )
    op.create_index(
        "ix_audit_events_created_at",
        "audit_events",
        ["created_at"],
    )
    op.create_index(
        "ix_audit_events_event_type",
        "audit_events",
        ["event_type"],
    )
    op.create_index(
        "ix_audit_events_org_id",
        "audit_events",
        ["org_id"],
    )
    op.create_index(
        "ix_audit_events_user_id",
        "audit_events",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_user_id", table_name="audit_events")
    op.drop_index("ix_audit_events_org_id", table_name="audit_events")
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_table("audit_events")
