"""Cold storage tiering, summaries, and tombstones.

Revision ID: 0002_cold_storage
Revises: 0001_initial
Create Date: 2026-01-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_cold_storage"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB if is_postgres else sa.JSON
    uuid_type = postgresql.UUID(as_uuid=True) if is_postgres else sa.String(length=36)
    default_false = sa.text("false") if is_postgres else sa.text("0")
    if is_postgres:
        memory_tier_enum = postgresql.ENUM("hot", "cold", name="memory_tier")
        tombstone_action_enum = postgresql.ENUM(
            "archived", "rehydrated", "purged", "summarized", name="tombstone_action"
        )
        memory_tier_enum.create(bind, checkfirst=True)
        tombstone_action_enum.create(bind, checkfirst=True)
    else:
        memory_tier_enum = sa.Enum("hot", "cold", name="memory_tier", native_enum=False)
        tombstone_action_enum = sa.Enum(
            "archived", "rehydrated", "purged", "summarized",
            name="tombstone_action",
            native_enum=False,
        )

    def _upgrade_memory_table(table_name: str) -> None:
        op.execute(f"UPDATE {table_name} SET access_count = 0 WHERE access_count IS NULL")
        if is_postgres:
            op.alter_column(
                table_name,
                "last_accessed",
                new_column_name="last_accessed_at",
            )
            op.alter_column(
                table_name,
                "access_count",
                existing_type=sa.Integer(),
                type_=sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
            op.add_column(
                table_name,
                sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
            )
            op.add_column(table_name, sa.Column("archived_at", sa.DateTime(timezone=True)))
            op.add_column(table_name, sa.Column("archived_reason", sa.Text()))
            op.add_column(table_name, sa.Column("archived_by", sa.String(length=100)))
            op.add_column(
                table_name,
                sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
            )
            op.add_column(
                table_name,
                sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
            )
            op.add_column(
                table_name,
                sa.Column(
                    "purge_eligible",
                    sa.Boolean(),
                    nullable=False,
                    server_default=default_false,
                ),
            )
            op.create_index(f"ix_{table_name}_tier", table_name, ["tier"])
            op.create_index(f"ix_{table_name}_score", table_name, ["score"])
        else:
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.alter_column(
                    "last_accessed",
                    new_column_name="last_accessed_at",
                )
                batch_op.alter_column(
                    "access_count",
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    nullable=False,
                    server_default="0",
                )
                batch_op.add_column(
                    sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
                )
                batch_op.add_column(sa.Column("archived_at", sa.DateTime(timezone=True)))
                batch_op.add_column(sa.Column("archived_reason", sa.Text()))
                batch_op.add_column(sa.Column("archived_by", sa.String(length=100)))
                batch_op.add_column(
                    sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
                )
                batch_op.add_column(
                    sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
                )
                batch_op.add_column(
                    sa.Column(
                        "purge_eligible",
                        sa.Boolean(),
                        nullable=False,
                        server_default=default_false,
                    ),
                )
                batch_op.create_index(f"ix_{table_name}_tier", ["tier"])
                batch_op.create_index(f"ix_{table_name}_score", ["score"])

    _upgrade_memory_table("observations")
    _upgrade_memory_table("patterns")
    _upgrade_memory_table("concepts")
    _upgrade_memory_table("documents")

    op.create_table(
        "memory_summaries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("source_ids", json_type, nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("metadata", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "access_count",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True)),
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        sa.Column("archived_reason", sa.Text()),
        sa.Column("archived_by", sa.String(length=100)),
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=default_false,
        ),
    )
    op.create_index(
        "ix_memory_summaries_source",
        "memory_summaries",
        ["source_type", "source_id"],
    )
    op.create_index("ix_memory_summaries_tier", "memory_summaries", ["tier"])
    op.create_index("ix_memory_summaries_score", "memory_summaries", ["score"])

    op.create_table(
        "memory_tombstones",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("memory_id", sa.String(length=255), nullable=False),
        sa.Column("action", tombstone_action_enum, nullable=False),
        sa.Column("from_tier", memory_tier_enum, nullable=True),
        sa.Column("to_tier", memory_tier_enum, nullable=True),
        sa.Column("reason", sa.Text()),
        sa.Column("actor", sa.String(length=100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", json_type, nullable=True),
    )
    op.create_index(
        "ix_memory_tombstones_memory_id",
        "memory_tombstones",
        ["memory_id"],
    )
    op.create_index(
        "ix_memory_tombstones_action",
        "memory_tombstones",
        ["action"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_tombstones_action", table_name="memory_tombstones")
    op.drop_index("ix_memory_tombstones_memory_id", table_name="memory_tombstones")
    op.drop_table("memory_tombstones")

    op.drop_index("ix_memory_summaries_score", table_name="memory_summaries")
    op.drop_index("ix_memory_summaries_tier", table_name="memory_summaries")
    op.drop_index("ix_memory_summaries_source", table_name="memory_summaries")
    op.drop_table("memory_summaries")

    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    def _downgrade_memory_table(table_name: str) -> None:
        if is_postgres:
            op.drop_index(f"ix_{table_name}_score", table_name=table_name)
            op.drop_index(f"ix_{table_name}_tier", table_name=table_name)
            op.drop_column(table_name, "purge_eligible")
            op.drop_column(table_name, "floor_score")
            op.drop_column(table_name, "score")
            op.drop_column(table_name, "archived_by")
            op.drop_column(table_name, "archived_reason")
            op.drop_column(table_name, "archived_at")
            op.drop_column(table_name, "tier")
            op.alter_column(
                table_name,
                "access_count",
                existing_type=sa.BigInteger(),
                type_=sa.Integer(),
                nullable=True,
                server_default=None,
            )
            op.alter_column(
                table_name,
                "last_accessed_at",
                new_column_name="last_accessed",
            )
        else:
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.drop_index(f"ix_{table_name}_score")
                batch_op.drop_index(f"ix_{table_name}_tier")
                batch_op.drop_column("purge_eligible")
                batch_op.drop_column("floor_score")
                batch_op.drop_column("score")
                batch_op.drop_column("archived_by")
                batch_op.drop_column("archived_reason")
                batch_op.drop_column("archived_at")
                batch_op.drop_column("tier")
                batch_op.alter_column(
                    "access_count",
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    nullable=True,
                    server_default=None,
                )
                batch_op.alter_column(
                    "last_accessed_at",
                    new_column_name="last_accessed",
                )

    _downgrade_memory_table("documents")
    _downgrade_memory_table("concepts")
    _downgrade_memory_table("patterns")
    _downgrade_memory_table("observations")

    if is_postgres:
        tombstone_action_enum = postgresql.ENUM(
            "archived", "rehydrated", "purged", "summarized", name="tombstone_action"
        )
        memory_tier_enum = postgresql.ENUM("hot", "cold", name="memory_tier")
        tombstone_action_enum.drop(bind, checkfirst=True)
        memory_tier_enum.drop(bind, checkfirst=True)
