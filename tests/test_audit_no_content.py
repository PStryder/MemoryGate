import os

import pytest

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")

from core.audit import log_event
from core.audit_constants import EVENT_MEMORY_ARCHIVED
from core.models import AuditEvent


def test_audit_rejects_content_metadata(db_session):
    before = db_session.query(AuditEvent).count()
    with pytest.raises(ValueError):
        log_event(
            db_session,
            event_type=EVENT_MEMORY_ARCHIVED,
            actor_type="system",
            target_type="memory",
            target_ids=[1],
            metadata={"content": "should_not_log"},
        )
    db_session.rollback()
    after = db_session.query(AuditEvent).count()
    assert after == before


def test_audit_rejects_long_strings(db_session):
    before = db_session.query(AuditEvent).count()
    with pytest.raises(ValueError):
        log_event(
            db_session,
            event_type=EVENT_MEMORY_ARCHIVED,
            actor_type="system",
            target_type="memory",
            target_ids=[1],
            metadata={"note": "x" * 600},
        )
    db_session.rollback()
    after = db_session.query(AuditEvent).count()
    assert after == before
