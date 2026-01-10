"""
Pytest fixtures for MemoryGate tests.

Ensures OAuth tables are created before tests run.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment before any imports
os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import Base first
from models import Base

# Import ALL model modules to register tables with Base.metadata
import models  # noqa: F401
import oauth_models  # noqa: F401

# Import specific classes to ensure they're registered
from oauth_models import User, OAuthState, UserSession, APIKey, OAuthAuthorizationCode  # noqa: F401

import pytest


@pytest.fixture
def db_engine():
    """
    Create a SHARED in-memory database that persists across connections.
    
    Critical: Use StaticPool and URI mode for in-memory SQLite to ensure
    all connections see the same database instance.
    """
    engine = create_engine(
        "sqlite:///:memory:",  # In-memory database
        connect_args={
            "check_same_thread": False,  # Allow multi-threading
        },
        poolclass=StaticPool,  # CRITICAL: Share single connection across all users
    )
    
    # Create ALL tables
    Base.metadata.create_all(engine)
    
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
