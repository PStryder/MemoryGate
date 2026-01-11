import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")
os.environ.setdefault("REQUIRE_MCP_AUTH", "false")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.db import DB
from core.models import Base
from core.services import memory as memory_service


def _store_observation(text: str) -> dict:
    return memory_service.memory_store(
        observation=text,
        confidence=0.9,
        domain="concurrency",
    )


def test_memory_store_concurrency(tmp_path):
    db_path = tmp_path / "concurrency.sqlite"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    previous_engine = DB.engine
    previous_session = DB.SessionLocal
    DB.engine = engine
    DB.SessionLocal = SessionLocal
    try:
        texts = ["Concurrent observation 1", "Concurrent observation 2"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(_store_observation, texts))

        assert all(result["status"] == "stored" for result in results)

        recall = memory_service.memory_recall(domain="concurrency", include_cold=True, limit=10)
        assert recall["count"] >= 2
    finally:
        DB.engine = previous_engine
        DB.SessionLocal = previous_session
        engine.dispose()
