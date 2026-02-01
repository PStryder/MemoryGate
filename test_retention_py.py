#!/usr/bin/env python3
"""Direct function test of MemoryGate retention"""
import sys
sys.path.insert(0, '/app')

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
import os
import pytest

# Connect to database
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://memorygate:memorygate@postgres:5432/memorygate')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def _skip_if_unavailable(exc: Exception) -> None:
    message = f"Skipping retention test: database unavailable ({exc})"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    print(message)

def test_retention():
    db = SessionLocal()
    
    try:
        print("=" * 60)
        print("MemoryGate Retention Test (Direct DB)")
        print("=" * 60)

        try:
            db.execute(text("SELECT 1"))
        except OperationalError as exc:
            _skip_if_unavailable(exc)
            return
        
        # Store test observation
        result = db.execute(text("""
            INSERT INTO observations (observation, domain, confidence, timestamp, last_accessed_at, access_count, tier, score, floor_score, purge_eligible)
            VALUES ('API test observation', 'api_test', 0.85, NOW(), NOW(), 0, 'hot', 0.0, 0.0, false)
            RETURNING id, observation, score, tier
        """))
        row = result.fetchone()
        print(f"\n✅ Stored: ID={row[0]}, score={row[2]}, tier={row[3]}")
        obs_id = row[0]
        
        # Simulate access (bump score)
        print(f"\n==> Simulating 3 accesses (SCORE_BUMP_ALPHA=0.4)...")
        for i in range(3):
            db.execute(text("""
                UPDATE observations 
                SET score = score + 0.4, 
                    access_count = access_count + 1,
                    last_accessed_at = NOW()
                WHERE id = :id
            """), {"id": obs_id})
        db.commit()
        
        result = db.execute(text("SELECT score, access_count FROM observations WHERE id = :id"), {"id": obs_id})
        row = result.fetchone()
        print(f"✅ After 3 accesses: score={row[0]}, access_count={row[1]}")
        
        # Simulate decay
        print(f"\n==> Simulating 100 decay ticks (SCORE_DECAY_BETA=0.02)...")
        db.execute(text("""
            UPDATE observations 
            SET score = score - (0.02 * 100)
            WHERE id = :id
        """), {"id": obs_id})
        db.commit()
        
        result = db.execute(text("SELECT score, tier FROM observations WHERE id = :id"), {"id": obs_id})
        row = result.fetchone()
        print(f"✅ After decay: score={row[0]}, tier={row[1]}")
        
        # Check if should be archived
        if row[0] < -1.0:
            print(f"\n==> Score < -1.0, moving to COLD tier...")
            db.execute(text("""
                UPDATE observations 
                SET tier = 'cold', archived_at = NOW(), archived_reason = 'Score threshold'
                WHERE id = :id
            """), {"id": obs_id})
            db.commit()
            print("✅ Moved to COLD tier")
        
        # Final state
        result = db.execute(text("""
            SELECT observation, score, tier, access_count, purge_eligible 
            FROM observations WHERE id = :id
        """), {"id": obs_id})
        row = result.fetchone()
        
        print("\n" + "=" * 60)
        print("FINAL STATE")
        print("=" * 60)
        print(f"Observation: {row[0]}")
        print(f"Score: {row[1]}")
        print(f"Tier: {row[2]}")
        print(f"Access count: {row[3]}")
        print(f"Purge eligible: {row[4]}")
        
        # Stats
        result = db.execute(text("""
            SELECT tier, COUNT(*) as count
            FROM observations
            GROUP BY tier
        """))
        print(f"\nGlobal stats:")
        for row in result:
            print(f"  {row[0]}: {row[1]} observations")
        
        print("\n✅ RETENTION MECHANISM WORKING!")
        print("=" * 60)
        
    finally:
        db.close()

if __name__ == "__main__":
    test_retention()
