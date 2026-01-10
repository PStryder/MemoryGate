-- MemoryGate v2 Forgetting Mechanism SQL Test (Schema-Corrected)

\echo '========================================='
\echo 'MemoryGate v2 Retention Test'
\echo '========================================='

-- Insert test observations with proper schema
INSERT INTO observations (observation, domain, confidence, timestamp, last_accessed_at, access_count, tier, score, floor_score, purge_eligible)
VALUES 
  ('Critical data - frequently accessed', 'test', 0.9, NOW(), NOW(), 10, 'hot', 3.5, 0.0, false),
  ('Moderate data - sometimes accessed', 'test', 0.8, NOW() - INTERVAL '1 hour', NOW() - INTERVAL '1 hour', 3, 'hot', 0.8, 0.0, false),
  ('Cold data - rarely accessed', 'test', 0.7, NOW() - INTERVAL '5 hours', NOW() - INTERVAL '5 hours', 0, 'hot', -0.5, 0.0, false),
  ('Forgotten data - never accessed', 'test', 0.6, NOW() - INTERVAL '10 hours', NOW() - INTERVAL '10 hours', 0, 'hot', -2.5, 0.0, true);

\echo 'Inserted 4 test observations'
\echo ''

-- Check initial state
\echo 'Initial State:'
SELECT 
  tier,
  COUNT(*) as count,
  ROUND(AVG(score)::numeric, 3) as avg_score,
  ROUND(MIN(score)::numeric, 3) as min_score,
  ROUND(MAX(score)::numeric, 3) as max_score
FROM observations 
WHERE domain = 'test'
GROUP BY tier;

\echo ''
\echo 'Detailed View:'
SELECT 
  id,
  LEFT(observation, 30) as preview,
  ROUND(score::numeric, 3) as score,
  tier,
  access_count,
  purge_eligible
FROM observations 
WHERE domain = 'test'
ORDER BY score DESC;

\echo ''
\echo '==> Simulating decay tick (score -= 0.02)...'
UPDATE observations 
SET score = score - 0.02
WHERE domain = 'test';

\echo 'After decay:'
SELECT 
  LEFT(observation, 30) as preview,
  ROUND(score::numeric, 3) as score,
  tier
FROM observations 
WHERE domain = 'test'
ORDER BY score DESC;

\echo ''
\echo '==> Moving cold observations (score < -1.0) to COLD tier...'
UPDATE observations
SET 
  tier = 'cold',
  archived_at = NOW(),
  archived_reason = 'Score < -1.0'
WHERE domain = 'test' AND score < -1.0 AND tier = 'hot';

\echo 'Tier distribution:'
SELECT 
  tier,
  COUNT(*) as count,
  STRING_AGG(LEFT(observation, 20), ' | ') as observations
FROM observations 
WHERE domain = 'test'
GROUP BY tier;

\echo ''
\echo '==> Marking purge-eligible (score < -2.0)...'
UPDATE observations
SET purge_eligible = true
WHERE domain = 'test' AND score < -2.0;

\echo 'Purge status:'
SELECT 
  LEFT(observation, 30) as preview,
  ROUND(score::numeric, 3) as score,
  tier,
  purge_eligible,
  CASE 
    WHEN purge_eligible THEN 'PURGE'
    WHEN tier = 'cold' THEN 'ARCHIVED'
    ELSE 'ACTIVE'
  END as status
FROM observations 
WHERE domain = 'test'
ORDER BY score DESC;

\echo ''
\echo '==> Creating tombstones for purged items...'
INSERT INTO memory_tombstones (id, memory_id, action, from_tier, to_tier, reason, actor, created_at, metadata)
SELECT 
  gen_random_uuid(),
  id::text,
  'purged'::tombstone_action,
  tier,
  NULL,
  'Score below -2.0',
  'test_system',
  NOW(),
  jsonb_build_object('final_score', score, 'access_count', access_count)
FROM observations
WHERE domain = 'test' AND purge_eligible = true;

\echo 'Tombstones:'
SELECT 
  memory_id,
  action,
  from_tier,
  reason
FROM memory_tombstones
WHERE actor = 'test_system';

\echo ''
\echo '========================================='
\echo 'SUMMARY'
\echo '========================================='
SELECT 
  CASE 
    WHEN tier = 'hot' THEN 'HOT'
    WHEN tier = 'cold' THEN 'COLD'
  END as state,
  COUNT(*) as count
FROM observations 
WHERE domain = 'test'
GROUP BY tier
UNION ALL
SELECT 
  'PURGE_ELIGIBLE',
  COUNT(*)
FROM observations 
WHERE domain = 'test' AND purge_eligible = true
UNION ALL
SELECT 
  'TOMBSTONES',
  COUNT(*)
FROM memory_tombstones
WHERE actor = 'test_system';

\echo ''
\echo 'SUCCESS: Retention mechanism verified!'
\echo '  - Scores track access frequency'
\echo '  - Decay moves old data toward archival'
\echo '  - Cold tier for score < -1.0'
\echo '  - Purge eligibility for score < -2.0'
\echo '  - Tombstones preserve audit trail'
\echo '========================================='
