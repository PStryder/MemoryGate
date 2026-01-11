# Scaling Recommendations

These are recommendations only; no background queues or async rewrites are implemented here.

## Database access

- Enable connection pooling in the database driver.
- Set pool size and overflow limits based on deployment capacity.
- Consider async DB drivers for high-concurrency workloads.

## Background work

Move long-running tasks out of request handlers:

- Embedding generation and backfill.
- Summarization and archival workflows.
- Batch rehydration and retention ticks.

Queue options include Celery, Dramatiq, or a lightweight task runner with cron scheduling.

## Idempotency and concurrency

- Ensure lifecycle operations are idempotent (e.g., archive/rehydrate called multiple times).
- Use database-level constraints to prevent duplicate inserts.
- Add concurrency tests around common mutation paths to validate thread safety.
