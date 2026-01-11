# Configuration

MemoryGate is configured via environment variables. Defaults are defined in
core/config.py, rate_limiter.py, security_middleware.py, and app/middleware.py.

## Database and Storage

- DB_BACKEND: Database backend (postgres or sqlite, default: postgres).
- VECTOR_BACKEND: Vector backend (pgvector, sqlite_vss, none, default: pgvector).
- DATABASE_URL: SQLAlchemy connection URL (required for postgres).
- SQLITE_PATH: SQLite file path when DB_BACKEND=sqlite (default: /data/memorygate.db).
- AUTO_CREATE_EXTENSIONS: Auto-create pgvector extension (default: true).
- AUTO_MIGRATE_ON_STARTUP: Auto-run alembic migrations (default: true).
- MEMORYGATE_TENANCY_MODE: Must be single (default: single).
- MEMORYGATE_INSTANCE_ID: Instance ID shown in /health (default: memorygate-1).

Notes:
- VECTOR_BACKEND=sqlite_vss is currently not wired and falls back to keyword search.
- DB_BACKEND=sqlite with VECTOR_BACKEND=pgvector is invalid and will fail startup.

## Embedding Provider

- OPENAI_API_KEY: Required for EMBEDDING_PROVIDER=openai.
- EMBEDDING_PROVIDER: openai, local_cpd, or none (default: openai).
- EMBEDDING_TIMEOUT_SECONDS: Request timeout (default: 30).
- EMBEDDING_RETRY_MAX: Retry count (default: 2).
- EMBEDDING_RETRY_BACKOFF_SECONDS: Backoff base seconds (default: 0.5).
- EMBEDDING_RETRY_JITTER_SECONDS: Backoff jitter seconds (default: 0.25).
- EMBEDDING_FAILURE_THRESHOLD: Circuit breaker failure threshold (default: 5).
- EMBEDDING_COOLDOWN_SECONDS: Circuit breaker cooldown (default: 60).
- EMBEDDING_HEALTHCHECK_ENABLED: Health endpoint external probe (default: true).
- EMBEDDING_BACKFILL_ENABLED: Background backfill task (default: true).
- EMBEDDING_BACKFILL_INTERVAL_SECONDS: Backfill cadence (default: 300).
- EMBEDDING_BACKFILL_BATCH_LIMIT: Backfill batch size (default: 50).

## Input and Validation Limits

- MEMORYGATE_MAX_RESULT_LIMIT: Max results returned (default: 100).
- MEMORYGATE_MAX_QUERY_LENGTH: Query length cap (default: 4000).
- MEMORYGATE_MAX_TEXT_LENGTH: Long text cap (default: 8000).
- MEMORYGATE_MAX_SHORT_TEXT_LENGTH: Short text cap (default: 255).
- MEMORYGATE_MAX_DOMAIN_LENGTH: Domain length cap (default: 100).
- MEMORYGATE_MAX_TITLE_LENGTH: Title length cap (default: 500).
- MEMORYGATE_MAX_URL_LENGTH: URL length cap (default: 1000).
- MEMORYGATE_MAX_DOC_TYPE_LENGTH: Document type cap (default: 50).
- MEMORYGATE_MAX_CONCEPT_TYPE_LENGTH: Concept type cap (default: 50).
- MEMORYGATE_MAX_STATUS_LENGTH: Status cap (default: 50).
- MEMORYGATE_MAX_METADATA_BYTES: JSON metadata size cap (default: 20000).
- MEMORYGATE_MAX_LIST_ITEMS: Max list size (default: 50).
- MEMORYGATE_MAX_LIST_ITEM_LENGTH: Max list item length (default: 1000).
- MEMORYGATE_MAX_TAG_ITEMS: Max tag items (default: same as MAX_LIST_ITEMS).
- MEMORYGATE_MAX_RELATIONSHIP_ITEMS: Max relationship items (default: MAX_LIST_ITEMS).
- MEMORYGATE_TOOL_INVENTORY_RETRY_SECONDS: Tool inventory backoff (default: 5).
- MEMORYGATE_MAX_EMBEDDING_TEXT_LENGTH: Embedding input cap (default: 8000).

## Retention and Scoring

- SCORE_BUMP_ALPHA: Access bump coefficient (default: 0.4).
- REHYDRATE_BUMP_ALPHA: Rehydrate bump coefficient (default: 0.2).
- SCORE_DECAY_BETA: Decay coefficient per tick (default: 0.02).
- SCORE_CLAMP_MIN: Minimum score (default: -3.0).
- SCORE_CLAMP_MAX: Maximum score (default: 1.0).
- SUMMARY_TRIGGER_SCORE: Archive threshold (default: -1.0).
- PURGE_TRIGGER_SCORE: Purge threshold (default: -2.0).
- RETENTION_PRESSURE: Pressure multiplier (default: 1.0).
- RETENTION_BUDGET: Budget for pressure calculation (default: 100000).
- RETENTION_TICK_SECONDS: Retention loop interval (default: 900).
- COLD_DECAY_MULTIPLIER: Cold-tier decay multiplier (default: 0.25).
- FORGET_MODE: soft or hard (default: soft).
- COLD_SEARCH_ENABLED: Allow search_cold_memory (default: true).
- ARCHIVE_LIMIT_DEFAULT: Default archive limit (default: 200).
- ARCHIVE_LIMIT_MAX: Max archive limit (default: 500).
- REHYDRATE_LIMIT_MAX: Max rehydrate limit (default: 200).
- TOMBSTONES_ENABLED: Write memory tombstones (default: true).
- SUMMARY_MAX_LENGTH: Summary truncation limit (default: 800).
- SUMMARY_BATCH_LIMIT: Summaries per tick (default: 100).
- RETENTION_PURGE_LIMIT: Purge batch size (default: 100).
- ALLOW_HARD_PURGE_WITHOUT_SUMMARY: Allow hard purge without summary (default: false).
- STORAGE_QUOTA_BYTES: Archive storage budget (default: 10000000000).
- ARCHIVE_MULTIPLIER: Archive capacity multiplier (default: 2.0).

## OAuth and API Keys

- REQUIRE_MCP_AUTH: Enforce API key gate on /mcp (default: true).
- CLEANUP_INTERVAL_SECONDS: OAuth state/session cleanup loop (default: 900).
- OAUTH_REDIRECT_BASE: OAuth issuer base (default: http://localhost:8000).
- FRONTEND_URL: UI base for redirects and CORS fallback (default: http://localhost:3000).
- GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET: OAuth provider config.
- GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET: OAuth provider config.
- PSTRYDER_DESKTOP: MCP client ID for PKCE flow.
- PSTRYDER_DESKTOP_SECRET: MCP client secret for PKCE flow.
- PSTRYDER_DESKTOP_REDIRECT_URIS: Allowed redirect URIs (comma separated).

## Rate Limiting

- RATE_LIMIT_ENABLED: Master switch (default: true).
- RATE_LIMIT_GLOBAL_PER_IP: Global IP limit (default: 120).
- RATE_LIMIT_GLOBAL_WINDOW_SECONDS: Global IP window (default: 60).
- RATE_LIMIT_API_KEY_PER_KEY: API key limit (default: 600).
- RATE_LIMIT_API_KEY_WINDOW_SECONDS: API key window (default: 60).
- RATE_LIMIT_AUTH_PER_IP: Auth path limit (default: 10).
- RATE_LIMIT_AUTH_WINDOW_SECONDS: Auth window (default: 60).
- RATE_LIMIT_MAX_CACHE_ENTRIES: In-memory limiter size (default: 10000).
- RATE_LIMIT_TRUSTED_PROXY_COUNT: Trusted proxy hop count (default: 0).
- RATE_LIMIT_TRUSTED_PROXY_IPS: Trusted proxy IPs (comma separated).
- RATE_LIMIT_REDIS_FAIL_OPEN: Use fallback if Redis fails (default: true).
- RATE_LIMIT_REDIS_URL: Redis URL for rate limiting (optional).
- REDIS_URL: Redis URL fallback key (optional).

## Security Headers and Request Limits

- SECURITY_HEADERS_ENABLED: Enable security headers (default: true).
- SECURITY_HEADERS_ENABLE_HSTS: Enable HSTS (default: false).
- SECURITY_HEADERS_HSTS_MAX_AGE: HSTS max-age (default: 31536000).
- SECURITY_HEADERS_HSTS_INCLUDE_SUBDOMAINS: Include subdomains (default: true).
- SECURITY_HEADERS_HSTS_PRELOAD: Enable preload flag (default: false).
- SECURITY_HEADERS_REFERRER_POLICY: Referrer-Policy (default: no-referrer).
- SECURITY_HEADERS_X_FRAME_OPTIONS: X-Frame-Options (default: DENY).
- SECURITY_HEADERS_PERMISSIONS_POLICY: Permissions-Policy (default: geolocation=(), microphone=(), camera=()).
- SECURITY_HEADERS_CSP: Content-Security-Policy override (optional).
- REQUEST_SIZE_LIMIT_ENABLED: Enable request body limits (default: true).
- MAX_REQUEST_BODY_BYTES: Max request size in bytes (default: 262144).

## CORS and Trusted Hosts

- CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins.
  If unset, defaults to FRONTEND_URL plus memorygate.ai and localhost.
- TRUSTED_HOSTS: Comma-separated allowed hosts for TrustedHostMiddleware.
