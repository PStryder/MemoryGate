# Production Environment Settings

These settings are configured via environment variables. Defaults remain developer-friendly; for production, override them explicitly.

## Database safety

- `AUTO_MIGRATE_ON_STARTUP=false`
- `AUTO_CREATE_EXTENSIONS=false` (manage extensions separately in production)

## CORS and trusted hosts

- `CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com`
- `TRUSTED_HOSTS=app.example.com,admin.example.com,api.example.com`

If `CORS_ALLOWED_ORIGINS` is unset, the app falls back to `FRONTEND_URL` and the default
MemoryGate domains. Always set the explicit list for production deployments.

## Security headers (HSTS)

- `SECURITY_HEADERS_ENABLE_HSTS=true`
- `SECURITY_HEADERS_HSTS_MAX_AGE=31536000`
- `SECURITY_HEADERS_HSTS_INCLUDE_SUBDOMAINS=true`
- `SECURITY_HEADERS_HSTS_PRELOAD=false`

Enable `SECURITY_HEADERS_HSTS_PRELOAD` only after verifying you meet the preload requirements.

## Rate limiting

Example baseline limits (tune based on traffic):

- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_GLOBAL_PER_IP=120`
- `RATE_LIMIT_GLOBAL_WINDOW_SECONDS=60`
- `RATE_LIMIT_API_KEY_PER_KEY=600`
- `RATE_LIMIT_API_KEY_WINDOW_SECONDS=60`
- `RATE_LIMIT_AUTH_PER_IP=10`
- `RATE_LIMIT_AUTH_WINDOW_SECONDS=60`
- `RATE_LIMIT_MAX_CACHE_ENTRIES=10000`
- `RATE_LIMIT_TRUSTED_PROXY_COUNT=1`
- `RATE_LIMIT_TRUSTED_PROXY_IPS=10.0.0.1,10.0.0.2`
- `RATE_LIMIT_REDIS_URL=redis://...` (optional)

## Request size limits

- `REQUEST_SIZE_LIMIT_ENABLED=true`
- `MAX_REQUEST_BODY_BYTES=262144`

## Archive retention and quotas

- `STORAGE_QUOTA_BYTES=10000000000`
- `ARCHIVE_MULTIPLIER=2.0`

Archive eviction is automatic when `STORAGE_QUOTA_BYTES * ARCHIVE_MULTIPLIER` is exceeded.
See `docs/ARCHIVE_RETENTION.md` for policy details.

## OAuth state TTL

OAuth state tokens expire after 10 minutes and are single-use. No additional configuration is required.
