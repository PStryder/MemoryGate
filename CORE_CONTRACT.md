# Core Contract

This repository is split into:

- `core/` — importable, standalone business logic shared with the SaaS version.
- `app/` — standalone wiring (FastAPI setup, middleware, routes, auth glue).

Core rules:

- `core/` MUST NOT import `app/` (core is a library).
- `app/` may import `core/` to compose a runnable service.
- Keep the core API stable so SaaS can rely on it without rework.

Stable public surface (intended for SaaS imports):

- `core.context` (`AuthContext`, `RequestContext`)
- `core.models` (SQLAlchemy models + enums)
- `core.services.memory` (public memory service functions)

If new features are added, prefer extending `core/` and only wiring in `app/`.
