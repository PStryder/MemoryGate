#!/bin/sh
set -e

mode="${1:-serve}"
shift || true

case "$mode" in
  migrate)
    alembic upgrade head
    ;;
  serve)
    exec uvicorn app.main:asgi_app --host 0.0.0.0 --port "${PORT:-8080}" "$@"
    ;;
  migrate-and-serve)
    alembic upgrade head
    exec uvicorn app.main:asgi_app --host 0.0.0.0 --port "${PORT:-8080}" "$@"
    ;;
  *)
    exec "$mode" "$@"
    ;;
esac
