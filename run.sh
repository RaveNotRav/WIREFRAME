#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

HOST="${WIREFRAME_HOST:-0.0.0.0}"
PORT="${WIREFRAME_PORT:-8000}"

echo "Starting WIREFRAME on ${HOST}:${PORT}"
exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
