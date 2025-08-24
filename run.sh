#!/usr/bin/env bash
set -euo pipefail
PORT_TO_USE="${PORT:-8000}"
echo "[KB] Starting on port ${PORT_TO_USE} ..."
uvicorn app:app --host 0.0.0.0 --port "${PORT_TO_USE}"
