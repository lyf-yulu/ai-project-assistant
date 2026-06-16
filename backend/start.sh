#!/bin/bash
set -e
cd "$(dirname "$0")"
exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 120
