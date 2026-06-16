#!/bin/bash
set -e
cd "$(dirname "$0")"

# 确保 HF 缓存目录存在
mkdir -p "${HF_HOME:-/opt/render/project/src/.hf_cache}"

exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 120
