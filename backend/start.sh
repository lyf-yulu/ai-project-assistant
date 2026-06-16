#!/bin/bash
set -e
# Render 启动脚本
cd "$(dirname "$0")"

# 预加载嵌入模型，避免首次请求时下载超时
echo "预加载嵌入模型..."
python3 -c "
from embedder import embed_query
embed_query('warmup')
print('模型就绪')
"

exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 120
