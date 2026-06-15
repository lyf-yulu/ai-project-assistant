# AI 项目助手

基于项目文档的 RAG 问答应用。同事遇到问题时，用自然语言提问即可获得基于项目源码和文档的准确回答。

## 架构

- `indexer/` — 本地索引脚本，将项目文件转为 FAISS 向量库
- `backend/` — Flask 后端，处理问答请求（部署到 Render）
- `frontend/` — 静态前端（部署到 GitHub Pages）

## 快速开始

### 1. 构建知识库

```bash
cd indexer
pip install -r requirements.txt
python indexer.py
```

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 3. 打开前端

用浏览器打开 `frontend/index.html`，或部署到 GitHub Pages。

## 添加新项目

在 `indexer/indexer.py` 的 `PROJECTS` dict 中添加一行，重新运行索引脚本。
