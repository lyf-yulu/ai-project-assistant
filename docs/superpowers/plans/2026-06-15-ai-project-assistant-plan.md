# AI 项目助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个网页端 RAG 问答应用，同事可以用自然语言对 ai-generation-portable-apps 项目进行故障排查。

**Architecture:** 三组件：本地索引脚本(indexer) + Flask 后端(Render) + 静态前端(GitHub Pages)。嵌入用 sentence-transformers (BGE-small-zh-v1.5)，问答用 DeepSeek Chat API，向量检索用 FAISS。

**Tech Stack:** Python 3.11, Flask, FAISS, sentence-transformers, DeepSeek API (openai SDK), HTML/CSS/JS

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `indexer/requirements.txt`
- Create: `backend/requirements.txt`
- Create: `backend/runtime.txt`
- Create: `backend/start.sh`

- [ ] **Step 1: 创建 README.md**

`/Users/260413a/Desktop/ai-project-assistant/README.md`
```markdown
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
```

- [ ] **Step 2: 创建 .gitignore**

`/Users/260413a/Desktop/ai-project-assistant/.gitignore`
```
__pycache__/
*.pyc
.env
.venv/
venv/
.DS_Store
```

- [ ] **Step 3: 创建 indexer/requirements.txt**

`/Users/260413a/Desktop/ai-project-assistant/indexer/requirements.txt`
```
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
numpy>=1.26.0
```

- [ ] **Step 4: 创建 backend/requirements.txt**

`/Users/260413a/Desktop/ai-project-assistant/backend/requirements.txt`
```
flask>=3.0.0
flask-cors>=4.0.0
gunicorn>=22.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0
sentence-transformers>=2.7.0
openai>=1.30.0
```

- [ ] **Step 5: 创建 backend/runtime.txt**

`/Users/260413a/Desktop/ai-project-assistant/backend/runtime.txt`
```
python-3.11.0
```

- [ ] **Step 6: 创建 backend/start.sh**

`/Users/260413a/Desktop/ai-project-assistant/backend/start.sh`
```bash
#!/bin/bash
# Render 启动脚本
cd "$(dirname "$0")"
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

- [ ] **Step 7: 赋予启动脚本执行权限**

Run: `chmod +x /Users/260413a/Desktop/ai-project-assistant/backend/start.sh`

- [ ] **Step 8: 初始化 Git 仓库**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git init
git add -A
git commit -m "chore: project skeleton

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 嵌入模型封装 (`embedder.py`)

**说明:** `embedder.py` 被索引脚本和后端共同导入，保证两端使用同一模型和同一向量维度。放在 `backend/` 下，索引脚本通过 `sys.path` 引用。

**Files:**
- Create: `backend/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: 创建 embedder.py**

`/Users/260413a/Desktop/ai-project-assistant/backend/embedder.py`
```python
"""嵌入模型封装。索引脚本和后端共用此模块，保证向量维度一致。"""
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512  # BGE-small-zh-v1.5 输出维度

_model = None


def _get_model():
    """延迟加载模型（单例）。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转为向量数组，shape=(len(texts), VECTOR_DIM)，已归一化。"""
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """将单个查询文本转为向量，shape=(VECTOR_DIM,)，已归一化。"""
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embedding, dtype=np.float32)
```

- [ ] **Step 2: 创建 test_embedder.py**

`/Users/260413a/Desktop/ai-project-assistant/tests/test_embedder.py`
```python
"""测试嵌入模型封装。需先安装 backend/requirements.txt 的依赖。"""
import sys
sys.path.insert(0, "backend")

import numpy as np
from embedder import embed_texts, embed_query, VECTOR_DIM


def test_embed_texts_shape():
    """embed_texts 返回正确形状的数组。"""
    texts = ["你好世界", "Hello world"]
    result = embed_texts(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, VECTOR_DIM)
    assert result.dtype == np.float32


def test_embed_texts_normalized():
    """嵌入向量已归一化（L2 范数 ≈ 1）。"""
    result = embed_texts(["测试文本"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 0.01


def test_embed_query_shape():
    """embed_query 返回正确形状。"""
    result = embed_query("这是一个问题")
    assert isinstance(result, np.ndarray)
    assert result.shape == (VECTOR_DIM,)
    assert result.dtype == np.float32


def test_embed_query_normalized():
    """查询向量已归一化。"""
    result = embed_query("查询")
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 0.01


def test_embeddings_are_reproducible():
    """同一文本两次嵌入结果一致。"""
    a = embed_query("同一个问题")
    b = embed_query("同一个问题")
    np.testing.assert_array_almost_equal(a, b)


def test_semantic_distance():
    """语义相似的文本向量距离更近。"""
    a = embed_query("生成视频失败")
    b = embed_query("seedance 报错")
    c = embed_query("今天天气很好")
    dist_ab = np.linalg.norm(a - b)
    dist_ac = np.linalg.norm(a - c)
    assert dist_ab < dist_ac
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
pip install -r backend/requirements.txt
python -m pytest tests/test_embedder.py -v
```

Expected: 6 tests PASS（首次运行会下载模型 ~100MB）

- [ ] **Step 4: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add backend/embedder.py tests/test_embedder.py
git commit -m "feat: add embedding model wrapper with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 索引脚本 `indexer.py`

**Files:**
- Create: `indexer/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: 创建 indexer.py**

`/Users/260413a/Desktop/ai-project-assistant/indexer/indexer.py`
```python
#!/usr/bin/env python3
"""索引脚本：读取项目文件 → 分块 → 嵌入 → FAISS 索引。

用法:
    python indexer.py

添加新项目：在 PROJECTS dict 中添加一行，重新运行即可。
"""
import sys
import os
import json
import re
from pathlib import Path

# 引用 backend/ 下的 embedder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from embedder import embed_texts

import faiss
import numpy as np

# ── 配置 ──────────────────────────────────────────────
PROJECTS = {
    "ai-gen-apps": str(Path.home() / "Desktop/ai-generation-portable-apps"),
    "ai-gen-apps-v0.1": str(Path.home() / "Desktop/ai-generation-portable-apps-v0.1"),
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "backend/knowledge"

EXTENSIONS = {".md", ".py", ".html", ".js", ".css", ".json"}

SKIP_DIRS = {"outputs", "archives", "logs", "state", "__pycache__",
             ".git", "node_modules", ".superpowers", "data", "release"}

SKIP_FILES = {"config.json", "config_github.json", ".DS_Store"}

CHUNK_SIZE = 800   # 每块最大字符数
CHUNK_OVERLAP = 100  # 块间重叠字符数


# ── 文件收集 ──────────────────────────────────────────
def collect_files(project_dir: str) -> list[Path]:
    """递归收集 project_dir 下所有应索引的文件路径。"""
    root = Path(project_dir)
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if path.name in SKIP_FILES:
            continue
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        files.append(path)
    return files


# ── 文本分块 ──────────────────────────────────────────
def split_by_paragraphs(text: str, file_path: Path) -> list[dict]:
    """按段落分块（通用策略，适用于 .md / .txt）。"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    current_line = 1
    line_start = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_lines = para.count("\n") + 1
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append({
                "content": current.strip(),
                "file": str(file_path),
                "line_start": line_start,
                "line_end": current_line - 1,
            })
            current = para
            line_start = current_line
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para
        current_line += para_lines

    if current.strip():
        chunks.append({
            "content": current.strip(),
            "file": str(file_path),
            "line_start": line_start,
            "line_end": current_line,
        })
    return chunks


def split_python(text: str, file_path: Path) -> list[dict]:
    """Python 文件：按函数/类边界分块。"""
    lines = text.split("\n")
    chunks = []
    current = []
    current_start = 1
    in_docstring = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # 检测函数或类定义
        is_def = bool(re.match(r"^(def |class )", stripped))

        if is_def and current and not in_docstring:
            chunk_text = "\n".join(current)
            if len(chunk_text) > 50:  # 过滤过短的块
                chunks.append({
                    "content": chunk_text,
                    "file": str(file_path),
                    "line_start": current_start,
                    "line_end": i - 1,
                })
            current = [line]
            current_start = i
        else:
            current.append(line)

        # 简单跟踪 docstring
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring

    if current:
        chunk_text = "\n".join(current)
        if len(chunk_text) > 30:
            chunks.append({
                "content": chunk_text,
                "file": str(file_path),
                "line_start": current_start,
                "line_end": len(lines),
            })
    return chunks


def chunk_file(file_path: Path) -> list[dict]:
    """读取文件并分块，返回 chunk 列表。"""
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    if not text.strip():
        return []

    ext = file_path.suffix.lower()
    if ext == ".py":
        chunks = split_python(text, file_path)
    elif ext == ".json":
        # JSON 作为一个块
        if len(text) < CHUNK_SIZE * 2:
            return [{"content": text, "file": str(file_path), "line_start": 1, "line_end": text.count(chr(10)) + 1}]
        else:
            chunks = split_by_paragraphs(text, file_path)
    else:
        chunks = split_by_paragraphs(text, file_path)

    # 为每个 chunk 追加 heading 上下文（从 .md 文件中提取）
    for chunk in chunks:
        if file_path.suffix == ".md":
            heading = extract_heading_context(text, chunk["line_start"])
            if heading:
                chunk["content"] = heading + "\n" + chunk["content"]

    return chunks


def extract_heading_context(text: str, line_num: int) -> str:
    """提取 .md 文件中指定行之前的标题层级作为上下文前缀。"""
    lines = text.split("\n")
    headings = []
    for i in range(min(line_num, len(lines))):
        m = re.match(r"^(#{1,4})\s+(.+)", lines[i])
        if m:
            level = len(m.group(1))
            headings = [h for h in headings if len(h.split()[0]) < level]
            headings.append(m.group(0))
    return " > ".join(headings) if headings else ""


# ── 索引构建 ──────────────────────────────────────────
def index_project(project_name: str, project_dir: str):
    """为一个项目构建 FAISS 索引。"""
    print(f"\n{'='*60}")
    print(f"索引项目: {project_name}")
    print(f"路径: {project_dir}")
    print(f"{'='*60}")

    files = collect_files(project_dir)
    print(f"收集到 {len(files)} 个文件")

    all_chunks = []
    for fp in files:
        chunks = chunk_file(fp)
        rel = fp.relative_to(project_dir)
        for ch in chunks:
            ch["file"] = str(rel)  # 转为相对路径
            ch["project"] = project_name
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  警告: 没有可索引的内容")
        return

    print(f"共 {len(all_chunks)} 个文本块")

    # 嵌入
    texts = [ch["content"] for ch in all_chunks]
    print("正在生成嵌入向量...")
    embeddings = embed_texts(texts)
    print(f"向量维度: {embeddings.shape}")

    # FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积搜索（配合归一化向量 = 余弦相似度）
    index.add(embeddings)

    # 写入文件
    out_dir = OUTPUT_DIR / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        for ch in all_chunks:
            # 不存 content 到 json（已经在向量里），只存元数据
            meta = {k: v for k, v in ch.items() if k != "content"}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"索引已写入: {out_dir}")
    print(f"  - index.faiss ({index.ntotal} 个向量)")
    print(f"  - chunks.json ({len(all_chunks)} 条元数据)")


# ── 入口 ──────────────────────────────────────────────
def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    for name, path in PROJECTS.items():
        if not Path(path).exists():
            print(f"警告: 项目路径不存在，跳过: {name} ({path})")
            continue
        index_project(name, path)

    print(f"\n完成！知识库位于: {OUTPUT_DIR}")
    print("请将 knowledge/ 目录提交到 Git 仓库。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建 test_indexer.py**

`/Users/260413a/Desktop/ai-project-assistant/tests/test_indexer.py`
```python
"""测试索引脚本的分块和文件收集逻辑。"""
import sys
import os
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "indexer"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import indexer


def test_collect_files_md_and_py():
    """collect_files 收集 .md 和 .py 文件，跳过配置中排除的目录和文件。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("# Test\n\nHello world")
        (root / "app.py").write_text("def foo():\n    pass\n")
        (root / "outputs").mkdir()
        (root / "outputs/result.png").write_text("fake")
        (root / "__pycache__").mkdir()
        (root / "__pycache__/x.pyc").write_text("fake")
        (root / "config.json").write_text("{}")

        files = indexer.collect_files(tmp)
        names = {f.name for f in files}

        assert "README.md" in names
        assert "app.py" in names
        assert "result.png" not in names
        assert "x.pyc" not in names
        assert "config.json" not in names


def test_split_by_paragraphs():
    """按段落分块，每个段落不是空行分割。"""
    text = "# 标题\n\n这是第一段内容。\n\n这是第二段内容。"
    chunks = indexer.split_by_paragraphs(text, Path("test.md"))
    assert len(chunks) >= 1
    # 至少有一个非空块
    for ch in chunks:
        assert ch["content"].strip()
        assert "file" in ch
        assert "line_start" in ch
        assert "line_end" in ch


def test_split_python():
    """Python 文件按函数边界分块。"""
    code = '''"""模块文档"""

def foo():
    """foo 的文档"""
    return 1

def bar():
    """bar 的文档"""
    x = 2
    return x
'''
    chunks = indexer.split_python(code, Path("test.py"))
    assert len(chunks) >= 2  # foo + bar
    for ch in chunks:
        assert "file" in ch
        assert "line_start" in ch
        assert "line_end" in ch
        assert len(ch["content"]) > 30


def test_split_python_empty():
    """空文件返回空列表。"""
    chunks = indexer.split_python("", Path("empty.py"))
    assert chunks == []


def test_chunk_file_md():
    """chunk_file 处理 .md 文件。"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write("# 标题\n\n段落一内容测试。\n\n## 子标题\n\n段落二内容测试。\n")
        tmp_path = f.name

    try:
        chunks = indexer.chunk_file(Path(tmp_path))
        assert len(chunks) > 0
    finally:
        os.unlink(tmp_path)


def test_chunk_file_py():
    """chunk_file 处理 .py 文件。"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as f:
        f.write("def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
        tmp_path = f.name

    try:
        chunks = indexer.chunk_file(Path(tmp_path))
        assert len(chunks) >= 2
        for ch in chunks:
            assert ch["file"] == tmp_path.name or ch["file"].endswith(f.name)
    finally:
        os.unlink(tmp_path)


def test_skip_dirs():
    """SKIP_DIRS 覆盖常见排除目录。"""
    assert "outputs" in indexer.SKIP_DIRS
    assert "archives" in indexer.SKIP_DIRS
    assert "__pycache__" in indexer.SKIP_DIRS
    assert ".git" in indexer.SKIP_DIRS


def test_projects_config_keys():
    """PROJECTS 配置至少包含两个已知项目。"""
    assert "ai-gen-apps" in indexer.PROJECTS
    assert "ai-gen-apps-v0.1" in indexer.PROJECTS
```

- [ ] **Step 3: 安装依赖并运行分块测试（不涉及嵌入模型）**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
pip install -r indexer/requirements.txt
pip install -r backend/requirements.txt
python -m pytest tests/test_indexer.py -v -k "not test_split_and" 
```

Expected: 所有分块相关测试 PASS（~4 tests）

- [ ] **Step 4: 运行全量索引脚本**

```bash
cd /Users/260413a/Desktop/ai-project-assistant/indexer
python indexer.py
```

Expected: 收集到文件，生成向量，输出 knowledge/ 目录下的 .faiss 和 chunks.json

- [ ] **Step 5: 验证输出文件存在**

```bash
ls -la /Users/260413a/Desktop/ai-project-assistant/backend/knowledge/ai-gen-apps/
ls -la /Users/260413a/Desktop/ai-project-assistant/backend/knowledge/ai-gen-apps-v0.1/
```

Expected: 每个目录下存在 index.faiss 和 chunks.json

- [ ] **Step 6: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add indexer/indexer.py tests/test_indexer.py backend/knowledge/
git commit -m "feat: add indexer script with chunking and FAISS output

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: FAISS 检索模块 (`retriever.py`)

**Files:**
- Create: `backend/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: 创建 retriever.py**

`/Users/260413a/Desktop/ai-project-assistant/backend/retriever.py`
```python
"""FAISS 检索模块：加载索引、执行语义搜索。"""
import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from embedder import embed_query

KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", Path(__file__).resolve().parent / "knowledge"))
SIMILARITY_THRESHOLD = 0.5
TOP_K = 5

# 缓存已加载的索引: {project_name: (index, chunks_meta)}
_index_cache: dict[str, tuple[faiss.Index, list[dict]]] = {}


def _load_project(project_name: str) -> tuple[faiss.Index, list[dict]]:
    """加载单个项目的 FAISS 索引和元数据（带缓存）。"""
    if project_name in _index_cache:
        return _index_cache[project_name]

    proj_dir = KNOWLEDGE_DIR / project_name
    index_path = proj_dir / "index.faiss"
    chunks_path = proj_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"知识库不存在: {project_name}")

    index = faiss.read_index(str(index_path))
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _index_cache[project_name] = (index, chunks)
    return index, chunks


def list_projects() -> list[str]:
    """列出 KNOWLEDGE_DIR 下所有可用的项目名称。"""
    if not KNOWLEDGE_DIR.exists():
        return []
    return sorted(
        d.name for d in KNOWLEDGE_DIR.iterdir()
        if d.is_dir() and (d / "index.faiss").exists()
    )


def search(query: str, project: Optional[str] = None) -> list[dict]:
    """语义搜索。

    Args:
        query: 用户问题
        project: 限定项目名称，None 则搜索所有项目

    Returns:
        [{project, file, line_start, line_end, similarity, snippet}, ...]
        按相似度降序排列，最多 TOP_K 条。
    """
    query_vec = embed_query(query).astype(np.float32).reshape(1, -1)

    if project:
        project_names = [project]
    else:
        project_names = list_projects()

    all_results = []
    for proj_name in project_names:
        try:
            index, chunks = _load_project(proj_name)
        except FileNotFoundError:
            continue

        if index.ntotal == 0:
            continue

        scores, indices = index.search(query_vec, TOP_K)

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            score = float(score)
            if score < SIMILARITY_THRESHOLD:
                continue
            meta = dict(chunks[idx])
            meta["project"] = proj_name
            meta["similarity"] = round(score, 4)
            meta["snippet"] = _get_snippet(proj_name, meta, max_len=200)
            all_results.append(meta)

    # 按相似度降序排列
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    return all_results[:TOP_K]


def _get_snippet(project_name: str, meta: dict, max_len: int = 200) -> str:
    """从源文件读取指定行范围的文本片段。"""
    try:
        proj_dir = None
        # 尝试定位项目根目录（此处用相对路径）
        # 实际存储的是相对项目根目录的路径
        file_path = KNOWLEDGE_DIR.parent.parent  # 回退
        full_path = None
        for candidate in [
            Path.home() / "Desktop/ai-generation-portable-apps",
            Path.home() / "Desktop/ai-generation-portable-apps-v0.1",
        ]:
            test_path = candidate / meta["file"]
            if test_path.exists():
                full_path = test_path
                break

        if full_path is None:
            return "(源文件未找到)"

        lines = full_path.read_text(encoding="utf-8").split("\n")
        start = max(0, meta["line_start"] - 1)
        end = min(len(lines), meta.get("line_end", start + 5))
        snippet = "\n".join(lines[start:end])
        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."
        return snippet
    except Exception:
        return "(无法读取文件)"
```

- [ ] **Step 2: 创建 test_retriever.py**

`/Users/260413a/Desktop/ai-project-assistant/tests/test_retriever.py`
```python
"""测试检索模块。依赖已构建的 knowledge/ 目录。"""
import sys
import os
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
import faiss
from retriever import search, list_projects, KNOWLEDGE_DIR, _load_project


def test_list_projects():
    """list_projects 返回已索引的项目列表。"""
    projects = list_projects()
    assert "ai-gen-apps" in projects
    assert "ai-gen-apps-v0.1" in projects


def test_search_all_projects():
    """搜索所有项目返回相关结果。"""
    results = search("seedance 生成失败 timeout")
    assert len(results) > 0
    assert len(results) <= 5
    for r in results:
        assert "project" in r
        assert "file" in r
        assert "similarity" in r
        assert r["similarity"] > 0


def test_search_specific_project():
    """限定项目搜索只返回该项目的文档。"""
    results = search("如何设置参数", project="ai-gen-apps")
    for r in results:
        assert r["project"] == "ai-gen-apps"


def test_search_no_results():
    """搜索无关内容返回空列表。"""
    results = search("asdfqwerzxcv12345无关内容")
    # 可能返回空或低相似度的结果被过滤
    assert len(results) == 0 or all(r["similarity"] < 0.5 for r in results)


def test_search_empty_query():
    """空查询返回空列表。"""
    results = search("")
    assert len(results) == 0
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
python -m pytest tests/test_retriever.py -v
```

Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add backend/retriever.py tests/test_retriever.py
git commit -m "feat: add FAISS retriever with semantic search

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Flask 后端 `app.py`

**Files:**
- Create: `backend/app.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: 创建 app.py**

`/Users/260413a/Desktop/ai-project-assistant/backend/app.py`
```python
"""AI 项目助手 - Flask 后端。

部署: Render Free Tier
启动: gunicorn app:app --bind 0.0.0.0:$PORT
"""
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

from retriever import search, list_projects

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_QUESTION_LENGTH = int(os.environ.get("MAX_QUESTION_LENGTH", "2000"))

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """你是 AI Generation Portable Apps 的技术支持助手。
你的知识来自项目文档和源码。请严格基于下面提供的文档片段回答问题。

规则：
1. 如果文档中有相关信息，请用中文直接回答，并在末尾标注来源文件。
2. 如果文档中没有涉及这个问题，请明确说"文档中没有涉及这个问题"，不要猜测或编造。
3. 回答要简洁、具体、可操作，像技术支持同事一样。
4. 不要提及"文档片段"、"上下文"这些内部术语，自然回答即可。"""


# ── 路由 ──────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """健康检查。"""
    return jsonify({
        "status": "ok",
        "projects": list_projects(),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """问答接口。"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_json", "message": "请求格式错误"}), 400

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question_required", "message": "请提供问题"}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({
            "error": "question_too_long",
            "message": f"问题不能超过{MAX_QUESTION_LENGTH}个字符"
        }), 400

    project = data.get("project") or None

    # 语义检索
    try:
        results = search(question, project=project)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return jsonify({"error": "search_error", "message": "检索服务异常"}), 500

    if not results:
        return jsonify({
            "answer": "文档中没有涉及这个问题。建议直接查看项目文档或联系开发者。",
            "sources": [],
        })

    # 构建上下文
    context_parts = []
    sources = []
    for r in results:
        context_parts.append(
            f"[来源: {r['project']}/{r['file']}:{r['line_start']}-{r.get('line_end', r['line_start'])}]\n"
            f"{r.get('snippet', '')}"
        )
        sources.append({
            "project": r["project"],
            "file": r["file"],
            "line": r["line_start"],
            "similarity": r["similarity"],
        })
    context = "\n\n---\n\n".join(context_parts)

    # 调用 DeepSeek
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"参考文档:\n{context}\n\n问题: {question}"},
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        return jsonify({"error": "api_error", "message": "服务暂时不可用，请稍后重试"}), 502

    return jsonify({
        "answer": answer,
        "sources": sources,
    })


# ── 入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
```

- [ ] **Step 2: 创建 test_backend.py**

`/Users/260413a/Desktop/ai-project-assistant/tests/test_backend.py`
```python
"""测试 Flask 后端。"""
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
from app import app as flask_app


@pytest.fixture
def client():
    """Flask 测试客户端。"""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_health(client):
    """健康检查返回 ok 和项目列表。"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert isinstance(data["projects"], list)
    assert "ai-gen-apps" in data["projects"]


def test_chat_empty_question(client):
    """空问题返回 400。"""
    resp = client.post("/api/chat", json={"question": ""})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "question_required"


def test_chat_missing_question(client):
    """缺少 question 字段返回 400。"""
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400


def test_chat_no_json(client):
    """非 JSON 请求返回 400。"""
    resp = client.post("/api/chat", data="not json")
    assert resp.status_code == 400


def test_chat_normal_question(client):
    """正常问题返回回答和来源。需要 DEEPSEEK_API_KEY 环境变量。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("需要 DEEPSEEK_API_KEY 环境变量")

    resp = client.post("/api/chat", json={
        "question": "Seedance 生成失败提示 timeout 怎么解决？"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_chat_with_project_filter(client):
    """限定项目搜索。需要 DEEPSEEK_API_KEY。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("需要 DEEPSEEK_API_KEY 环境变量")

    resp = client.post("/api/chat", json={
        "question": "nano banana 如何设置参数",
        "project": "ai-gen-apps",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["sources"]) > 0
    for s in data["sources"]:
        assert s["project"] == "ai-gen-apps"


def test_chat_irrelevant_question(client):
    """无关问题返回空来源。需要 DEEPSEEK_API_KEY。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("需要 DEEPSEEK_API_KEY 环境变量")

    resp = client.post("/api/chat", json={
        "question": "今天天气怎么样"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["sources"]) == 0
```

- [ ] **Step 3: 运行不需要 API Key 的测试**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
python -m pytest tests/test_backend.py -v -k "not chat_normal and not chat_with_project and not chat_irrelevant"
```

Expected: 4 tests PASS

- [ ] **Step 4: 设置 DEEPSEEK_API_KEY 并运行完整测试**

```bash
export DEEPSEEK_API_KEY="your-key-here"
cd /Users/260413a/Desktop/ai-project-assistant
python -m pytest tests/test_backend.py -v
```

Expected: 7 tests PASS（如有跳过的测试，设置 KEY 后重跑）

- [ ] **Step 5: 手动测试后端**

```bash
cd /Users/260413a/Desktop/ai-project-assistant/backend
python app.py
# 另开终端:
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Seedance 怎么设置参考图？"}'
```

Expected: 返回 JSON，包含 answer 和 sources

- [ ] **Step 6: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add backend/app.py tests/test_backend.py
git commit -m "feat: add Flask backend with /api/chat and /api/health

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 前端 HTML + CSS

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/styles.css`
- Create: `frontend/config.js`

- [ ] **Step 1: 创建 config.js**

`/Users/260413a/Desktop/ai-project-assistant/frontend/config.js`
```javascript
// 后端地址。部署到 Render 后替换为实际 URL。
// 本地开发: http://localhost:5000
// Render: https://your-app.onrender.com
const BACKEND_URL = "http://localhost:5000";
```

- [ ] **Step 2: 创建 styles.css**

`/Users/260413a/Desktop/ai-project-assistant/frontend/styles.css`
```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    height: 100vh;
    display: flex;
    background: #f5f5f5;
    color: #333;
}

/* ── 侧栏 ── */
.sidebar {
    width: 240px;
    min-width: 240px;
    background: #fff;
    border-right: 1px solid #e0e0e0;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}

.sidebar-section {
    padding: 16px;
    border-bottom: 1px solid #f0f0f0;
}

.sidebar-section h3 {
    font-size: 12px;
    text-transform: uppercase;
    color: #999;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.project-item {
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    margin-bottom: 4px;
    transition: background 0.15s;
}

.project-item:hover {
    background: #f5f5f5;
}

.project-item.active {
    background: #e6f4ff;
    color: #1677ff;
    font-weight: 500;
}

.history-item {
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    color: #666;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.15s;
}

.history-item:hover {
    background: #f5f5f5;
}

/* ── 主聊天区 ── */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.header {
    padding: 16px 24px;
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}

.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* 示例问题 */
.examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 16px;
}

.example-btn {
    padding: 8px 16px;
    background: #fff;
    border: 1px solid #d9d9d9;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
    color: #555;
    transition: all 0.15s;
}

.example-btn:hover {
    border-color: #1677ff;
    color: #1677ff;
}

/* 消息气泡 */
.message {
    display: flex;
    margin-bottom: 8px;
}

.message.user {
    justify-content: flex-end;
}

.message.assistant {
    justify-content: flex-start;
}

.bubble {
    max-width: 75%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.6;
    word-break: break-word;
}

.message.user .bubble {
    background: #1677ff;
    color: #fff;
    border-bottom-right-radius: 4px;
}

.message.assistant .bubble {
    background: #fff;
    border: 1px solid #e8e8e8;
    border-bottom-left-radius: 4px;
}

.bubble .source {
    display: block;
    margin-top: 8px;
    font-size: 12px;
    color: #1677ff;
    cursor: pointer;
}

.message.assistant .bubble .source {
    color: #1677ff;
}

.message.user .bubble .source {
    color: rgba(255, 255, 255, 0.8);
}

/* 输入区 */
.input-area {
    padding: 16px 24px;
    background: #fff;
    border-top: 1px solid #e0e0e0;
    display: flex;
    gap: 12px;
}

.input-area input {
    flex: 1;
    padding: 10px 16px;
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.15s;
}

.input-area input:focus {
    border-color: #1677ff;
}

.input-area button {
    padding: 10px 24px;
    background: #1677ff;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.15s;
}

.input-area button:hover {
    background: #4096ff;
}

.input-area button:disabled {
    background: #d9d9d9;
    cursor: not-allowed;
}

/* 状态 */
.loading-dots::after {
    content: "";
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0% { content: ""; }
    25% { content: "."; }
    50% { content: ".."; }
    75% { content: "..."; }
}

.error-banner {
    padding: 10px 16px;
    background: #fff2f0;
    border: 1px solid #ffccc7;
    border-radius: 8px;
    color: #ff4d4f;
    font-size: 13px;
    text-align: center;
}

.empty-state {
    text-align: center;
    color: #999;
    padding: 40px;
    font-size: 14px;
}

/* 响应式 */
@media (max-width: 640px) {
    .sidebar {
        display: none;
    }
    .bubble {
        max-width: 90%;
    }
}
```

- [ ] **Step 3: 创建 index.html**

`/Users/260413a/Desktop/ai-project-assistant/frontend/index.html`
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 项目助手</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <!-- 侧栏 -->
    <aside class="sidebar">
        <div class="sidebar-section">
            <h3>📂 项目</h3>
            <div id="project-list"></div>
        </div>
        <div class="sidebar-section">
            <h3>💬 最近对话</h3>
            <div id="history-list"></div>
        </div>
    </aside>

    <!-- 主聊天区 -->
    <div class="main">
        <div class="header">🤖 AI 项目助手</div>

        <div class="chat-area" id="chat-area">
            <!-- 示例问题（首次显示） -->
            <div class="examples" id="examples">
                <button class="example-btn">生成的时候提示 timeout 怎么解决？</button>
                <button class="example-btn">Seedance 如何设置参考图？</button>
                <button class="example-btn">Nano Banana 的参数是什么意思？</button>
            </div>
            <div class="empty-state" id="empty-state">
                选择一个项目或直接提问，AI 助手会基于项目文档回答。
            </div>
        </div>

        <!-- 输入区 -->
        <div class="input-area">
            <input
                type="text"
                id="question-input"
                placeholder="输入你的问题..."
                autocomplete="off"
            />
            <button id="send-btn">发送</button>
        </div>
    </div>

    <script src="config.js"></script>
    <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add frontend/
git commit -m "feat: add frontend layout - sidebar + chat UI

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 前端 JavaScript (`app.js`)

**Files:**
- Create: `frontend/app.js`

- [ ] **Step 1: 创建 app.js**

`/Users/260413a/Desktop/ai-project-assistant/frontend/app.js`
```javascript
(function () {
    "use strict";

    // ── 状态 ──────────────────────────────────────────
    let currentProject = null;       // 当前选中的项目名，null = 全部
    let messages = [];               // [{role, content, sources?}]
    const HISTORY_KEY = "ai_assistant_history";
    const MAX_HISTORY = 20;

    // ── DOM ───────────────────────────────────────────
    const projectList = document.getElementById("project-list");
    const historyList = document.getElementById("history-list");
    const chatArea = document.getElementById("chat-area");
    const examples = document.getElementById("examples");
    const emptyState = document.getElementById("empty-state");
    const questionInput = document.getElementById("question-input");
    const sendBtn = document.getElementById("send-btn");

    // ── 初始化 ────────────────────────────────────────
    function init() {
        loadProjects();
        loadHistory();
        bindEvents();

        // 检查后端是否配置
        if (BACKEND_URL === "http://localhost:5000") {
            console.warn("使用本地后端地址。部署前请修改 frontend/config.js 中的 BACKEND_URL。");
        }
    }

    // ── 加载项目列表 ──────────────────────────────────
    async function loadProjects() {
        try {
            const resp = await fetch(BACKEND_URL + "/api/health");
            const data = await resp.json();
            renderProjects(data.projects || []);
        } catch (e) {
            projectList.innerHTML = '<div style="padding:8px;color:#999;font-size:13px;">无法连接服务</div>';
        }
    }

    function renderProjects(projects) {
        let html = '<div class="project-item active" data-project="">全部项目</div>';
        for (const p of projects) {
            const shortName = p.replace("ai-gen-apps", "AI生成工具").replace("-v0.1", " v0.1").replace("-v", " v");
            html += `<div class="project-item" data-project="${escapeHtml(p)}">${escapeHtml(shortName)}</div>`;
        }
        projectList.innerHTML = html;
    }

    // ── 历史记录 ──────────────────────────────────────
    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            const history = raw ? JSON.parse(raw) : [];
            renderHistory(history);
        } catch (e) {
            renderHistory([]);
        }
    }

    function renderHistory(history) {
        if (history.length === 0) {
            historyList.innerHTML = '<div style="color:#bbb;font-size:12px;">暂无对话</div>';
            return;
        }
        let html = "";
        for (const h of history.slice(-MAX_HISTORY)) {
            html += `<div class="history-item" data-question="${escapeHtml(h)}">${escapeHtml(h)}</div>`;
        }
        historyList.innerHTML = html;
    }

    function saveToHistory(question) {
        try {
            let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
            history = history.filter(h => h !== question);
            history.push(question);
            if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            renderHistory(history);
        } catch (e) { /* localStorage 不可用 */ }
    }

    // ── 事件绑定 ──────────────────────────────────────
    function bindEvents() {
        // 项目切换
        projectList.addEventListener("click", function (e) {
            const item = e.target.closest(".project-item");
            if (!item) return;
            currentProject = item.dataset.project || null;
            document.querySelectorAll(".project-item").forEach(el => el.classList.remove("active"));
            item.classList.add("active");
        });

        // 历史点击
        historyList.addEventListener("click", function (e) {
            const item = e.target.closest(".history-item");
            if (!item) return;
            questionInput.value = item.dataset.question;
            sendMessage();
        });

        // 示例点击
        examples.addEventListener("click", function (e) {
            if (!e.target.classList.contains("example-btn")) return;
            questionInput.value = e.target.textContent;
            sendMessage();
        });

        // 发送按钮
        sendBtn.addEventListener("click", sendMessage);

        // 回车发送
        questionInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") sendMessage();
        });
    }

    // ── 发送消息 ──────────────────────────────────────
    async function sendMessage() {
        const question = questionInput.value.trim();
        if (!question) return;

        questionInput.value = "";
        setLoading(true);

        // 添加用户消息
        addMessage("user", question);

        try {
            const body = { question: question };
            if (currentProject) body.project = currentProject;

            const resp = await fetch(BACKEND_URL + "/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            const data = await resp.json();

            if (!resp.ok) {
                addMessage("assistant", data.message || "服务异常，请稍后重试。", []);
            } else {
                addMessage("assistant", data.answer, data.sources || []);
                saveToHistory(question);
            }
        } catch (e) {
            addMessage("assistant", "无法连接到服务。请检查网络连接，或确认后端服务正在运行。", []);
        }

        setLoading(false);
    }

    // ── 渲染消息 ──────────────────────────────────────
    function addMessage(role, content, sources) {
        messages.push({ role, content, sources: sources || [] });

        // 隐藏空状态和示例
        if (emptyState) emptyState.style.display = "none";
        if (examples) examples.style.display = messages.length > 0 ? "none" : "";

        const div = document.createElement("div");
        div.className = "message " + (role === "user" ? "user" : "assistant");

        let html = '<div class="bubble">' + formatContent(content);

        if (sources && sources.length > 0) {
            html += '<div style="margin-top:8px;font-size:11px;color:#999;">📎 参考来源:</div>';
            for (const s of sources) {
                const label = s.project + "/" + s.file + ":" + s.line;
                html += '<span class="source" title="' + escapeHtml(label) + '">📄 ' + escapeHtml(label) + '</span><br>';
            }
        }

        html += "</div>";
        div.innerHTML = html;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function formatContent(text) {
        // 简单 Markdown: **bold** → <strong>, 换行 → <br>
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    // ── 加载状态 ──────────────────────────────────────
    function setLoading(loading) {
        sendBtn.disabled = loading;
        sendBtn.textContent = loading ? "思考中..." : "发送";
        questionInput.disabled = loading;
    }

    // ── 工具函数 ──────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── 启动 ──────────────────────────────────────────
    init();
})();
```

- [ ] **Step 2: 本地预览前端**

在浏览器中打开 `file:///Users/260413a/Desktop/ai-project-assistant/frontend/index.html`，或启动后端后用 `python3 -m http.server 8080 -d frontend` 在 `frontend/` 目录启动静态服务。

Expected: 侧栏显示项目列表，输入问题后返回回答。

- [ ] **Step 3: Commit**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add frontend/app.js
git commit -m "feat: add chat JavaScript logic with API integration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 部署

**说明:** 将后端部署到 Render，前端部署到 GitHub Pages。

- [ ] **Step 1: 在 Render 创建 Web Service**

1. 访问 https://dashboard.render.com
2. 点击 "New +" → "Web Service"
3. 连接 GitHub 仓库（或直接上传）
4. 配置:
   - **Name**: `ai-project-assistant`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `./backend/start.sh`
   - **Plan**: Free
5. 添加环境变量:
   - `DEEPSEEK_API_KEY`: 你的 DeepSeek API Key
   - `KNOWLEDGE_DIR`: `backend/knowledge`

- [ ] **Step 2: 等待部署完成，验证后端**

```bash
curl https://YOUR-APP.onrender.com/api/health
```

Expected: `{"projects":["ai-gen-apps","ai-gen-apps-v0.1"],"status":"ok"}`

- [ ] **Step 3: 更新前端配置**

修改 `frontend/config.js`，将 `BACKEND_URL` 改为 Render 服务地址。

`/Users/260413a/Desktop/ai-project-assistant/frontend/config.js`
```javascript
const BACKEND_URL = "https://YOUR-APP.onrender.com";
```

- [ ] **Step 4: 部署前端到 GitHub Pages**

1. 在 GitHub 创建仓库（如 `ai-project-assistant`）
2. 推送代码
3. Settings → Pages → Source: "Deploy from a branch" → 选择 `main` → 文件夹 `/frontend`
4. 等待部署，获取 GitHub Pages URL

- [ ] **Step 5: 端到端验证**

在浏览器打开 GitHub Pages URL，输入问题验证完整流程。

- [ ] **Step 6: Commit 部署配置**

```bash
cd /Users/260413a/Desktop/ai-project-assistant
git add frontend/config.js
git commit -m "chore: update frontend config for production

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Render 休眠防护（可选但推荐）

**说明:** Render 免费层 15 分钟无请求会休眠，唤醒约 30 秒。可用免费监控服务定期 ping 保持活跃。

- [ ] **Step 1: 使用 UptimeRobot 或 cron-job.org**

1. 访问 https://cron-job.org（免费）
2. 创建定时任务，每 10 分钟 GET `https://YOUR-APP.onrender.com/api/health`
3. 保存

Expected: Render 实例保持活跃，不会休眠。
