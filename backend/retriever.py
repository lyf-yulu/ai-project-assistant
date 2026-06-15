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
    if not query or not query.strip():
        return []

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
        project_root = meta.get("project_root")
        if not project_root:
            return "(项目路径未记录)"

        full_path = Path(project_root) / meta["file"]
        if not full_path.exists():
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
