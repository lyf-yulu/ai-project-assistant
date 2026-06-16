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
             ".git", "node_modules", ".superpowers", "data", "release",
             ".venv", "venv", "env", ".env", "__MACOSX"}

SKIP_FILES = {"config.json", "config_github.json", ".DS_Store"}

CHUNK_SIZE = 800   # 每块最大字符数


# ── 文件收集 ──────────────────────────────────────────
def collect_files(project_dir: str) -> list[Path]:
    """递归收集 project_dir 下所有应索引的文件路径 (os.walk + topdown pruning)."""
    root = Path(project_dir)
    files = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in EXTENSIONS:
                continue
            if fname in SKIP_FILES:
                continue
            files.append(fpath)
    return files


# ── 文本分块 ──────────────────────────────────────────
def split_by_paragraphs(text: str, file_path: Path) -> list[dict]:
    """按段落分块（通用策略，适用于 .md / .txt），基于偏移量计算行号。"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    current_start_offset = 0

    # Pre-compute line boundaries for offset→line mapping
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def offset_to_line(offset):
        # Binary search the line number for an offset
        lo, hi = 0, len(line_starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= offset:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > CHUNK_SIZE and current:
            line_start = offset_to_line(current_start_offset)
            line_end = offset_to_line(current_start_offset + len(current))
            chunks.append({
                "content": current.strip(),
                "file": str(file_path),
                "line_start": line_start,
                "line_end": line_end,
            })
            current_start_offset = text.find(para)
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para
                current_start_offset = text.find(para)

    if current.strip():
        line_start = offset_to_line(current_start_offset)
        line_end = offset_to_line(current_start_offset + len(current))
        chunks.append({
            "content": current.strip(),
            "file": str(file_path),
            "line_start": line_start,
            "line_end": line_end,
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
            if len(chunk_text) > 30:  # 过滤过短的块
                chunks.append({
                    "content": chunk_text,
                    "file": str(file_path),
                    "line_start": current_start,
                    "line_end": i - 1,
                })
                current = [line]
                current_start = i
            else:
                # 短前导（decorator/注释）合并到新函数
                current.append(line)
        else:
            current.append(line)

        # 简单跟踪 docstring（处理单行 docstring，如 """text"""）
        if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
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
            ch["project_root"] = project_dir
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
