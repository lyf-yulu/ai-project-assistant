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
            assert ch["file"] == tmp_path
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
