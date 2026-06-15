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
    """搜索无关内容返回空列表或低相似度结果。"""
    results = search("asdfqwerzxcv12345无关内容")
    # BGE 归一化嵌入后任何文本都有向量，但结果仍应包含必要字段
    for r in results:
        assert "project" in r
        assert "file" in r
        assert "similarity" in r
    # 结果按相似度降序排列
    for i in range(len(results) - 1):
        assert results[i]["similarity"] >= results[i + 1]["similarity"]


def test_search_empty_query():
    """空查询返回空列表。"""
    results = search("")
    assert len(results) == 0
