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
