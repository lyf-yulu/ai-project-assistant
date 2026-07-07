"""AI 项目助手 - Flask 后端。

部署: Render Free Tier
启动: gunicorn app:app --bind 0.0.0.0:$PORT
版本: 2026-07-07 - 补充 JSON 解析报错的供应商配置条目
"""
import json
import os
import logging
from pathlib import Path

from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

from retriever import search, list_projects, KNOWLEDGE_DIR

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

## 你的知识来源

你有两个知识来源，按优先级排列：
1. **项目全景知识**（下方提供）：包含项目概述、架构、子应用说明、已知问题、排错指南。这是你的主要知识来源，用于回答概念类、架构类、排错类问题。
2. **参考文档片段**（用户消息中提供）：从项目源码和文档中检索的原文片段。用于补充具体代码细节。

## 回答规则

1. 优先使用项目全景知识回答问题。结合参考文档片段中的代码细节补充。
2. 如果全景知识和参考文档中都没有相关信息，说"文档中没有涉及这个问题"，不要编造。
3. 回答要简洁、具体、可操作，像技术支持同事一样。
4. 在末尾标注信息来源（文件名或章节）。
5. 不要提及"全景知识"、"文档片段"、"上下文"这些内部术语。"""

# ── 全景知识加载 ──────────────────────────────────────
_panorama_cache: dict[str, str] = {}


def _format_panorama(data: dict) -> str:
    """将全景知识 JSON 转为紧凑文本。"""
    parts = []

    overview = data.get("project_overview", "")
    if overview:
        parts.append(f"## 项目概述\n{overview}")

    arch = data.get("architecture", {})
    if arch.get("topology"):
        parts.append(f"## 系统架构\n{arch['topology']}")
    if arch.get("data_flow"):
        parts.append(f"数据流: {arch['data_flow']}")

    sub_apps = data.get("sub_apps", {})
    if sub_apps:
        lines = []
        for name, info in sub_apps.items():
            lines.append(f"\n### {name}")
            if info.get("purpose"):
                lines.append(f"功能: {info['purpose']}")
            if info.get("port"):
                lines.append(f"端口: {info['port']}")
            if info.get("entry_point"):
                lines.append(f"启动: {info['entry_point']}")
            endpoints = info.get("key_endpoints", [])
            if endpoints:
                ep_lines = [f"  {e['method']} {e['path']} - {e.get('purpose', '')}" for e in endpoints]
                lines.append("API端点:\n" + "\n".join(ep_lines))
        parts.append("## 子应用" + "\n".join(lines))

    known = data.get("known_issues", [])
    if known:
        issues = []
        for i in known:
            issues.append(f"- {i.get('symptom', '')}\n  原因: {i.get('cause', '')}\n  解决: {i.get('solution', '')}")
        parts.append("## 已知问题\n" + "\n".join(issues))

    ts = data.get("troubleshooting_guide", "")
    if ts:
        parts.append(f"## 排错指南\n{ts}")

    return "\n\n".join(parts)


def load_panorama(project_name: str) -> str:
    """加载项目全景知识文档，返回格式化文本。"""
    if project_name in _panorama_cache:
        return _panorama_cache[project_name]

    path = KNOWLEDGE_DIR / project_name / "panorama.json"
    if not path.exists():
        return ""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = _format_panorama(data)
        _panorama_cache[project_name] = text
        return text
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"加载全景知识失败 ({project_name}): {e}")
        return ""


def build_system_prompt(project: Optional[str]) -> str:
    """构建带全景知识的系统提示。"""
    prompt = SYSTEM_PROMPT

    if project:
        panorama = load_panorama(project)
    else:
        # 合并所有项目全景知识
        parts = []
        for proj_name in list_projects():
            p = load_panorama(proj_name)
            if p:
                parts.append(f"--- {proj_name} ---\n{p}")
        panorama = "\n\n".join(parts)

    if panorama:
        prompt += f"\n\n## 项目全景知识\n{panorama}"

    return prompt


# ── 路由 ──────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """健康检查。"""
    projects = list_projects()
    # 检查全景知识加载状态
    panorama_status = {}
    for p in projects:
        path = KNOWLEDGE_DIR / p / "panorama.json"
        panorama_status[p] = path.exists()
    return jsonify({
        "status": "ok",
        "projects": projects,
        "panorama": panorama_status,
    })


@app.route("/api/debug/panorama", methods=["GET"])
def debug_panorama():
    """诊断端点：返回加载的 panorama 内容摘要。"""
    result = {}
    for p in list_projects():
        path = KNOWLEDGE_DIR / p / "panorama.json"
        if not path.exists():
            result[p] = {"error": "file missing"}
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            issues = data.get("known_issues", [])
            result[p] = {
                "file_size": path.stat().st_size,
                "issue_count": len(issues),
                "symptoms": [i.get("symptom", "") for i in issues],
                "meta": data.get("meta", {}),
            }
        except Exception as e:
            result[p] = {"error": str(e)}
    return jsonify(result)


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
        return jsonify({"error": "search_error", "message": f"检索服务异常: {e}"}), 500

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

    # 构建带全景知识的系统提示
    system_prompt = build_system_prompt(project)

    # 调用 DeepSeek
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
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
