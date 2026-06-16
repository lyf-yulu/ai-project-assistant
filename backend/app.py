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
    from pathlib import Path
    import os as _os
    kd = Path(_os.environ.get("KNOWLEDGE_DIR", "knowledge"))
    info = {"kd": str(kd), "abs": str(kd.absolute()), "exists": kd.exists()}
    if kd.exists():
        info["subdirs"] = [{"name": d.name, "has_faiss": (d / "index.faiss").exists(), "files": [f.name for f in d.iterdir()]} for d in kd.iterdir() if d.is_dir()]
    return jsonify({
        "status": "ok",
        "projects": list_projects(),
        "_debug": info,
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
