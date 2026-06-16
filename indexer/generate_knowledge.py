#!/usr/bin/env python3
"""使用 DeepSeek 生成项目全景知识文档。

读取项目关键文件，发送给 DeepSeek 分析，生成结构化 JSON 知识文档。
输出到 backend/knowledge/<project>/panorama.json。

用法:
    python indexer/generate_knowledge.py                  # 生成所有项目
    python indexer/generate_knowledge.py --project ai-gen-apps  # 单个项目
    python indexer/generate_knowledge.py --force           # 忽略 hash 检查
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

PROJECTS = {
    "ai-gen-apps": str(Path.home() / "Desktop/ai-generation-portable-apps"),
    "ai-gen-apps-v0.1": str(Path.home() / "Desktop/ai-generation-portable-apps-v0.1"),
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "backend/knowledge"

# 要发送给 DeepSeek 分析的关键文件模式
KEY_FILES = ["README.md", "CLAUDE.md", "API调用说明.md"]

KEY_SUBDIRS = {
    "portal": ["app.py"],
    "seedance": ["app.py", "providers.json"],
    "nano-banana": ["app.py", "providers.json"],
    "dreamina": ["app.py", "config.json"],
}

MAX_FILE_BYTES = 80_000  # 单个文件最大读取字节

SKIP_DIRS = {"outputs", "archives", "logs", "state", "__pycache__",
             ".git", "node_modules", ".superpowers", "data", "release",
             ".venv", "venv", "env", ".env", "__MACOSX", "static"}

# ── 生成 Prompt ───────────────────────────────────────
GENERATION_PROMPT = """你是一个软件架构分析专家。请仔细阅读下面提供的项目源文件，生成一份完整的项目知识文档。

输出必须是合法的 JSON 对象，严格遵循以下结构（不要输出任何 JSON 以外的内容）：

{
  "project_overview": "项目概述：这个项目是什么，给谁用，包含哪些应用。2-3段中文。",
  "architecture": {
    "topology": "系统拓扑：各应用如何部署、端口分配、代理关系",
    "data_flow": "数据流：用户请求如何流转，从浏览器到各子应用",
    "patterns": ["关键设计模式1", "关键设计模式2"]
  },
  "sub_apps": {
    "应用名": {
      "purpose": "功能说明，一句话",
      "port": 端口号或null,
      "entry_point": "启动命令",
      "key_endpoints": [
        {"path": "/api/xxx", "method": "POST", "purpose": "用途"}
      ],
      "key_patterns": ["模式1", "模式2"]
    }
  },
  "cross_cutting": {
    "job_system": "任务系统说明：如何创建、轮询、存储任务",
    "provider_system": "Provider 配置机制：providers.json 结构、切换逻辑",
    "error_handling": "错误处理模式",
    "state_management": "状态管理：文件约定、原子写入"
  },
  "known_issues": [
    {"symptom": "问题现象", "cause": "根因（基于代码逻辑）", "solution": "解决步骤"}
  ],
  "troubleshooting_guide": "常见排错指南，Q&A 格式，中文",
  "file_conventions": {
    "directories": {"目录名": "用途"},
    "key_files": {"文件名": "用途"}
  }
}

## 关键要求

1. **严格基于源文件**：所有内容必须来自提供的源文件，不要编造任何信息
2. **推断标注**：如果某个结论是从代码逻辑推断的（而非文档明确写的），在文本前加 `[推测]`
3. **排错指南要实用**：基于代码中的异常处理、超时逻辑、配置校验等实际实现来写
4. **known_issues 要具体**：基于代码中`try/except`、`timeout`、错误码等实际存在的逻辑
5. **中文撰写**：所有文本内容用中文
6. **没有信息就留空**：某个 section 在源文件中没有依据时，用空字符串/null/空数组"""


def get_git_head(project_dir: str) -> str:
    """获取项目当前 git HEAD hash。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def collect_files(project_dir: str) -> dict[str, str]:
    """收集项目关键文件内容。返回 {relative_path: content}。"""
    root = Path(project_dir)
    files = {}

    # 根目录关键文件
    for fname in KEY_FILES:
        fpath = root / fname
        if fpath.exists():
            files[fname] = fpath.read_text(encoding="utf-8")[:MAX_FILE_BYTES]

    # 子目录关键文件
    for subdir, subfiles in KEY_SUBDIRS.items():
        subdir_path = root / subdir
        if not subdir_path.exists():
            continue
        for sf in subfiles:
            fpath = subdir_path / sf
            if fpath.exists():
                rel = str(fpath.relative_to(root))
                files[rel] = fpath.read_text(encoding="utf-8")[:MAX_FILE_BYTES]

    return files


def build_messages(project_name: str, files: dict[str, str]) -> list[dict]:
    """构建发送给 DeepSeek 的消息列表。"""
    # 拼接源文件
    sources_text_parts = []
    for path, content in files.items():
        sources_text_parts.append(f"### 文件: {path}\n```\n{content}\n```")

    sources_text = "\n\n".join(sources_text_parts)

    user_message = f"""请分析以下项目 "{project_name}" 的源文件，生成项目知识文档。

{sources_text}

请按照要求输出 JSON 格式的知识文档。"""

    return [
        {"role": "system", "content": GENERATION_PROMPT},
        {"role": "user", "content": user_message},
    ]


def call_deepseek(messages: list[dict]) -> str:
    """调用 DeepSeek API，返回响应文本。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.2,
        messages=messages,
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict:
    """从 DeepSeek 响应中提取 JSON，支持 markdown 代码块包裹。"""
    text = text.strip()

    # 移除可能的 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```json 或 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 去掉末行 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


def validate_panorama(data: dict, project_name: str) -> list[str]:
    """验证生成的知识文档，返回警告列表。"""
    warnings = []

    # 必需字段检查
    required = ["project_overview", "architecture", "sub_apps",
                "cross_cutting", "known_issues", "troubleshooting_guide"]
    for key in required:
        if key not in data:
            warnings.append(f"缺少必需字段: {key}")

    # 内容长度检查
    overview = data.get("project_overview", "")
    if len(overview) < 50:
        warnings.append(f"project_overview 过短 ({len(overview)} 字符)")

    ts = data.get("troubleshooting_guide", "")
    if len(ts) < 100:
        warnings.append("troubleshooting_guide 内容偏少，可能覆盖不全")

    # 可疑短语检查
    text = json.dumps(data, ensure_ascii=False)
    suspicious = ["根据您提供的代码", "Based on the provided", "I'll analyze", "Here is"]
    for s in suspicious:
        if s in text:
            warnings.append(f"包含元评论文本: '{s}'")

    return warnings


def needs_regeneration(project_dir: str, panorama_path: Path) -> bool:
    """检查是否需要重新生成。"""
    if not panorama_path.exists():
        return True
    try:
        data = json.loads(panorama_path.read_text(encoding="utf-8"))
        stored_hash = data.get("meta", {}).get("git_hash", "")
        if not stored_hash:
            return True
        current_hash = get_git_head(project_dir)
        return stored_hash != current_hash
    except (json.JSONDecodeError, OSError):
        return True


def generate_panorama(project_name: str, project_dir: str, force: bool = False) -> dict | None:
    """为一个项目生成全景知识文档。"""
    print(f"\n{'='*60}")
    print(f"生成知识文档: {project_name}")
    print(f"项目路径: {project_dir}")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR / project_name
    out_path = out_dir / "panorama.json"

    # 检查是否需要生成
    if not force and not needs_regeneration(project_dir, out_path):
        print(f"  → 跳过（源码未变更，使用缓存）")
        return json.loads(out_path.read_text(encoding="utf-8"))

    # 收集文件
    files = collect_files(project_dir)
    if not files:
        print("  错误: 未找到任何关键文件")
        return None
    print(f"  收集到 {len(files)} 个关键文件")

    # 估算大小
    total_chars = sum(len(c) for c in files.values())
    print(f"  总字符数: {total_chars:,} (~{total_chars // 3} tokens)")

    # 调用 DeepSeek
    print("  正在调用 DeepSeek 分析...")
    messages = build_messages(project_name, files)
    try:
        response = call_deepseek(messages)
    except Exception as e:
        print(f"  错误: DeepSeek API 调用失败: {e}")
        return None

    # 解析 JSON
    try:
        data = extract_json(response)
    except json.JSONDecodeError as e:
        print(f"  错误: JSON 解析失败: {e}")
        print(f"  原始响应前 500 字符: {response[:500]}")
        return None

    # 验证
    warnings = validate_panorama(data, project_name)
    for w in warnings:
        print(f"  ⚠ {w}")

    # 添加元数据
    data["meta"] = {
        "project": project_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": get_git_head(project_dir),
        "model": DEEPSEEK_MODEL,
        "file_count": len(files),
    }

    # 写入
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 已写入: {out_path}")
    print(f"  文档大小: {out_path.stat().st_size:,} bytes")

    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成项目全景知识文档")
    parser.add_argument("--project", help="指定项目名称（默认全部）")
    parser.add_argument("--force", action="store_true", help="忽略 hash 检查强制重新生成")
    args = parser.parse_args()

    if args.project:
        if args.project not in PROJECTS:
            print(f"错误: 未找到项目 '{args.project}'，可用: {list(PROJECTS.keys())}")
            sys.exit(1)
        projects = {args.project: PROJECTS[args.project]}
    else:
        projects = PROJECTS

    for name, path in projects.items():
        if not Path(path).exists():
            print(f"警告: 项目路径不存在，跳过: {name} ({path})")
            continue
        generate_panorama(name, path, force=args.force)

    print(f"\n完成！知识文档位于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
