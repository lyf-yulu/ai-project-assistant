# AI 项目助手 — 设计文档

> 日期：2026-06-15
> 状态：已确认

## 1. 概述

### 1.1 目标

为没有代码基础的同事提供一个网页端 RAG 问答应用，能基于项目文档和源码准确回答故障排查类问题。比通用 AI 更准确，因为它知道项目上下文。

### 1.2 核心场景

同事在使用 AI Generation Portable Apps（Seedance / Nano Banana / Dreamina）时遇到问题，例如：
- "生成的时候提示 timeout 怎么解决？"
- "为什么上传的参考图不生效？"
- "Nano Banana 的参数 vary_seed 是什么意思？"

### 1.3 非目标

- 不提供项目开发/扩展指导（仅面向使用者的故障排查）
- 不替代实时日志分析
- 不处理 API Key 配置等私密信息

## 2. 架构

```
┌─────────────────────────────────────────────────┐
│                    你的电脑                        │
│  indexer.py                                       │
│  项目文件 → 分块 → DeepSeek Embedding → FAISS     │
│  输出: knowledge/{project}/index.faiss + chunks    │
│  提交到后端 Git 仓库                               │
└───────────────────────┬─────────────────────────┘
                        │ git push
┌───────────────────────▼─────────────────────────┐
│                   Render (免费层)                  │
│  app.py (Flask)                                   │
│  POST /api/chat  ┌──────────────────────┐        │
│  GET  /api/health│   FAISS 索引文件      │        │
│                  │   knowledge/          │        │
│                  │     ai-gen-apps/      │        │
│                  │     ai-gen-apps-v0.1/ │        │
│                  └──────────────────────┘        │
│       │                    │                      │
│       ▼                    ▼                      │
│  DeepSeek Embedding   DeepSeek Chat               │
└───────────────────────┬─────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────┐
│              GitHub Pages (静态前端)               │
│  单页 HTML + CSS + JS                             │
│  侧栏（项目列表 + 历史） + 聊天区                    │
└─────────────────────────────────────────────────┘
```

## 3. 组件详设

### 3.1 索引脚本 `indexer.py`

**运行方式**：手动执行，本地运行。

```python
PROJECTS = {
    "ai-gen-apps": "/Users/260413a/Desktop/ai-generation-portable-apps",
    "ai-gen-apps-v0.1": "/Users/260413a/Desktop/ai-generation-portable-apps-v0.1",
    # 新增项目在此添加
}
```

**分块规则**：
- `.md` 文件：按段落切分，保留标题层级作为上下文前缀
- `.py` 文件：按函数/类边界切分，保留 docstring
- `.html/.js/.css`：按语义块切分
- 每块约 500-1000 字符，附带元数据：`{project, file_path, line_start, line_end, heading}`
- 过滤：跳过 `outputs/`、`archives/`、`logs/`、`state/`、`__pycache__/`

**流程**：
1. 遍历 PROJECTS，收集所有目标文件
2. 文本分块
3. 批量调用 DeepSeek Embedding API（`deepseek-chat` 或专用 embedding 模型）
4. 写入 FAISS 索引 → `knowledge/{project}/index.faiss`
5. 写入元数据 → `knowledge/{project}/chunks.json`（每行一个 chunk 对象）
6. 提示用户提交 `knowledge/` 目录到后端仓库

### 3.2 后端 `app.py`

**框架**：Flask
**部署**：Render Free Tier
**依赖**：`flask`, `faiss-cpu`, `numpy`, `openai`（兼容 DeepSeek SDK）, `flask-cors`

**端点**：

| 端点 | 方法 | 请求体 | 响应体 |
|------|------|--------|--------|
| `/api/chat` | POST | `{"question": "...", "project": "ai-gen-apps"}` (project 选填) | `{"answer": "...", "sources": [{"project": "...", "file": "...", "line": 123, "snippet": "..."}]}` |
| `/api/health` | GET | - | `{"status": "ok", "projects": ["ai-gen-apps", "ai-gen-apps-v0.1"]}` |

**`/api/chat` 处理流程**：
1. 校验 `question` 非空，长度 ≤ 2000 字符
2. `question` → DeepSeek Embedding → 向量
3. 向量检索：若指定 `project` 则只搜该项目的 FAISS 索引，否则搜所有项目
4. 取 similarity ≥ 阈值（默认 0.5）的 Top-5 结果
5. 构造消息：

```
System: 你是 AI Generation Portable Apps 的技术支持助手。
你的知识来自项目文档和源码。只基于提供的文档片段回答问题。
如果文档中没有相关信息，请明确说"文档中没有涉及这个问题"，
不要猜测或编造。回答使用中文。

Context（来自项目文档）:
[chunk_1 内容]（来源: ai-gen-apps/seedance/app.py:150-165）
[chunk_2 内容]（来源: ai-gen-apps/API调用说明.md:30-45）
...

User: {question}
```

6. 调用 DeepSeek Chat API（`deepseek-chat` 模型，temperature=0.3）
7. 返回 `{answer, sources}`

**错误响应**：
```json
{"error": "question_required", "message": "请提供问题"}
{"error": "no_relevant_docs", "message": "没有找到相关文档"}
{"error": "api_error", "message": "服务暂时不可用，请稍后重试"}
```

**配置**（环境变量）：
- `DEEPSEEK_API_KEY`：DeepSeek API Key
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`
- `KNOWLEDGE_DIR`：知识库目录路径，默认 `./knowledge`
- `MAX_QUESTION_LENGTH`：默认 2000

### 3.3 前端

**部署**：GitHub Pages
**技术**：纯 HTML + CSS + JavaScript，无框架

**布局**：
- 左侧栏（240px，始终展开）：
  - 项目列表（单选，点击切换知识库范围）
  - 历史对话（本地 localStorage 存储，最近 20 条）
- 右侧聊天区：
  - 顶部标题 "🤖 AI 项目助手"
  - 消息列表（气泡式，用户靠右蓝色，助手靠左灰色）
  - 示例问题（首次使用时显示，点击可快速提问）
  - 输入框 + 发送按钮（底部固定）
  - 回答带来源引用标记，点击跳转 GitHub 源码

**状态**：
- 加载中：发送按钮显示加载动画
- 错误：红色提示条
- 空结果：显示"没有找到相关文档"

**API 调用**：
```javascript
POST {BACKEND_URL}/api/chat
Headers: { "Content-Type": "application/json" }
Body: { "question": "用户输入", "project": "当前选中的项目" }
```

`BACKEND_URL` 通过配置文件设置，指向 Render 服务地址。

## 4. 扩展接口

### 4.1 添加新项目

在索引脚本 `PROJECTS` dict 中添加一项，运行索引，提交新的 `knowledge/{project}/` 目录。后端自动识别 `knowledge/` 下的新子目录，无需改代码。

### 4.2 知识库目录结构

```
knowledge/
├── ai-gen-apps/
│   ├── index.faiss
│   └── chunks.json
├── ai-gen-apps-v0.1/
│   ├── index.faiss
│   └── chunks.json
└── {未来项目}/
    ├── index.faiss
    └── chunks.json
```

### 4.3 未来可扩展方向（不在此次范围内）

- 自动索引（webhook 触发或定时任务）
- 多语言支持
- 用户反馈机制（踩/赞）
- 简单的访问密码保护

## 5. 约束与风险

| 约束 | 说明 |
|------|------|
| Render 免费层 | 每月 750 小时，15 分钟无请求会休眠，唤醒约 30 秒 |
| DeepSeek API 费用 | Chat 按量计费 |
| Embedding 方案 | 优先使用 DeepSeek Embedding API；如不可用，改用本地轻量模型（sentence-transformers / BGE-small）生成向量，索引脚本和检索端共用同一模型。不增加外部 API 依赖 |
| FAISS 内存 | 知识库较小时（<1000 chunks）内存影响可忽略 |
| 手动索引 | 项目更新后需手动重跑 indexer.py 并部署 |

## 6. 文件结构

```
ai-project-assistant/
├── indexer/
│   ├── indexer.py          # 索引脚本
│   └── requirements.txt    # faiss-cpu, numpy, openai
├── backend/
│   ├── app.py              # Flask 后端
│   ├── requirements.txt    # flask, faiss-cpu, numpy, openai, flask-cors
│   ├── knowledge/           # 知识库（git 跟踪）
│   │   ├── ai-gen-apps/
│   │   └── ai-gen-apps-v0.1/
│   └── start.sh            # Render 启动脚本
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── config.js            # BACKEND_URL 配置
└── README.md
```
