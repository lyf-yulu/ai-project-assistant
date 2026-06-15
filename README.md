# AI 项目助手

基于项目文档的 RAG 问答应用。同事遇到问题时，用自然语言提问即可获得基于项目源码和文档的准确回答。

## 架构

```
你的电脑                    Render (免费)           GitHub Pages
┌──────────┐    git push   ┌──────────────┐   HTTP  ┌──────────────┐
│ indexer  │ ──────────→  │ Flask 后端    │ ←───── │ 静态前端      │
│ 本地索引  │              │ /api/chat    │        │ 聊天界面      │
└──────────┘              │ /api/health  │        └──────────────┘
                          │ FAISS 检索    │
                          │ DeepSeek API  │
                          └──────────────┘
```

## 本地开发

### 1. 构建知识库

```bash
cd indexer
pip install -r requirements.txt
HF_HOME=../.hf_cache python indexer.py
```

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
# 需要设置 DEEPSEEK_API_KEY 环境变量
HF_HOME=../.hf_cache python app.py
# 后端运行在 http://localhost:5000
```

### 3. 打开前端

用浏览器打开 `frontend/index.html` 即可使用。

### 4. 运行测试

```bash
# 不需要 API Key 的测试
HF_HOME=.hf_cache python -m pytest tests/ -v -k "not chat_normal and not chat_with_project and not chat_irrelevant"

# 全量测试（需要 DEEPSEEK_API_KEY）
DEEPSEEK_API_KEY=sk-xxx HF_HOME=.hf_cache python -m pytest tests/ -v
```

## 生产部署

### 第一步：部署后端到 Render

1. 将代码推送到 GitHub 仓库
2. 访问 [Render Dashboard](https://dashboard.render.com)
3. 点击 **New +** → **Web Service**
4. 连接 GitHub 仓库
5. Render 会自动识别 `render.yaml`，填充配置
6. 手动设置环境变量：点击 **Environment** → 添加 `DEEPSEEK_API_KEY`
7. 点击 **Create Web Service**，等待部署完成（约 3-5 分钟）
8. 记下服务地址，如 `https://ai-project-assistant.onrender.com`

### 第二步：部署前端到 GitHub Pages

1. GitHub 仓库 → **Settings** → **Pages**
2. Source 选择 **GitHub Actions**（`.github/workflows/deploy-frontend.yml` 已配置）
3. 修改 `frontend/config.js` 的 `BACKEND_URL` 为 Render 地址：
   ```javascript
   const BACKEND_URL = "https://ai-project-assistant.onrender.com";
   ```
4. 提交并推送，GitHub Actions 自动部署
5. 部署完成后，访问 `https://<你的用户名>.github.io/<仓库名>/`

### 第三步：防止 Render 休眠（推荐）

Render 免费层 15 分钟无请求会休眠，唤醒约 30 秒。

1. 访问 [cron-job.org](https://cron-job.org)（免费，无需注册）
2. 创建定时任务：
   - **URL**: `https://你的服务.onrender.com/api/health`
   - **间隔**: 每 10 分钟
3. 保存即可。这会定期 ping 后端，保持实例活跃。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | - | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | - | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | - | `deepseek-chat` | 对话模型 |
| `KNOWLEDGE_DIR` | - | `backend/knowledge` | 知识库路径 |
| `MAX_QUESTION_LENGTH` | - | `2000` | 问题最大字符数 |
| `HF_HOME` | - | 系统默认 | HuggingFace 缓存目录 |

## 添加新项目

在 `indexer/indexer.py` 的 `PROJECTS` dict 中添加一行：

```python
PROJECTS = {
    "ai-gen-apps": "~/Desktop/ai-generation-portable-apps",
    "ai-gen-apps-v0.1": "~/Desktop/ai-generation-portable-apps-v0.1",
    "新项目名": "/path/to/new/project",   # 添加这行
}
```

然后重新运行索引脚本并推送：

```bash
HF_HOME=.hf_cache python indexer/indexer.py
git add backend/knowledge/
git commit -m "chore: add 新项目 to knowledge base"
git push
```

Render 会自动检测到推送并重新部署。
