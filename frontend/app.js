(function () {
    "use strict";

    // ── 状态 ──────────────────────────────────────────
    let currentProject = null;       // 当前选中的项目名，null = 全部
    let messages = [];               // [{role, content, sources?}]
    const HISTORY_KEY = "ai_assistant_history";
    const MAX_HISTORY = 20;

    // ── DOM ───────────────────────────────────────────
    const projectList = document.getElementById("project-list");
    const historyList = document.getElementById("history-list");
    const chatArea = document.getElementById("chat-area");
    const examples = document.getElementById("examples");
    const emptyState = document.getElementById("empty-state");
    const questionInput = document.getElementById("question-input");
    const sendBtn = document.getElementById("send-btn");

    // ── 初始化 ────────────────────────────────────────
    function init() {
        loadProjects();
        loadHistory();
        bindEvents();

        if (BACKEND_URL === "http://localhost:5000") {
            console.warn("使用本地后端地址。部署前请修改 frontend/config.js 中的 BACKEND_URL。");
        }
    }

    // ── 加载项目列表 ──────────────────────────────────
    async function loadProjects() {
        try {
            const resp = await fetch(BACKEND_URL + "/api/health");
            const data = await resp.json();
            renderProjects(data.projects || []);
        } catch (e) {
            projectList.innerHTML = '<div style="padding:8px;color:#999;font-size:13px;">无法连接服务</div>';
        }
    }

    function renderProjects(projects) {
        let html = '<div class="project-item active" data-project="">全部项目</div>';
        for (const p of projects) {
            const shortName = p.replace("ai-gen-apps", "AI生成工具").replace("-v0.1", " v0.1").replace("-v", " v");
            html += `<div class="project-item" data-project="${escapeHtml(p)}">${escapeHtml(shortName)}</div>`;
        }
        projectList.innerHTML = html;
    }

    // ── 历史记录 ──────────────────────────────────────
    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            const history = raw ? JSON.parse(raw) : [];
            renderHistory(history);
        } catch (e) {
            renderHistory([]);
        }
    }

    function renderHistory(history) {
        if (history.length === 0) {
            historyList.innerHTML = '<div style="color:#bbb;font-size:12px;">暂无对话</div>';
            return;
        }
        let html = "";
        for (const h of history.slice(-MAX_HISTORY)) {
            html += `<div class="history-item" data-question="${escapeHtml(h)}">${escapeHtml(h)}</div>`;
        }
        historyList.innerHTML = html;
    }

    function saveToHistory(question) {
        try {
            let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
            history = history.filter(h => h !== question);
            history.push(question);
            if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            renderHistory(history);
        } catch (e) { /* localStorage 不可用 */ }
    }

    // ── 事件绑定 ──────────────────────────────────────
    function bindEvents() {
        projectList.addEventListener("click", function (e) {
            const item = e.target.closest(".project-item");
            if (!item) return;
            currentProject = item.dataset.project || null;
            document.querySelectorAll(".project-item").forEach(el => el.classList.remove("active"));
            item.classList.add("active");
        });

        historyList.addEventListener("click", function (e) {
            const item = e.target.closest(".history-item");
            if (!item) return;
            questionInput.value = item.dataset.question;
            sendMessage();
        });

        examples.addEventListener("click", function (e) {
            if (!e.target.classList.contains("example-btn")) return;
            questionInput.value = e.target.textContent;
            sendMessage();
        });

        sendBtn.addEventListener("click", sendMessage);

        questionInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") sendMessage();
        });
    }

    // ── 发送消息 ──────────────────────────────────────
    async function sendMessage() {
        const question = questionInput.value.trim();
        if (!question) return;

        questionInput.value = "";
        setLoading(true);

        addMessage("user", question);

        try {
            const body = { question: question };
            if (currentProject) body.project = currentProject;

            const resp = await fetch(BACKEND_URL + "/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            const data = await resp.json();

            if (!resp.ok) {
                addMessage("assistant", data.message || "服务异常，请稍后重试。", []);
            } else {
                addMessage("assistant", data.answer, data.sources || []);
                saveToHistory(question);
            }
        } catch (e) {
            addMessage("assistant", "无法连接到服务。请检查网络连接，或确认后端服务正在运行。", []);
        }

        setLoading(false);
    }

    // ── 渲染消息 ──────────────────────────────────────
    function addMessage(role, content, sources) {
        messages.push({ role, content, sources: sources || [] });

        if (emptyState) emptyState.style.display = "none";
        if (examples) examples.style.display = messages.length > 0 ? "none" : "";

        const div = document.createElement("div");
        div.className = "message " + (role === "user" ? "user" : "assistant");

        let html = '<div class="bubble">' + formatContent(content);

        if (sources && sources.length > 0) {
            html += '<div style="margin-top:8px;font-size:11px;color:#999;">📎 参考来源:</div>';
            for (const s of sources) {
                const label = s.project + "/" + s.file + ":" + s.line;
                html += '<span class="source" title="' + escapeHtml(label) + '">📄 ' + escapeHtml(label) + '</span><br>';
            }
        }

        html += "</div>";
        div.innerHTML = html;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function formatContent(text) {
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    // ── 加载状态 ──────────────────────────────────────
    function setLoading(loading) {
        sendBtn.disabled = loading;
        sendBtn.textContent = loading ? "思考中..." : "发送";
        questionInput.disabled = loading;
    }

    // ── 工具函数 ──────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── 启动 ──────────────────────────────────────────
    init();
})();
