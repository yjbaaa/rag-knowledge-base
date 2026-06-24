"""
RAG Knowledge Base — Streamlit Frontend
Swiss Style · Minimalism · Streaming Chat · Source Citations · Per-User API Key
"""

import sys
import json
from pathlib import Path

import streamlit as st
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import API_PORT

API_BASE = f"http://127.0.0.1:{API_PORT}"

# ═══════════════════════════════════════════
#  Page config
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="RAG 知识库 · 智能问答",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "RAG 知识库 — 企业文档智能问答系统",
    },
)

# ═══════════════════════════════════════════
#  Design System CSS (Swiss Style)
# ═══════════════════════════════════════════

st.markdown(
    """<style>
/* Google Fonts: Lexend + Source Sans 3 */
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, .stApp, .stMarkdown, .stText, .stChatMessage {
    font-family: 'Source Sans 3', -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}
h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm {
    font-family: 'Lexend', 'Source Sans 3', -apple-system, 'PingFang SC', sans-serif !important;
    font-weight: 500;
    letter-spacing: -0.01em;
}

/* ── Colors ── */
:root {
    --primary: #475569;
    --secondary: #64748B;
    --cta: #2563EB;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --text: #1E293B;
    --text-muted: #64748B;
    --border: #E2E8F0;
    --radius: 8px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .st-emotion-cache-1gwvycy {
    padding: 1.5rem 1.2rem;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #1E293B !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* ── Main content ── */
.stApp {
    background: #F8FAFC;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 900px;
}

/* ── Chat bubbles ── */
.stChatMessage {
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid #E2E8F0;
}
.stChatMessage [data-testid="chatAvatarIcon-user"] {
    background: #2563EB !important;
}
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
    background: #475569 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
    transition: all 0.2s ease;
    border: 1px solid #E2E8F0;
    font-family: 'Source Sans 3', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
.stButton > button:hover {
    border-color: #CBD5E1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #2563EB;
    color: white;
    border-color: #2563EB;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    border-radius: 6px;
    border-color: #E2E8F0;
    font-family: 'Source Sans 3', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 0.88rem;
    padding: 8px 12px;
}
.stTextInput > div > div > input:focus {
    border-color: #2563EB;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.12);
}

/* ── File uploader ── */
.stFileUploader {
    border-radius: 6px;
}
.stFileUploader > section {
    border: 1.5px dashed #CBD5E1 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
.stFileUploader > section:hover {
    border-color: #94A3B8 !important;
    background: #F8FAFC;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.82rem;
    color: #475569;
    border-radius: 6px;
}

/* ── Divider ── */
hr {
    border-color: #E2E8F0 !important;
    margin: 1rem 0;
}

/* ── Empty/placeholder ── */
.stAlert {
    border-radius: 6px;
    font-size: 0.85rem;
}

/* ── Chat input ── */
.stChatInput textarea {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    font-family: 'Source Sans 3', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.12) !important;
}

/* ── Status badge ── */
.api-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.api-badge.online {
    background: #F0FDF4;
    color: #15803D;
    border: 1px solid #BBF7D0;
}
.api-badge.offline {
    background: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FECACA;
}

/* ── Chat streaming cursor ── */
@keyframes blink {
    50% { opacity: 0; }
}

/* ── Source cards ── */
.source-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #475569;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.source-card .src-header {
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 4px;
    font-size: 0.76rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.source-card .src-body {
    color: #475569;
    line-height: 1.6;
    font-size: 0.78rem;
}

/* ── API key masked input ── */
.api-key-field input {
    -webkit-text-security: disc;
    font-family: monospace !important;
}

/* ── Welcome card ── */
.welcome-card {
    text-align: center;
    padding: 48px 24px;
    color: #64748B;
}
.welcome-card svg {
    margin-bottom: 16px;
}
.welcome-card h2 {
    font-size: 1.15rem;
    color: #475569;
    margin-bottom: 8px;
}
.welcome-card p {
    font-size: 0.85rem;
    line-height: 1.7;
    max-width: 400px;
    margin: 0 auto;
}

/* ── Sidebar section ── */
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 8px;
    margin-top: 16px;
}
</style>""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════
#  Session state init
# ═══════════════════════════════════════════

DEFAULTS = {
    "messages": [],
    "session_id": "default",
    "session_label": "Default",
    "uploaded_files": [],
    "api_online": None,
    "api_ready": False,
    "api_uptime": 0,
    "indexed_sources": [],
    # Per-user LLM config — this is what the user enters
    "user_api_key": "",
    "user_api_base": "https://api.deepseek.com",
    "user_api_model": "deepseek-chat",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════
#  API helpers
# ═══════════════════════════════════════════

def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=2.5)
        if r.status_code == 200:
            return {"online": True, **r.json()}
    except Exception:
        pass
    return {"online": False}

def upload_files(files) -> dict:
    payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_BASE}/api/upload", files=payload, timeout=120)
    if r.status_code == 200:
        return r.json()
    raise Exception(r.text)

def get_sessions() -> list:
    try:
        r = requests.get(f"{API_BASE}/api/sessions", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []

def delete_session(sid: str) -> dict:
    r = requests.delete(f"{API_BASE}/api/sessions/{sid}", timeout=5)
    return r.json()

# ═══════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════

with st.sidebar:
    # ── Brand header ──
    st.markdown(
        """<div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><line x1="8" y1="7" x2="16" y2="7"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
<h2 style="margin:0;font-size:1.05rem;font-weight:600;color:#1E293B;">RAG 知识库</h2>
</div>""",
        unsafe_allow_html=True,
    )

    # ── API status ──
    if st.button("↻ 检查服务状态", use_container_width=True):
        with st.spinner("检测中..."):
            health = check_health()
            st.session_state.api_online = health.get("online", False)
            st.session_state.api_ready = health.get("ready", False)
            st.session_state.api_uptime = health.get("uptime_seconds", 0)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.api_online and st.session_state.api_ready:
            st.markdown('<span class="api-badge online">● 服务就绪</span>', unsafe_allow_html=True)
        elif st.session_state.api_online:
            st.markdown('<span class="api-badge online">● 预热中</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="api-badge offline">○ 未连接</span>', unsafe_allow_html=True)

    st.divider()

    # ── LLM 配置 (User's own API key) ──
    st.markdown('<p class="sidebar-section-title">⚙ 大模型配置</p>', unsafe_allow_html=True)
    st.caption("请填入您自己的 API Key 与服务商信息")

    user_api_key = st.text_input(
        "API Key",
        value=st.session_state.user_api_key,
        type="password",
        placeholder="sk-...",
        help="您的 API Key 仅保存在当前会话中，不会被上传到服务器。",
        key="sidebar_api_key",
    )
    user_api_base = st.text_input(
        "API Base URL",
        value=st.session_state.user_api_base,
        placeholder="https://api.deepseek.com",
        help="API 服务商地址。默认 DeepSeek。",
    )
    user_api_model = st.text_input(
        "模型名称",
        value=st.session_state.user_api_model,
        placeholder="deepseek-chat",
        help="模型标识，如 deepseek-chat、gpt-4o-mini 等。",
    )

    # Update session state
    st.session_state.user_api_key = user_api_key
    st.session_state.user_api_base = user_api_base
    st.session_state.user_api_model = user_api_model

    if not user_api_key:
        st.warning("⚠ 请先填入 API Key 才能提问", icon="⚠")

    st.divider()

    # ── 文档上传 ──
    st.markdown('<p class="sidebar-section-title">📄 文档上传</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "支持 PDF / Word / Markdown / TXT",
        type=["pdf", "docx", "doc", "md", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded and st.button("⟳ 上传并索引", use_container_width=True, type="primary"):
        if not st.session_state.api_online:
            st.error("API 服务未连接，请先启动后端: python run.py api")
        else:
            with st.spinner("正在解析文档并构建向量索引..."):
                try:
                    result = upload_files(uploaded)
                    st.success(f"已索引 {result.get('chunk_count', 0)} 个文本块")
                    st.session_state.uploaded_files = result.get("files", [])
                    # Refresh indexed sources
                    h = check_health()
                    st.session_state.api_online = h.get("online", False)
                except Exception as e:
                    st.error(f"上传失败: {e}")

    # ── 会话管理 ──
    st.divider()
    st.markdown('<p class="sidebar-section-title">💬 会话管理</p>', unsafe_allow_html=True)

    sessions = get_sessions()
    if sessions:
        session_options = {s["session_id"]: f"{s['session_id'][:8]} ({s['turn_count']} 轮)" for s in sessions}
        session_options["default"] = "默认会话"
        selected_session = st.selectbox(
            "切换会话",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=list(session_options.keys()).index(st.session_state.session_id) if st.session_state.session_id in session_options else 0,
            label_visibility="collapsed",
        )
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.session_state.messages = []
            st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = "default"
            st.rerun()
    with col_b:
        if st.button("🔄 新建会话", use_container_width=True):
            st.session_state.session_id = "default"
            st.session_state.messages = []
            st.rerun()

# ═══════════════════════════════════════════
#  Main content area
# ═══════════════════════════════════════════

# Remove the old banner — the sidebar already has the brand

# ── Render chat history ──
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        # Welcome state
        st.markdown(
            """<div class="welcome-card">
<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 0 0-16 0"/></svg>
<h2>欢迎使用 RAG 知识库</h2>
<p>在左侧填入您的 API Key，上传文档，然后开始提问。<br>所有回答都将附带引用来源。</p>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 引用来源", expanded=False):
                        for i, src in enumerate(msg["sources"]):
                            filename = src.get("filename", "未知文件")
                            page = src.get("page")
                            page_badge = f'<span style="display:inline-block;background:#F1F5F9;color:#475569;padding:1px 8px;border-radius:10px;font-size:0.68rem;font-weight:500;margin-left:6px;">第{page}页</span>' if page else ""
                            content = src.get("content", "")[:250]
                            st.markdown(
                                f"""<div class="source-card">
<div class="src-header">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
[{i+1}] {filename}{page_badge}
</div>
<div class="src-body">{content}&hellip;</div>
</div>""",
                                unsafe_allow_html=True,
                            )

# ── Chat input ──
if prompt := st.chat_input("输入您的问题..."):
    if not st.session_state.api_online:
        st.error("API 服务未连接，请先启动后端: python run.py api")
    elif not st.session_state.user_api_key:
        st.error("请先在左侧填入您的 API Key")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            sources = []

            try:
                resp = requests.post(
                    f"{API_BASE}/api/query/stream",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                        "top_k": 5,
                        "api_key": st.session_state.user_api_key,
                        "api_base": st.session_state.user_api_base,
                        "model": st.session_state.user_api_model,
                    },
                    stream=True,
                    timeout=300,
                )

                if resp.status_code == 200:
                    full_answer = []
                    current_event = None

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("event:"):
                            current_event = line[6:].strip() or "token"
                            continue
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            if not payload:
                                continue
                            if current_event == "meta":
                                try:
                                    meta = json.loads(payload)
                                    sources = meta.get("sources", [])
                                except Exception:
                                    sources = []
                            elif current_event == "token":
                                full_answer.append(payload)
                                cursor = """<span style="display:inline-block;width:2px;height:1em;background:#2563EB;margin-left:1px;animation:blink 1s step-end infinite;vertical-align:text-bottom;"></span>"""
                                answer_placeholder.markdown("".join(full_answer) + cursor, unsafe_allow_html=True)
                            elif current_event == "error":
                                st.error(payload)

                    answer_text = "".join(full_answer) or "未能生成回答。"
                    answer_placeholder.markdown(answer_text)
                else:
                    resp2 = requests.post(
                        f"{API_BASE}/api/query",
                        json={
                            "query": prompt,
                            "session_id": st.session_state.session_id,
                            "top_k": 5,
                            "api_key": st.session_state.user_api_key,
                            "api_base": st.session_state.user_api_base,
                            "model": st.session_state.user_api_model,
                        },
                        timeout=120,
                    )
                    if resp2.status_code == 200:
                        data = resp2.json()
                        answer_text = data.get("answer", "未能生成回答。")
                        sources = data.get("sources", [])
                        st.markdown(answer_text)
                    else:
                        answer_text = f"请求失败: {resp2.text}"
                        sources = []
                        st.error(answer_text)

            except requests.ConnectionError:
                answer_text = "无法连接 API 服务，请先启动: python run.py api"
                sources = []
                st.error(answer_text)
            except Exception as e:
                answer_text = f"错误: {e}"
                sources = []
                st.error(answer_text)

            # Show sources
            if sources:
                with st.expander("📎 引用来源", expanded=False):
                    for i, src in enumerate(sources):
                        filename = src.get("filename", "未知文件")
                        page = src.get("page")
                        page_badge = f'<span style="display:inline-block;background:#F1F5F9;color:#475569;padding:1px 8px;border-radius:10px;font-size:0.68rem;font-weight:500;margin-left:6px;">第{page}页</span>' if page else ""
                        content = src.get("content", "")[:250]
                        st.markdown(
                            f"""<div class="source-card">
<div class="src-header">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
[{i+1}] {filename}{page_badge}
</div>
<div class="src-body">{content}&hellip;</div>
</div>""",
                            unsafe_allow_html=True,
                        )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
        })
