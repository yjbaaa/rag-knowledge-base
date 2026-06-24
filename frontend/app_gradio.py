"""
RAG 知识库 - Gradio Web 界面 (直连版)
直接调用 RAG Pipeline，无需独立 API 服务
"""

import sys
import os
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app.rag.pipeline import RAGPipeline
from app.utils.citation import format_citation_card

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

pipeline = None
uploaded_sources = {}  # filename -> full_path


def init_pipeline():
    global pipeline
    if pipeline is None:
        import time
        t0 = time.perf_counter()
        pipeline = RAGPipeline(enable_rewrite=True, enable_hyde=False, enable_rerank=True, top_k=5)
        print(f"[STARTUP] Pipeline ready in {time.perf_counter()-t0:.1f}s")
        try:
            _ = pipeline.retriever.vector._embedding.embed_query("warmup")
        except:
            pass
    return pipeline


def refresh_file_dropdown():
    p = init_pipeline()
    sources = p.get_indexed_sources()
    choices = []
    for src in sources:
        name = Path(src).name
        choices.append(f"{name}  [{src}]")
    return gr.update(choices=choices, value=None)


def do_clear():
    global uploaded_sources
    p = init_pipeline()
    p.clear_all()
    uploaded_sources.clear()
    dd = refresh_file_dropdown()
    return "知识库已清空", "", dd


def do_upload(files):
    global uploaded_sources
    if not files:
        return "请先选择文件", "", gr.update()
    p = init_pipeline()
    paths = [str(f) for f in files]
    for fp in paths:
        uploaded_sources[Path(fp).name] = fp
    count = p.index_files(paths)
    dd = refresh_file_dropdown()
    fl = chr(10).join(f"  {i+1}. {n}" for i, n in enumerate(uploaded_sources.keys()))
    return f"已索引 {len(paths)} 个文件（{count} 个文本块）", fl, dd


def do_delete(selected):
    global uploaded_sources
    if not selected:
        return "请先选择要删除的文件", "", gr.update()
    p = init_pipeline()
    import re
    m = re.search(r"\[([^\]]+)\]", selected)
    if not m:
        return "无法解析文件路径", "", gr.update()
    filepath = m.group(1)
    filename = Path(filepath).name
    deleted = p.delete_files([filepath])
    if filename in uploaded_sources:
        del uploaded_sources[filename]
    dd = refresh_file_dropdown()
    fl = chr(10).join(f"  {i+1}. {n}" for i, n in enumerate(uploaded_sources.keys()))
    return f"已删除 {filename}（{deleted} 个文本块）", fl, dd


def do_chat(message, history):
    if not message or not message.strip():
        return history
    p = init_pipeline()
    try:
        result = p.ask(message, session_id="gradio_session")
        answer = result.answer or "未能生成回答。"
        if result.sources:
            answer += chr(10) + chr(10) + "---" + chr(10) + "**引用来源：**" + chr(10)
            answer += format_citation_card(result.sources, fmt="markdown")
    except Exception as e:
        answer = f"错误：{e}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history


def new_chat():
    p = init_pipeline()
    p.clear_session("gradio_session")
    return [], "会话已新建"


with gr.Blocks(title="RAG 知识库问答系统") as demo:
    gr.Markdown("# RAG 知识库智能问答")
    gr.Markdown("向知识库提问，回答附带原文引用来源。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 上传文档")
            f_upload = gr.File(label="选择文档", file_types=[".pdf", ".docx", ".doc", ".md", ".txt"], file_count="multiple")
            btn_upload = gr.Button("索引文档", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)

            gr.Markdown("### 管理文档")
            dd_files = gr.Dropdown(label="已索引文件", choices=[], interactive=True)
            btn_delete = gr.Button("删除文档", variant="stop")
            btn_clear = gr.Button("清空知识库", variant="stop", size="sm")
            file_list = gr.Textbox(label="已索引列表", interactive=False, lines=3)

            gr.Markdown("---")
            gr.Markdown("### 会话")
            btn_new = gr.Button("新建会话", size="sm")
            session_info = gr.Textbox(label="状态", value="就绪", interactive=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=550)
            msg = gr.Textbox(placeholder="在知识库中提问...", label="输入", show_label=False)
            btn_send = gr.Button("发送", variant="primary")

    btn_upload.click(do_upload, [f_upload], [status, file_list, dd_files])
    btn_delete.click(do_delete, [dd_files], [status, file_list, dd_files])
    btn_clear.click(do_clear, [], [status, file_list, dd_files])
    btn_new.click(new_chat, [], [chatbot, session_info])
    btn_send.click(do_chat, [msg, chatbot], [chatbot]).then(lambda: "", outputs=[msg])
    msg.submit(do_chat, [msg, chatbot], [chatbot]).then(lambda: "", outputs=[msg])


if __name__ == "__main__":
    import threading, time

    def _prewarm():
        time.sleep(0.5)
        print("[PREWARM] Loading model...")
        init_pipeline()

    threading.Thread(target=_prewarm, daemon=True).start()

    demo.launch(server_name="0.0.0.0", server_port=8501, share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)