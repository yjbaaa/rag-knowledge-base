"""
FastAPI REST API - RAG Knowledge Base
"""

import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.core.config import DATA_DIR, API_HOST, API_PORT
from app.rag.pipeline import get_pipeline


# 进程级状态：模型预热进度，供 /api/health 上报给前端
_app_state = {"ready": False, "started_at": None, "warmup_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动钩子：在进程启动时就预热 pipeline，把 ~12s 的模型加载
    开销前置到服务启动阶段，避免用户第一次请求时才触发加载。"""
    print("[STARTUP] Warming up RAG pipeline (loading embedding model)...")
    t0 = time.perf_counter()
    try:
        pipeline = get_pipeline()
        # 触发 retriever / VectorEngine / embedding 模型的实际加载
        _ = pipeline.retriever
        elapsed = time.perf_counter() - t0
        _app_state["ready"] = True
        _app_state["started_at"] = time.time()
        print(f"[STARTUP] Pipeline ready in {elapsed:.2f}s.")
    except Exception as e:
        _app_state["warmup_error"] = str(e)
        print(f"[STARTUP] Warmup FAILED: {e}")
    yield
    print("[SHUTDOWN] RAG pipeline shutting down.")


app = FastAPI(
    title="RAG Knowledge Base API",
    version="1.0.0",
    lifespan=lifespan,
)


# ========== Schemas ==========

class ApiConfig(BaseModel):
    api_key: str = Field(default="")
    api_base: str = Field(default="")
    model: str = Field(default="")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    api_key: str = Field(default="")
    api_base: str = Field(default="")
    model: str = Field(default="")


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str = ""
    answer: str
    sources: List[dict] = []
    source_count: int = 0
    session_id: str = ""
    citation_result: Optional[dict] = None


class UploadResponse(BaseModel):
    success: bool
    message: str
    files: List[str] = []
    chunk_count: int = 0


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int


# ========== Health ==========

@app.get("/api/health")
async def health():
    uptime = (time.time() - _app_state["started_at"]) if _app_state["started_at"] else 0
    return {
        "status": "ok",
        "version": "1.0.0",
        "ready": _app_state["ready"],          # 模型是否预热完成
        "uptime_seconds": round(uptime, 1),
        "warmup_error": _app_state["warmup_error"],
    }


# ========== Upload ==========

@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not _app_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Service is still warming up the embedding model. Retry in a few seconds.",
        )
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    pipeline = get_pipeline()
    saved_paths = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in {".pdf", ".docx", ".doc", ".md", ".txt"}:
            raise HTTPException(status_code=400, detail=f"Unsupported: {ext}")
        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = DATA_DIR / safe_name
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_paths.append(str(file_path))

    t0 = time.perf_counter()
    chunk_count = pipeline.index_files(saved_paths)
    elapsed = time.perf_counter() - t0
    print(f"[UPLOAD] Indexed {len(saved_paths)} file(s) -> {chunk_count} chunks in {elapsed:.3f}s")
    return UploadResponse(
        success=True,
        message=f"Uploaded {len(saved_paths)} file(s)",
        files=[Path(p).name for p in saved_paths],
        chunk_count=chunk_count,
    )


# ========== Query ==========

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    pipeline = get_pipeline()
    result = pipeline.ask(
        query=req.query,
        session_id=req.session_id,
        top_k=req.top_k,
        api_key=req.api_key,
        api_base=req.api_base,
        model=req.model,
    )
    return QueryResponse(**result.to_dict())


# ========== Query Stream ==========

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    pipeline = get_pipeline()

    async def event_gen():
        try:
            for chunk in pipeline.ask_stream(
                query=req.query,
                session_id=req.session_id,
                top_k=req.top_k,
                api_key=req.api_key,
                api_base=req.api_base,
                model=req.model,
            ):
                if isinstance(chunk, dict):
                    # Metadata event: sources + rewrite info (emitted once, up front)
                    yield {"event": "meta", "data": json.dumps(chunk, ensure_ascii=False)}
                else:
                    # Token event: answer text chunk
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_gen())


# ========== Sessions ==========

@app.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions():
    pipeline = get_pipeline()
    result = []
    for sid in pipeline.session_mgr.list_sessions():
        mem = pipeline.session_mgr.get_session(sid)
        if mem.turn_count > 0:
            result.append(SessionInfo(session_id=sid, turn_count=mem.turn_count))
    return result


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    get_pipeline().clear_session(session_id)
    return {"success": True, "message": f"Session cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.routes:app", host=API_HOST, port=API_PORT)
