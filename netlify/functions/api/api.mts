// Main API function for RAG Knowledge Base
import type { Context, Config } from "@netlify/functions";
import { chunkText } from "../_shared/chunker.ts";
import { embedTexts, embedQuery } from "../_shared/embedding.ts";
import { storeChunks, searchSimilar, formatContext, chunksToSources, getChunkCount, getIndexedSources } from "../_shared/vector-store.ts";
import { generateAnswer, generateAnswerStream } from "../_shared/llm.ts";
import { parseFile } from "../_shared/parser.ts";
import { getSession, addMessage, getHistory, listSessions, deleteSession } from "../_shared/session.ts";
import type { Chunk, UploadResponse, QueryRequest } from "../_shared/types.ts";

const startTime = Date.now();

function corsHeaders(origin?: string): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
  };
}

function json(data: unknown, status = 200, extraHeaders: Record<string, string> = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...extraHeaders,
    },
  });
}

export default async (req: Request, context: Context) => {
  const origin = req.headers.get("origin") || undefined;
  const headers = corsHeaders(origin);
  
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers });
  }
  
  const url = new URL(req.url);
  const path = url.pathname;
  
  try {
    // ── Health ──
    if (path === "/api/health" && req.method === "GET") {
      return json({
        status: "ok",
        version: "2.0.0-netlify",
        ready: true,
        uptime: Math.floor((Date.now() - startTime) / 1000),
        chunkCount: await getChunkCount(),
      }, 200, headers);
    }
    
    // ── Upload ──
    if (path === "/api/upload" && req.method === "POST") {
      const formData = await req.formData();
      const files = formData.getAll("files") as File[];
      
      if (!files || files.length === 0) {
        return json({ error: "No files provided" }, 400, headers);
      }
      
      let allChunks: Chunk[] = [];
      const savedFiles: string[] = [];
      
      for (const file of files) {
        const ext = file.name.toLowerCase().split('.').pop() || '';
        if (!["pdf", "docx", "doc", "md", "txt"].includes(ext)) {
          return json({ error: `Unsupported file type: ${ext}` }, 400, headers);
        }
        
        const buffer = await file.arrayBuffer();
        let text: string;
        try {
          text = await parseFile(buffer, file.name, file.type);
        } catch (e: any) {
          return json({ error: `Failed to parse ${file.name}: ${e.message}` }, 400, headers);
        }
        
        if (!text || text.trim().length === 0) {
          continue;
        }
        
        // Chunk
        const chunks = chunkText(text, 500, 100);
        
        // Get API key from form data
        const apiKey = formData.get("api_key") as string || "";
        const apiBase = formData.get("api_base") as string || "https://api.openai.com/v1";
        
        if (!apiKey) {
          return json({ error: "API Key is required" }, 400, headers);
        }
        
        // Embed
        const embeddings = await embedTexts(chunks, apiKey, apiBase);
        
        // Create chunk objects
        const fileChunks: Chunk[] = chunks.map((content, i) => ({
          id: `${file.name.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}_${i}`,
          content,
          embedding: embeddings[i],
          metadata: {
            filename: file.name,
            fileType: ext,
            chunkIndex: i,
          },
        }));
        
        allChunks.push(...fileChunks);
        savedFiles.push(file.name);
      }
      
      if (allChunks.length === 0) {
        return json({ error: "No text extracted from files" }, 400, headers);
      }
      
      await storeChunks(allChunks);
      
      const result: UploadResponse = {
        success: true,
        message: `Uploaded ${savedFiles.length} file(s)`,
        files: savedFiles,
        chunkCount: allChunks.length,
      };
      
      return json(result, 200, headers);
    }
    
    // ── Query ──
    if (path === "/api/query" && req.method === "POST") {
      return handleQuery(req, headers, context, false);
    }
    
    // ── Query Stream ──
    if (path === "/api/query/stream" && req.method === "POST") {
      return handleQuery(req, headers, context, true);
    }
    
    // ── Sessions ──
    if (path === "/api/sessions" && req.method === "GET") {
      const sessions = await listSessions();
      return json(sessions, 200, headers);
    }
    
    // ── Delete session ──
    if (path.startsWith("/api/sessions/") && req.method === "DELETE") {
      const sessionId = path.split("/api/sessions/")[1];
      await deleteSession(sessionId);
      return json({ success: true, message: "Session cleared" }, 200, headers);
    }
    
    // ── Sources ──
    if (path === "/api/sources" && req.method === "GET") {
      const sources = await getIndexedSources();
      return json({ sources }, 200, headers);
    }
    
    return json({ error: "Not found" }, 404, headers);
    
  } catch (e: any) {
    console.error("API error:", e);
    return json({ error: e.message || "Internal server error" }, 500, headers);
  }
};

async function handleQuery(
  req: Request,
  headers: Record<string, string>,
  context: Context,
  stream: boolean
): Promise<Response> {
  const body = await req.json() as QueryRequest;
  const { query, session_id = "default", top_k = 5, api_key, api_base = "https://api.openai.com/v1", model = "gpt-4o-mini" } = body;
  
  if (!query || query.trim().length === 0) {
    return json({ error: "Query is required" }, 400, headers);
  }
  
  if (!api_key) {
    return json({ error: "API Key is required" }, 400, headers);
  }
  
  // Embed query
  const queryEmbedding = await embedQuery(query, api_key, api_base);
  
  // Search
  const results = await searchSimilar(queryEmbedding, top_k);
  const sources = chunksToSources(results);
  const contextStr = formatContext(results);
  
  // Get history
  const history = await getHistory(session_id);
  
  // Save user message
  await addMessage(session_id, "user", query);
  
  if (!contextStr) {
    const answer = "知识库中暂无相关文档，请先上传文件。";
    await addMessage(session_id, "assistant", answer);
    return json({ query, answer, sources: [], source_count: 0, session_id }, 200, headers);
  }
  
  if (stream) {
    // Streaming response
    const encoder = new TextEncoder();
    const streamHeaders = { ...headers, "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" };
    
    let fullAnswer = "";
    
    const readable = new ReadableStream({
      async start(controller) {
        try {
          // Send metadata first
          const meta = JSON.stringify({
            sources,
            rewritten_query: query,
            source_count: sources.length,
          });
          controller.enqueue(encoder.encode(`event:meta\ndata:${meta}\n\n`));
          
          // Stream tokens
          for await (const chunk of generateAnswerStream(contextStr, query, api_key, api_base, model, history)) {
            fullAnswer += chunk;
            controller.enqueue(encoder.encode(`event:token\ndata:${JSON.stringify(chunk)}\n\n`));
          }
          
          controller.enqueue(encoder.encode("event:done\ndata:\n\n"));
          controller.close();
          
          // Save answer
          await addMessage(session_id, "assistant", fullAnswer);
        } catch (e: any) {
          controller.enqueue(encoder.encode(`event:error\ndata:${JSON.stringify(e.message)}\n\n`));
          controller.close();
        }
      },
    });
    
    return new Response(readable, { status: 200, headers: streamHeaders });
  } else {
    // Non-streaming
    const answer = await generateAnswer(contextStr, query, api_key, api_base, model, history);
    await addMessage(session_id, "assistant", answer);
    
    return json({
      query,
      answer,
      sources,
      source_count: sources.length,
      session_id,
    }, 200, headers);
  }
}

export const config: Config = {
  path: "/api/*",
};
