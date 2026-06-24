// Shared types for RAG Knowledge Base

export interface Chunk {
  id: string;
  content: string;
  embedding: number[];
  metadata: {
    filename: string;
    fileType: string;
    page?: number;
    chunkIndex: number;
  };
}

export interface Source {
  content: string;
  filename: string;
  fileType: string;
  page?: number;
  chunkIndex: number;
  source: string;
  chunkId: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  ready: boolean;
  uptime: number;
  chunkCount: number;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  files: string[];
  chunkCount: number;
}

export interface QueryRequest {
  query: string;
  session_id?: string;
  top_k?: number;
  api_key: string;
  api_base?: string;
  model?: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: Source[];
  source_count: number;
  session_id: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Session {
  id: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}
