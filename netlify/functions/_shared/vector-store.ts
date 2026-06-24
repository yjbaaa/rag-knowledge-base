// Vector store using Netlify Blobs
import { getStore } from "@netlify/blobs";
import type { Chunk, Source } from "./types.ts";
import { cosineSimilarity } from "./embedding.ts";

const STORE_NAME = "rag-vectors";
const CHUNK_PREFIX = "chunks/";

export async function storeChunks(chunks: Chunk[]): Promise<void> {
  const store = getStore(STORE_NAME);
  // Store an index list
  const existing = await listChunkIds();
  const newIds = chunks.map(c => c.id);
  const allIds = [...existing, ...newIds];
  await store.setJSON("index", allIds);
  
  // Store each chunk individually
  for (const chunk of chunks) {
    await store.setJSON(CHUNK_PREFIX + chunk.id, chunk);
  }
}

async function listChunkIds(): Promise<string[]> {
  const store = getStore(STORE_NAME);
  const ids = await store.get("index", { type: "json" });
  return (ids as string[]) || [];
}

export async function getAllChunks(): Promise<Chunk[]> {
  const store = getStore(STORE_NAME);
  const ids = await listChunkIds();
  if (ids.length === 0) return [];
  
  const chunks: Chunk[] = [];
  for (const id of ids) {
    const chunk = await store.get(CHUNK_PREFIX + id, { type: "json" });
    if (chunk) chunks.push(chunk as Chunk);
  }
  return chunks;
}

export async function searchSimilar(
  queryEmbedding: number[],
  topK: number = 5
): Promise<{ chunk: Chunk; score: number }[]> {
  const chunks = await getAllChunks();
  if (chunks.length === 0) return [];
  
  const scored = chunks.map(chunk => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));
  
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

export function chunksToSources(results: { chunk: Chunk; score: number }[]): Source[] {
  return results.map((r, i) => ({
    content: r.chunk.content.slice(0, 200),
    filename: r.chunk.metadata.filename,
    fileType: r.chunk.metadata.fileType,
    page: r.chunk.metadata.page,
    chunkIndex: r.chunk.metadata.chunkIndex,
    source: r.chunk.metadata.filename,
    chunkId: r.chunk.id,
  }));
}

export function formatContext(results: { chunk: Chunk; score: number }[]): string {
  return results.map((r, i) => 
    `[Source ${i + 1}: ${r.chunk.metadata.filename}${r.chunk.metadata.page ? ` (Page ${r.chunk.metadata.page})` : ''}]\n${r.chunk.content}`
  ).join('\n\n---\n\n');
}

export async function getChunkCount(): Promise<number> {
  const ids = await listChunkIds();
  return ids.length;
}

export async function getIndexedSources(): Promise<string[]> {
  const chunks = await getAllChunks();
  const sources = new Set(chunks.map(c => c.metadata.filename));
  return Array.from(sources);
}

export async function clearAll(): Promise<void> {
  const store = getStore(STORE_NAME);
  const ids = await listChunkIds();
  for (const id of ids) {
    await store.delete(CHUNK_PREFIX + id);
  }
  await store.delete("index");
}
