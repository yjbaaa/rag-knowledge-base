// OpenAI-compatible embedding wrapper
import OpenAI from "openai";

export async function embedTexts(
  texts: string[],
  apiKey: string,
  apiBase: string = "https://api.openai.com/v1",
  model: string = "text-embedding-3-small"
): Promise<number[][]> {
  const client = new OpenAI({ apiKey, baseURL: apiBase });
  
  const batchSize = 20; // OpenAI recommends batching
  const allEmbeddings: number[][] = [];
  
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const response = await client.embeddings.create({
      model,
      input: batch,
    });
    allEmbeddings.push(...response.data.map(d => d.embedding));
  }
  
  return allEmbeddings;
}

export async function embedQuery(
  query: string,
  apiKey: string,
  apiBase: string = "https://api.openai.com/v1",
  model: string = "text-embedding-3-small"
): Promise<number[]> {
  const client = new OpenAI({ apiKey, baseURL: apiBase });
  const response = await client.embeddings.create({
    model,
    input: [query],
  });
  return response.data[0].embedding;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
