// OpenAI-compatible chat LLM wrapper
import OpenAI from "openai";
import type { ChatMessage } from "./types.ts";

const SYSTEM_PROMPT = `You are a knowledgeable assistant for an enterprise knowledge base. Answer the user's question based on the provided context. You MUST answer in Simplified Chinese (简体中文).

Rules:
1. Answer ONLY using information from the context below. If the context doesn't contain the answer, say "提供的文档中未包含相关信息。"
2. Cite sources inline using [1], [2] format matching the source numbers in the context.
3. Be concise and accurate. Use bullet points for lists when appropriate.
4. If the question is ambiguous, ask for clarification rather than guessing.
5. Always respond in Chinese (简体中文), regardless of the language of the source documents.`;

export async function generateAnswer(
  context: string,
  query: string,
  apiKey: string,
  apiBase: string = "https://api.openai.com/v1",
  model: string = "gpt-4o-mini",
  history?: ChatMessage[]
): Promise<string> {
  const client = new OpenAI({ apiKey, baseURL: apiBase });
  
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
  ];
  
  if (history && history.length > 0) {
    messages.push(...history.map(m => ({ role: m.role as "user" | "assistant", content: m.content })));
  }
  
  messages.push({
    role: "user",
    content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer:`,
  });
  
  const response = await client.chat.completions.create({
    model,
    messages,
    temperature: 0.3,
    max_tokens: 2000,
  });
  
  return response.choices[0]?.message?.content || "未能生成回答。";
}

export async function* generateAnswerStream(
  context: string,
  query: string,
  apiKey: string,
  apiBase: string = "https://api.openai.com/v1",
  model: string = "gpt-4o-mini",
  history?: ChatMessage[]
): AsyncGenerator<string> {
  const client = new OpenAI({ apiKey, baseURL: apiBase });
  
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
  ];
  
  if (history && history.length > 0) {
    messages.push(...history.map(m => ({ role: m.role as "user" | "assistant", content: m.content })));
  }
  
  messages.push({
    role: "user",
    content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer:`,
  });
  
  const stream = await client.chat.completions.create({
    model,
    messages,
    temperature: 0.3,
    max_tokens: 2000,
    stream: true,
  });
  
  for await (const chunk of stream) {
    if (chunk.choices[0]?.delta?.content) {
      yield chunk.choices[0].delta.content;
    }
  }
}
