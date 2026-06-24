// Session management using Netlify Blobs
import { getStore } from "@netlify/blobs";
import type { Session, ChatMessage } from "./types.ts";

const SESSION_STORE = "rag-sessions";

export async function getSession(sessionId: string): Promise<Session> {
  const store = getStore(SESSION_STORE);
  const session = await store.get(sessionId, { type: "json" });
  if (session) return session as Session;
  
  // Create new session
  const newSession: Session = {
    id: sessionId,
    messages: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  await store.setJSON(sessionId, newSession);
  return newSession;
}

export async function addMessage(sessionId: string, role: "user" | "assistant", content: string): Promise<void> {
  const session = await getSession(sessionId);
  session.messages.push({ role, content });
  // Keep last 20 messages
  if (session.messages.length > 20) {
    session.messages = session.messages.slice(-20);
  }
  session.updatedAt = new Date().toISOString();
  const store = getStore(SESSION_STORE);
  await store.setJSON(sessionId, session);
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const session = await getSession(sessionId);
  return session.messages.slice(-10);
}

export async function listSessions(): Promise<{ session_id: string; turn_count: number }[]> {
  const store = getStore(SESSION_STORE);
  const { blobs } = await store.list();
  const sessions: { session_id: string; turn_count: number }[] = [];
  for (const blob of blobs) {
    const session = await store.get(blob.key, { type: "json" }) as Session | null;
    if (session && session.messages.length > 0) {
      sessions.push({ session_id: session.id, turn_count: session.messages.length });
    }
  }
  return sessions;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const store = getStore(SESSION_STORE);
  await store.delete(sessionId);
}
