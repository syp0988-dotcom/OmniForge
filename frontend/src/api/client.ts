import axios from 'axios'

import type { AgentInfo, CreatedFile, FilePreview, Session, ToolInfo, ToolCapability, ToolExecutorSummary, SourceMode } from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

/** Retry a fetch on network errors with exponential backoff.
 *
 * Only retries on ``TypeError`` (DNS failure, connection refused, CORS error).
 * Does NOT retry on ``AbortError`` (user cancellation) or HTTP error responses.
 *
 * @returns A tuple of [Response, retryCount] where retryCount = 0 means success on first try.
 */
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries = 2,
  onRetry?: (attempt: number, delayMs: number) => void,
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)
      return response
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))

      // Don't retry user-initiated abort
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err
      }

      // Only retry network errors (TypeError from fetch)
      if (!(err instanceof TypeError)) {
        throw err
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000 // 1s, 2s, 4s
        onRetry?.(attempt + 1, delay)
        await new Promise((resolve) => setTimeout(resolve, delay))
      }
    }
  }

  throw lastError || new Error('Fetch failed after retries')
}

/** Stream chat via SSE — calls onEvent for each SSE event, returns final data. */
export async function postChatStream(
  message: string,
  history: Array<{ role: string; content: string }>,
  sessionId: number | undefined,
  sourceMode: SourceMode,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<{ answer: string; session_id?: number; degraded?: boolean; degraded_reason?: string }> {
  const response = await fetchWithRetry(
    `${API_BASE}/chat/stream`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history, session_id: sessionId, source_mode: sourceMode }),
      signal,
    },
    2,
    (attempt, delay) => {
      onEvent('reconnecting', { attempt, maxRetries: 2, delay })
    },
  )

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''
  let finalAnswer = ''
  let finalSessionId: number | undefined
  let finalDegraded: boolean | undefined
  let finalDegradedReason: string | undefined
  let currentEvent = ''
  let currentData = ''

  const dispatchEvent = () => {
    if (!currentEvent) return
    try {
      const parsed = JSON.parse(currentData) as Record<string, unknown>
      onEvent(currentEvent, parsed)
      if (currentEvent === 'done') {
        finalAnswer = (parsed.answer as string) || ''
        finalSessionId = parsed.session_id as number | undefined
        finalDegraded = parsed.degraded as boolean | undefined
        finalDegradedReason = parsed.degraded_reason as string | undefined
      }
    } catch {
      // Ignore malformed/incomplete events; the next chunk can still recover.
    } finally {
      currentEvent = ''
      currentData = ''
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE events from buffer
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // keep incomplete line

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        currentData += currentData ? `\n${line.slice(6).trim()}` : line.slice(6).trim()
      } else if (line === '' && currentEvent) {
        dispatchEvent()
      }
    }
  }

  dispatchEvent()

  return { answer: finalAnswer, session_id: finalSessionId, degraded: finalDegraded, degraded_reason: finalDegradedReason }
}

/**
 * Non-streaming chat — only kept for backward compatibility.
 * New code should use postChatStream exclusively.
 * This now calls the async (non-blocking) endpoint internally.
 */
export async function postChat(message: string, history?: Array<{ role: string; content: string }>, sessionId?: number) {
  const resp = await axios.post(`${API_BASE}/chat`, { message, history: history || [], session_id: sessionId })
  return resp.data
}

export async function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  const resp = await axios.post(`${API_BASE}/upload`, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return resp.data
}

export async function getDocuments() {
  const resp = await axios.get(`${API_BASE}/knowledge/documents`)
  return resp.data as Array<{
    id: number
    filename: string
    file_type: string
    file_size: number
    doc_metadata: string
    created_at: string
  }>
}

export async function deleteDocument(docId: number) {
  const resp = await axios.delete(`${API_BASE}/knowledge/documents/${docId}`)
  return resp.data
}

export async function searchKnowledge(query: string, topK = 5) {
  const resp = await axios.post(`${API_BASE}/knowledge/search`, null, {
    params: { query, top_k: topK }
  })
  return resp.data as Array<{
    chunk_id: number
    document_id: number
    filename: string
    content: string
    score: number
  }>
}

/* ---- Agents introspection ------------------------------------------- */

export async function getAgents() {
  const resp = await axios.get(`${API_BASE}/agents`)
  return resp.data as AgentInfo[]
}

/* ---- Tools introspection ------------------------------------------------ */

export async function getTools() {
  const resp = await axios.get(`${API_BASE}/tools`)
  return resp.data as ToolInfo[]
}

export async function getCapabilities() {
  const resp = await axios.get(`${API_BASE}/tools/capabilities`)
  return resp.data as ToolCapability[]
}

export async function getExecutor() {
  const resp = await axios.get(`${API_BASE}/tools/executor`)
  return resp.data as ToolExecutorSummary
}

/* ---- Sessions ----------------------------------------------------------- */

export async function createSession() {
  const resp = await axios.post(`${API_BASE}/sessions/create`)
  return resp.data as Session
}

export async function listSessions(limit = 50) {
  const resp = await axios.get(`${API_BASE}/sessions`, { params: { limit } })
  return resp.data as Session[]
}

export async function getSessionMessages(sessionId: number) {
  const resp = await axios.get(`${API_BASE}/sessions/${sessionId}/messages`)
  return resp.data as Array<{ id: number; role: string; content: string; created_at: string }>
}

export async function renameSession(sessionId: number, title: string) {
  const resp = await axios.put(`${API_BASE}/sessions/${sessionId}/rename`, { title })
  return resp.data as { status: string }
}

export async function deleteSession(sessionId: number) {
  const resp = await axios.delete(`${API_BASE}/sessions/${sessionId}`)
  return resp.data as { status: string }
}

/* ---- Chat history ------------------------------------------------------- */

export async function getHistory(limit = 50) {
  const resp = await axios.get(`${API_BASE}/history`, { params: { limit } })
  return resp.data as Array<{ role: string; content: string; created_at: string }>
}

/* ---- Agent-generated file operations ------------------------------------ */

export async function createFile(filename: string, content: string, workspacePath?: string) {
  const resp = await axios.post(`${API_BASE}/files/create`, { filename, content, workspace_path: workspacePath })
  return resp.data as { status: string; filename: string; path: string }
}

export async function getOutputFiles(workspacePath?: string) {
  const params = workspacePath ? { workspace_path: workspacePath } : {}
  const resp = await axios.get(`${API_BASE}/files`, { params })
  return resp.data as CreatedFile[]
}

export async function readOutputFile(path: string, workspacePath?: string) {
  const resp = await axios.post(`${API_BASE}/files/read`, { path, workspace_path: workspacePath })
  return resp.data as FilePreview
}

/* ---- Workspace operations ----------------------------------------------- */

export async function setWorkspace(path: string) {
  const resp = await axios.post(`${API_BASE}/workspace/set`, { path })
  return resp.data as { status: string; path: string }
}

export async function createServerFolder(parentPath: string, folderName: string) {
  const resp = await axios.post(`${API_BASE}/workspace/create-folder`, {
    parent_path: parentPath,
    folder_name: folderName,
  })
  return resp.data as { status: string; path: string }
}

export async function browseDirectory(path: string) {
  const resp = await axios.get(`${API_BASE}/workspace/browse`, { params: { path } })
  return resp.data as {
    current_path: string
    entries: Array<{ name: string; is_dir: boolean; path: string }>
  }
}

/* ---- Model configuration ---- */

export async function getModels() {
  const resp = await axios.get(`${API_BASE}/models`)
  return resp.data as Array<{
    id: number
    name: string
    provider: string
    base_url: string
    model_name: string
    temperature: number
    max_tokens: number
    is_active: boolean
    created_at: string
    updated_at: string
  }>
}

export async function createModel(data: {
  name: string
  provider?: string
  base_url: string
  api_key: string
  model_name: string
  temperature?: number
  max_tokens?: number
}) {
  const resp = await axios.post(`${API_BASE}/models`, data)
  return resp.data as { id: number; status: string }
}

export async function updateModel(id: number, data: Record<string, unknown>) {
  const resp = await axios.put(`${API_BASE}/models/${id}`, data)
  return resp.data as { status: string }
}

export async function deleteModel(id: number) {
  const resp = await axios.delete(`${API_BASE}/models/${id}`)
  return resp.data as { status: string }
}

export async function activateModel(id: number) {
  const resp = await axios.post(`${API_BASE}/models/${id}/activate`)
  return resp.data as { status: string; model_name: string }
}
