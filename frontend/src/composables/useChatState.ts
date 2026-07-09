import { ref, watch } from 'vue'
import {
  postChatStream,
  uploadDocument,
  getDocuments,
  deleteDocument,
  searchKnowledge,
  getAgents,
  getTools,
  getCapabilities,
  getExecutor,
  createFile,
  getOutputFiles,
  listSessions,
  createSession,
  getSessionMessages,
  deleteSession,
  renameSession,
  setWorkspace,
  createServerFolder,
  browseDirectory,
} from '@/api/client'
import type { Msg, Section, DebugData, KnowledgeDoc, SearchResult, AgentInfo, FileProposal, CreatedFile, Session, ToolInfo, ToolCapability, ToolExecutorSummary, ExecutionTask } from '@/types'

/* ------------------------------------------------------------------ */
/*  Singleton reactive state shared across all components              */
/* ------------------------------------------------------------------ */

const messages = ref<Msg[]>([])
const sessions = ref<Session[]>([])
const currentSessionId = ref<number | null>(null)

/* Load sessions on startup, then load the most recent session's messages */
;(async () => {
  try {
    const sessList = await listSessions(50)
    // Dedup in case server returns duplicates
    const seen = new Set<number>()
    sessions.value = sessList.filter(s => {
      if (seen.has(s.id)) return false
      seen.add(s.id)
      return true
    })
    if (sessList.length > 0) {
      await _loadSessionMessages(sessList[0].id)
    }
  } catch (e) {
    console.warn('Failed to load sessions on startup:', e)
  }
})()

const thinking = ref(false)
const _savedSection = localStorage.getItem('active_section') as Section | null
const activeSection = ref<Section>(_savedSection || 'chat')

watch(activeSection, (val) => {
  localStorage.setItem('active_section', val)
})
const debugData = ref<DebugData | null>(null)

/* Streaming abort — created per request, null when idle */
const abortController = ref<AbortController | null>(null)

/* Streaming phase — updated live during SSE streaming */
const streamingPhase = ref<string>('')
const streamingCategory = ref<string>('')

/* Task queue — populated by task_update SSE events */
const tasks = ref<ExecutionTask[]>([])

const documents = ref<KnowledgeDoc[]>([])
const searchQuery = ref('')
const searchResults = ref<SearchResult[] | null>(null)
const uploading = ref(false)
const uploadStatus = ref<string | null>(null)

const agents = ref<AgentInfo[]>([])
const tools = ref<ToolInfo[]>([])
const toolCapabilities = ref<ToolCapability[]>([])
const toolExecutor = ref<ToolExecutorSummary | null>(null)

const outputFiles = ref<CreatedFile[]>([])
const fileProposalStatuses = ref<Record<string, 'pending' | 'created' | 'dismissed'>>({})

/* ---- Workspace state ---- */

const workspacePath = ref<string | null>(localStorage.getItem('workspace_path'))
const showFolderReminder = ref(false)
const dirEntries = ref<Array<{ name: string; is_dir: boolean; path: string }>>([])
const browseCurrentPath = ref('')
const workspaceError = ref<string | null>(null)

/* ---- Internal: load messages for a session into `messages` ref ---- */

async function _loadSessionMessages(sessionId: number) {
  try {
    const msgs = await getSessionMessages(sessionId)
    const loaded: Msg[] = msgs.map((m) => ({
      id: String(m.id),
      role: m.role as 'user' | 'agent',
      text: m.content,
    }))
    messages.value = loaded
    currentSessionId.value = sessionId
  } catch {
    messages.value = []
    currentSessionId.value = null
  }
}

/* ---- Internal: refresh session list (dedup + merge) ---- */

async function _refreshSessions() {
  try {
    const remote = await listSessions(50)
    // Merge: update existing entries, add new ones, dedup by id
    const existingMap = new Map(sessions.value.map(s => [s.id, s]))
    const merged: Session[] = []
    const seen = new Set<number>()
    for (const s of remote) {
      if (seen.has(s.id)) continue
      seen.add(s.id)
      const local = existingMap.get(s.id)
      if (local && local.title !== '新对话' && s.title === '新对话') {
        merged.push({ ...s, title: local.title })
      } else {
        merged.push(s)
      }
    }
    sessions.value = merged
  } catch (e) {
    console.warn('Failed to refresh sessions:', e)
  }
}

/** Update a single session in-place (moves to top, no full refresh). */
function _upsertSession(sess: Session) {
  const idx = sessions.value.findIndex(s => s.id === sess.id)
  if (idx !== -1) {
    sessions.value.splice(idx, 1)
  }
  sessions.value.unshift(sess)
  // Safety: remove any lingering duplicate
  const dup = sessions.value.findIndex((s, i) => i > 0 && s.id === sess.id)
  if (dup !== -1) sessions.value.splice(dup, 1)
}

/* ------------------------------------------------------------------ */
/*  File-system helpers (drag-and-drop directory traversal)            */
/* ------------------------------------------------------------------ */

async function traverseDir(entry: FileSystemEntry): Promise<File[]> {
  const files: File[] = []
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      ;(entry as FileSystemFileEntry).file(resolve, reject)
    })
    files.push(file)
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader()
    const entries = await new Promise<FileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject)
    })
    for (const child of entries) {
      files.push(...(await traverseDir(child)))
    }
  }
  return files
}

async function collectFilesFromDrop(items: DataTransferItemList): Promise<File[]> {
  const all: File[] = []
  for (let i = 0; i < items.length; i++) {
    const entry = items[i].webkitGetAsEntry()
    if (entry) {
      all.push(...(await traverseDir(entry)))
    } else if (items[i].kind === 'file') {
      const file = items[i].getAsFile()
      if (file) all.push(file)
    }
  }
  return all
}

/* ------------------------------------------------------------------ */
/*  Exported composable                                                */
/* ------------------------------------------------------------------ */

export function useChatState() {
  /* ---- Chat ---- */

  const handleSend = async (text: string) => {
    const localId = String(Date.now())
    const userMsg: Msg = { id: localId, role: 'user', text }
    messages.value = [...messages.value, userMsg]
    thinking.value = true
    streamingPhase.value = '发送中...'
    tasks.value = []

    // Create abort controller for this request
    abortController.value = new AbortController()
    const signal = abortController.value!.signal

    const history = messages.value
      .filter((m) => m.id !== localId)
      .map((m) => ({
        role: m.role === 'user' ? 'user' as const : 'assistant' as const,
        content: m.text,
      }))

    // Pre-create a placeholder agent message that gets filled progressively
    const agentMsgId = String(Date.now() + 1)
    const agentMsg: Msg = { id: agentMsgId, role: 'agent', text: '...' }
    messages.value = [...messages.value, agentMsg]

    try {
      // Streaming-only — no blocking fallback
      const result = await postChatStream(
        text,
        history,
        currentSessionId.value ?? undefined,
        (event, data) => {
          if (event === 'start') {
            streamingPhase.value = (data.phase as string) || '正在处理...'
          } else if (event === 'thinking') {
            streamingPhase.value = `分析中...`
            streamingCategory.value = (data.category as string) || ''
          } else if (event === 'planning') {
            streamingPhase.value = '制定计划...'
          } else if (event === 'searching') {
            streamingPhase.value = (data.phase as string) || '搜索中...'
          } else if (event === 'executing') {
            streamingPhase.value = (data.phase as string) || '执行中...'
          } else if (event === 'generating') {
            streamingPhase.value = '生成回答...'
          } else if (event === 'text') {
            // Progressive text delivery — replace placeholder on first text, then append
            const newText = (data.text as string) || ''
            if (agentMsg.text === '...') {
              agentMsg.text = newText
            } else {
              agentMsg.text += newText
            }
            // Trigger reactivity via splice (O(1) near-end, vs O(n) array spread)
            const idx = messages.value.findIndex((m) => m.id === agentMsgId)
            if (idx !== -1) {
              messages.value.splice(idx, 1, { ...agentMsg })
            }
          } else if (event === 'task_update') {
            tasks.value = (data.tasks as ExecutionTask[]) || []
          } else if (event === 'tools') {
            streamingPhase.value = '加载工具列表...'
          } else if (event === 'cancelled') {
            // User cancelled — remove empty placeholder and show state
            messages.value = messages.value.filter((m) => m.id !== agentMsgId)
            streamingPhase.value = '已中断'
          } else if (event === 'reconnecting') {
            streamingPhase.value = `正在重连... (${data.attempt}/${data.maxRetries})`
          }
        },
        signal,
      )

      // Finalize: apply answer from done event if text events didn't deliver it
      const idx = messages.value.findIndex((m) => m.id === agentMsgId)
      const noRealContent = !agentMsg.text || agentMsg.text === '...'
      if (idx !== -1 && noRealContent) {
        if (result.answer) {
          agentMsg.text = result.answer
        } else if (result.degraded) {
          agentMsg.text = '系统处于受限模式，部分功能暂时不可用。'
        } else {
          agentMsg.text = '抱歉，我没有生成有效的回答。请重试或换个方式提问。'
        }
        messages.value = [...messages.value.slice(0, idx), { ...agentMsg }, ...messages.value.slice(idx + 1)]
      }

      // Check degraded mode flag
      if (result.degraded) {
        streamingPhase.value = '受限模式'
        streamingCategory.value = '系统部分功能不可用'
      }

      // Track the session id from streaming response
      if (result.session_id) {
        currentSessionId.value = result.session_id
      }

      // Update the current session in-place (move to top) instead of full refresh.
      // This prevents duplicates and avoids an extra network round-trip.
      const cid = currentSessionId.value
      if (cid) {
        const current = sessions.value.find(s => s.id === cid)
        if (current) {
          _upsertSession({ ...current, updated_at: new Date().toISOString() })
        } else {
          await _refreshSessions()
        }
      } else {
        await _refreshSessions()
      }
      streamingPhase.value = ''
    } catch (err) {
      // User aborted → clean up
      if (err instanceof DOMException && err.name === 'AbortError') {
        // Remove the empty placeholder
        messages.value = messages.value.filter((m) => m.id !== agentMsgId)
        streamingPhase.value = '已中断'
        return
      }
      // Streaming failed — update the placeholder with error message
      agentMsg.text = '请求失败，请检查后端是否正常运行。'
      const idx = messages.value.findIndex((m) => m.id === agentMsgId)
      if (idx !== -1) {
        messages.value = [...messages.value.slice(0, idx), { ...agentMsg }, ...messages.value.slice(idx + 1)]
      }
    } finally {
      thinking.value = false
      abortController.value = null
      if (streamingPhase.value === '已中断') {
        setTimeout(() => { streamingPhase.value = '' }, 1000)
      } else {
        streamingPhase.value = ''
      }
    }
  }

  const stopChat = () => {
    if (abortController.value) {
      abortController.value.abort()
    }
  }

  const newChat = async () => {
    messages.value = []
    currentSessionId.value = null
    try {
      const sess = await createSession()
      currentSessionId.value = sess.id
      _upsertSession(sess)
    } catch (e) {
      console.warn('Failed to create new session:', e)
    }
  }

  const switchingSession = ref(false)

  const switchSession = async (sessionId: number) => {
    // Guard: don't switch to the already-active session
    if (currentSessionId.value === sessionId && !switchingSession.value) return
    // Guard: prevent concurrent session switches
    if (switchingSession.value) return
    // Abort any in-flight stream before switching
    if (abortController.value) {
      abortController.value.abort()
      abortController.value = null
    }
    thinking.value = false
    streamingPhase.value = ''
    tasks.value = []
    switchingSession.value = true
    try {
      activeSection.value = 'chat'
      await _loadSessionMessages(sessionId)
    } finally {
      switchingSession.value = false
    }
  }

  const deleteSessionById = async (sessionId: number) => {
    try {
      await deleteSession(sessionId)
      sessions.value = sessions.value.filter((s) => s.id !== sessionId)
      if (currentSessionId.value === sessionId) {
        if (sessions.value.length > 0) {
          await _loadSessionMessages(sessions.value[0].id)
        } else {
          messages.value = []
          currentSessionId.value = null
          const sess = await createSession()
          currentSessionId.value = sess.id
          _upsertSession(sess)
        }
      }
    } catch (e) {
      console.warn('Failed to delete session:', e)
    }
  }

  const renameSessionById = async (sessionId: number, title: string) => {
    try {
      await renameSession(sessionId, title)
      const idx = sessions.value.findIndex((s) => s.id === sessionId)
      if (idx !== -1) {
        sessions.value[idx] = { ...sessions.value[idx], title }
      }
    } catch (e) {
      console.warn('Failed to rename session:', e)
    }
  }

  /* ---- Agents ---- */

  const loadAgents = async () => {
    try {
      agents.value = await getAgents()
    } catch (e) {
      console.warn('Failed to load agents:', e)
      agents.value = []
    }
  }

  /* ---- Tools ---- */

  const loadTools = async () => {
    try {
      const [toolList, caps, executorSummary] = await Promise.all([
        getTools(),
        getCapabilities(),
        getExecutor(),
      ])
      tools.value = toolList
      toolCapabilities.value = caps
      toolExecutor.value = executorSummary
    } catch (e) {
      console.warn('Failed to load tools:', e)
    }
  }

  /* ---- Output files ---- */

  const createOutputFile = async (proposal: FileProposal) => {
    // Check if workspace is set
    if (!workspacePath.value) {
      showFolderReminder.value = true
      return
    }
    fileProposalStatuses.value = { ...fileProposalStatuses.value }
    try {
      const result = await createFile(proposal.filename, proposal.content, workspacePath.value ?? undefined)
      fileProposalStatuses.value = {
        ...fileProposalStatuses.value,
        [proposal.suggestion_id]: 'created',
      }
      outputFiles.value = [
        ...outputFiles.value,
        {
          filename: proposal.filename,
          size: proposal.content.length,
          created_at: new Date().toISOString(),
          path: result.path,
        },
      ]
    } catch {
      // Keep as pending so user can retry
      fileProposalStatuses.value = {
        ...fileProposalStatuses.value,
        [proposal.suggestion_id]: 'pending',
      }
    }
  }

  const dismissProposal = (suggestionId: string) => {
    fileProposalStatuses.value = {
      ...fileProposalStatuses.value,
      [suggestionId]: 'dismissed',
    }
  }

  const loadOutputFiles = async () => {
    try {
      const wp = workspacePath.value
      outputFiles.value = wp ? await getOutputFiles(wp) : await getOutputFiles()
    } catch (e) {
      console.warn('Failed to load output files:', e)
    }
  }

  /* ---- Workspace ---- */

  const handleSetWorkspace = async (path: string) => {
    try {
      workspaceError.value = null
      const result = await setWorkspace(path)
      workspacePath.value = result.path
      localStorage.setItem('workspace_path', result.path)
      showFolderReminder.value = false
      return result.path
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      workspaceError.value = msg
      throw e
    }
  }

  const handleCreateFolder = async (parentPath: string, folderName: string) => {
    const result = await createServerFolder(parentPath, folderName)
    // Auto-set as workspace
    workspacePath.value = result.path
    localStorage.setItem('workspace_path', result.path)
    showFolderReminder.value = false
    workspaceError.value = null
    return result.path
  }

  const handleBrowse = async (path: string = '.') => {
    try {
      const data = await browseDirectory(path)
      browseCurrentPath.value = data.current_path
      dirEntries.value = data.entries
    } catch (e) {
      console.warn('Failed to browse directory:', e)
      dirEntries.value = []
    }
  }

  const clearWorkspace = () => {
    workspacePath.value = null
    localStorage.removeItem('workspace_path')
    dirEntries.value = []
    browseCurrentPath.value = ''
    workspaceError.value = null
  }

  /* ---- Knowledge ---- */

  const loadDocs = async () => {
    try {
      documents.value = await getDocuments()
    } catch (e) {
      console.warn('Failed to load documents:', e)
    }
  }

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) return
    const allowedExts = [
      '.pdf', '.docx', '.doc', '.txt', '.md', '.markdown',
      '.html', '.htm', '.xlsx', '.xls', '.pptx', '.csv', '.epub',
      '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
      '.c', '.cpp', '.h', '.hpp',
    ]
    const filtered = files.filter((f) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      return allowedExts.includes(ext)
    })
    if (filtered.length === 0) {
      uploadStatus.value = '没有支持的文档格式（PDF、DOCX、TXT、MD、HTML、XLSX、PPTX、CSV、EPUB、代码文件）'
      return
    }
    uploading.value = true
    let successCount = 0
    let failCount = 0
    for (let i = 0; i < filtered.length; i++) {
      const file = filtered[i]
      uploadStatus.value = `正在上传 ${i + 1}/${filtered.length}: ${file.name}...`
      try {
        await uploadDocument(file)
        successCount++
      } catch (e) {
        console.warn('Failed to upload document:', file.name, e)
        failCount++
      }
    }
    uploadStatus.value = `上传完成：${successCount} 个成功${
      failCount > 0 ? `，${failCount} 个失败` : ''
    }`
    uploading.value = false
    await loadDocs()
  }

  const handleDelete = async (docId: number, filename: string) => {
    if (!confirm(`确定删除 "${filename}"？`)) return
    try {
      await deleteDocument(docId)
      documents.value = documents.value.filter((d) => d.id !== docId)
    } catch {
      uploadStatus.value = '删除失败'
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.value.trim()) return
    try {
      searchResults.value = await searchKnowledge(searchQuery.value.trim(), 10)
    } catch (e) {
      console.warn('Failed to search knowledge base:', e)
      searchResults.value = []
    }
  }

  /* ---- Formatting helpers ---- */

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateStr: string): string => {
    const d = new Date(dateStr + 'Z')
    return d.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  /* ---- Drag-drop helpers (re-exported for components) ---- */

  return {
    // state
    messages,
    tasks,
    sessions,
    currentSessionId,
    thinking,
    switchingSession,
    streamingPhase,
    streamingCategory,
    activeSection,
    debugData,
    documents,
    searchQuery,
    searchResults,
    uploading,
    uploadStatus,
    agents,
    tools,
    toolCapabilities,
    toolExecutor,
    outputFiles,
    fileProposalStatuses,
    // workspace state
    workspacePath,
    showFolderReminder,
    dirEntries,
    browseCurrentPath,
    workspaceError,
    // chat methods
    handleSend,
    stopChat,
    newChat,
    switchSession,
    deleteSessionById,
    renameSessionById,
    // agent methods
    loadAgents,
    // tool methods
    loadTools,
    // file methods
    createOutputFile,
    dismissProposal,
    loadOutputFiles,
    // workspace methods
    handleSetWorkspace,
    handleCreateFolder,
    handleBrowse,
    clearWorkspace,
    // knowledge methods
    loadDocs,
    uploadFiles,
    handleDelete,
    handleSearch,
    // helpers
    formatSize,
    formatDate,
    // file helpers
    collectFilesFromDrop,
  }
}

export type ChatState = ReturnType<typeof useChatState>
