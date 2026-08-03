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
  getSessionMessages,
  deleteSession,
  renameSession,
  setWorkspace,
  createServerFolder,
  browseDirectory,
} from '@/api/client'
import type { Msg, Section, DebugData, KnowledgeDoc, SearchResult, AgentInfo, FileProposal, CreatedFile, Session, ToolInfo, ToolCapability, ToolExecutorSummary, ExecutionTask, SourceMode, ThemeMode, LanguageMode, DensityMode, MotionMode } from '@/types'

/* ------------------------------------------------------------------ */
/*  Singleton reactive state shared across all components              */
/* ------------------------------------------------------------------ */

const messages = ref<Msg[]>([])
const sessions = ref<Session[]>([])
const currentSessionId = ref<number | null>(null)

/* ---- SessionStorage persistence (survives HMR / component teardown) ---- */

const _MSG_KEY = 'omni_msgs'
const _SID_KEY = 'omni_sid'

watch(messages, (val) => {
  try { sessionStorage.setItem(_MSG_KEY, JSON.stringify(val)) } catch { /* quota */ }
}, { deep: true })

watch(currentSessionId, (val) => {
  if (val != null) sessionStorage.setItem(_SID_KEY, String(val))
  else sessionStorage.removeItem(_SID_KEY)
})

/** Restore messages + sessionId from sessionStorage when in-memory state is
 *  empty (recovers from HMR or unexpected component teardown).  Returns true
 *  when recovery data was found and applied. */
function _recoverFromStorage(): boolean {
  let recovered = false
  if (messages.value.length === 0) {
    try {
      const raw = sessionStorage.getItem(_MSG_KEY)
      if (raw) { messages.value = JSON.parse(raw); recovered = true }
    } catch { /* ignore */ }
  }
  if (currentSessionId.value == null) {
    const raw = sessionStorage.getItem(_SID_KEY)
    if (raw) { currentSessionId.value = Number(raw); recovered = true }
  }
  return recovered
}

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
    // Fallback: restore whatever was in sessionStorage
    _recoverFromStorage()
  }
})()

const thinking = ref(false)
const _savedSection = localStorage.getItem('active_section') as Section | null
const activeSection = ref<Section>(_savedSection || 'chat')

watch(activeSection, (val) => {
  localStorage.setItem('active_section', val)
})
const debugData = ref<DebugData | null>(null)
const sourceMode = ref<SourceMode>((localStorage.getItem('source_mode') as SourceMode | null) || 'auto')
const themeMode = ref<ThemeMode>((localStorage.getItem('theme_mode') as ThemeMode | null) || 'light')
const languageMode = ref<LanguageMode>((localStorage.getItem('language_mode') as LanguageMode | null) || 'zh-CN')
const densityMode = ref<DensityMode>((localStorage.getItem('density_mode') as DensityMode | null) || 'comfortable')
const motionMode = ref<MotionMode>((localStorage.getItem('motion_mode') as MotionMode | null) || 'full')

function applyAppearance() {
  const root = document.documentElement
  root.dataset.theme = themeMode.value
  root.dataset.density = densityMode.value
  root.dataset.motion = motionMode.value
  root.lang = languageMode.value
}

applyAppearance()

watch(sourceMode, (val) => {
  localStorage.setItem('source_mode', val)
})

watch([themeMode, languageMode, densityMode, motionMode], () => {
  localStorage.setItem('theme_mode', themeMode.value)
  localStorage.setItem('language_mode', languageMode.value)
  localStorage.setItem('density_mode', densityMode.value)
  localStorage.setItem('motion_mode', motionMode.value)
  applyAppearance()
})

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

/* ---- Session switching guard (moved here so it is a true singleton) ---- */
const switchingSession = ref(false)

export {
  messages, sessions, currentSessionId, thinking, activeSection, debugData,
  sourceMode, themeMode, languageMode, densityMode, motionMode,
  abortController, streamingPhase, streamingCategory, tasks,
  documents, searchQuery, searchResults, uploading, uploadStatus,
  agents, tools, toolCapabilities, toolExecutor,
  outputFiles, fileProposalStatuses,
  workspacePath, showFolderReminder, dirEntries, browseCurrentPath, workspaceError,
  switchingSession,
  _MSG_KEY, _SID_KEY,
  _recoverFromStorage, _loadSessionMessages, _refreshSessions, _upsertSession,
  traverseDir, collectFilesFromDrop,
}
