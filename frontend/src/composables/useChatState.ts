import { ref, type Ref } from 'vue'
import {
  postChat,
  uploadDocument,
  getDocuments,
  deleteDocument,
  searchKnowledge,
  getAgents,
  createFile,
  getOutputFiles,
} from '@/api/client'
import type { Msg, Section, DebugData, KnowledgeDoc, SearchResult, AgentInfo, FileProposal, CreatedFile } from '@/types'

/* ------------------------------------------------------------------ */
/*  Singleton reactive state shared across all components              */
/* ------------------------------------------------------------------ */

const messages = ref<Msg[]>([
  { id: '1', role: 'agent', text: '欢迎使用 OmniForge。' },
])
const thinking = ref(false)
const activeSection = ref<Section>('chat')
const debugData = ref<DebugData | null>(null)

const documents = ref<KnowledgeDoc[]>([])
const searchQuery = ref('')
const searchResults = ref<SearchResult[] | null>(null)
const uploading = ref(false)
const uploadStatus = ref<string | null>(null)

const agents = ref<AgentInfo[]>([])

const outputFiles = ref<CreatedFile[]>([])
const fileProposalStatuses = ref<Record<string, 'pending' | 'created' | 'dismissed'>>({})

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
    const id = String(Date.now())
    messages.value = [...messages.value, { id, role: 'user', text }]
    thinking.value = true

    try {
      // Build history from previous messages (exclude the welcome message)
      const history = messages.value
        .filter((m) => m.id !== '1')
        .map((m) => ({
          role: m.role === 'user' ? 'user' as const : 'assistant' as const,
          content: m.text,
        }))
      const data = await postChat(text, history)
      const reply = data.reply || '[no reply]'
      debugData.value = data.debug || null

      const agentMsg: Msg = {
        id: String(Date.now()),
        role: 'agent',
        text: reply,
      }

      // Attach file proposals if present
      if (data.proposed_files && data.proposed_files.length > 0) {
        agentMsg.proposals = data.proposed_files as FileProposal[]
        const statusMap: Record<string, 'pending' | 'created' | 'dismissed'> = {}
        for (const p of data.proposed_files as FileProposal[]) {
          statusMap[p.suggestion_id] = 'pending'
        }
        fileProposalStatuses.value = statusMap
      }

      messages.value = [...messages.value, agentMsg]
    } catch {
      messages.value = [
        ...messages.value,
        {
          id: String(Date.now()),
          role: 'agent',
          text: '请求失败，请检查后端。',
        },
      ]
    } finally {
      thinking.value = false
    }
  }

  const newChat = () => {
    messages.value = []
  }

  /* ---- Agents ---- */

  const loadAgents = async () => {
    try {
      agents.value = await getAgents()
    } catch {
      agents.value = []
    }
  }

  /* ---- Output files ---- */

  const createOutputFile = async (proposal: FileProposal) => {
    fileProposalStatuses.value = { ...fileProposalStatuses.value }
    try {
      const result = await createFile(proposal.filename, proposal.content)
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
      outputFiles.value = await getOutputFiles()
    } catch {
      /* silently fail */
    }
  }

  /* ---- Knowledge ---- */

  const loadDocs = async () => {
    try {
      documents.value = await getDocuments()
    } catch {
      /* silently fail */
    }
  }

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) return
    const allowedExts = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown']
    const filtered = files.filter((f) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      return allowedExts.includes(ext)
    })
    if (filtered.length === 0) {
      uploadStatus.value = '没有支持的文档格式（PDF、DOCX、TXT、MD）'
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
      } catch {
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
    } catch {
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
    thinking,
    activeSection,
    debugData,
    documents,
    searchQuery,
    searchResults,
    uploading,
    uploadStatus,
    agents,
    outputFiles,
    fileProposalStatuses,
    // chat methods
    handleSend,
    newChat,
    // agents methods
    loadAgents,
    // file methods
    createOutputFile,
    dismissProposal,
    loadOutputFiles,
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

/* Type helper – allows `inject('chatState')` to infer the right shape */
export type ChatState = ReturnType<typeof useChatState>
