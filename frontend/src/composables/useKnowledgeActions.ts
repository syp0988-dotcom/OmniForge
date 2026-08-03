import {
  getAgents, getTools, getCapabilities, getExecutor, createFile, getOutputFiles,
  getDocuments, uploadDocument, deleteDocument, searchKnowledge,
} from '@/api/client'
import type { FileProposal, CreatedFile } from '@/types'
import {
  documents, searchQuery, searchResults, uploading, uploadStatus, agents, tools,
  toolCapabilities, toolExecutor, outputFiles, fileProposalStatuses,
  workspacePath, showFolderReminder,
} from './chatState'

export function useKnowledgeActions() {
  const loadAgents = async () => {
    try {
      agents.value = await getAgents()
    } catch (e) {
      console.warn('Failed to load agents:', e)
      agents.value = []
    }
  }
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
      '.c', '.cpp', '.h', '.hpp', '.zip',
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
    const failedMessages: string[] = []
    for (let i = 0; i < filtered.length; i++) {
      const file = filtered[i]
      uploadStatus.value = `正在上传 ${i + 1}/${filtered.length}: ${file.name}...`
      try {
        await uploadDocument(file)
        successCount++
      } catch (e) {
        console.warn('Failed to upload document:', file.name, e)
        failCount++
        failedMessages.push(`${file.name}: ${getUploadErrorMessage(e)}`)
      }
    }
    uploadStatus.value = `上传完成：${successCount} 个成功${
      failCount > 0 ? `，${failCount} 个失败` : ''
    }`
    if (failedMessages.length > 0) {
      uploadStatus.value += `。${failedMessages.slice(0, 2).join('；')}`
    }
    uploading.value = false
    await loadDocs()
  }
  const getUploadErrorMessage = (error: unknown): string => {
    if (typeof error === 'object' && error !== null && 'response' in error) {
      const response = (error as { response?: { data?: { detail?: unknown }; status?: number } }).response
      const detail = response?.data?.detail
      if (typeof detail === 'string' && detail.trim()) return detail
      if (response?.status) return `HTTP ${response.status}`
    }
    if (error instanceof Error && error.message) return error.message
    return '未知错误'
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

  return { loadAgents, loadTools, createOutputFile, dismissProposal, loadOutputFiles, loadDocs, uploadFiles, handleDelete, handleSearch, formatSize, formatDate }
}
