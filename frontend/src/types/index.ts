export interface ExecutionTask {
  id: string
  title: string
  tool: string
  status: 'todo' | 'running' | 'done' | 'failed' | 'blocked' | 'skipped'
}

export interface Msg {
  id: string
  role: 'user' | 'agent'
  text: string
  proposals?: FileProposal[]
}

export type Section = 'chat' | 'knowledge' | 'artifacts' | 'projects' | 'settings'

export type SourceMode = 'auto' | 'web' | 'knowledge'
export type ThemeMode = 'light' | 'dark' | 'ocean'
export type LanguageMode = 'zh-CN' | 'en-US'
export type DensityMode = 'comfortable' | 'compact'
export type MotionMode = 'full' | 'reduced'

export interface DebugData {
  category?: string
  workflow?: string[]
  search_results?: Array<{ title: string; url: string; snippet?: string }>
  router?: Record<string, unknown>
}

export interface KnowledgeDoc {
  id: number
  filename: string
  file_type: string
  file_size: number
  doc_metadata: string
  created_at: string
}

export interface SearchResult {
  chunk_id: number
  document_id: number
  filename: string
  content: string
  score: number
}

export interface ModelConfig {
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
}

export interface ModelCreateRequest {
  name: string
  provider?: string
  base_url: string
  api_key: string
  model_name: string
  temperature?: number
  max_tokens?: number
}

export interface ModelUpdateRequest {
  name?: string
  provider?: string
  base_url?: string
  api_key?: string
  model_name?: string
  temperature?: number
  max_tokens?: number
}

export interface FileProposal {
  suggestion_id: string
  filename: string
  language: string
  content: string
  preview: string
}

export interface CreatedFile {
  filename: string
  size: number
  created_at: string
  path: string
}

export interface FilePreview {
  filename: string
  path: string
  content: string
  truncated: boolean
  size: number
}

export interface Session {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface AgentInfo {
  key: string
  name: string
  description: string
  category: string
  status: 'active' | 'inactive'
  capabilities: string[]
  module_path: string
}

export interface ToolInfo {
  name: string
  description: string
  actions: string[]
  version: string
  metadata?: Record<string, unknown>
}

export interface ToolCapability {
  action: string
  tool: string
  description: string
}

export interface ToolExecutorSummary {
  registry: Record<string, string[]>
  capabilities: string[]
}
