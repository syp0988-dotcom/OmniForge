<template>
  <div class="max-w-content mx-auto px-6 py-8 space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-lg font-semibold text-text">项目</h2>
      <p class="text-sm text-secondary mt-1">管理工作文件夹，导入或创建文件夹来存放对话生成的文件。</p>
    </div>

    <!-- Workspace section -->
    <div class="rounded-xl border border-border bg-white p-5 space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Folder class="w-5 h-5 text-primary" />
          <span class="text-sm font-medium text-text">工作文件夹</span>
        </div>
        <button
          v-if="chatState.workspacePath.value"
          class="px-3 py-1.5 rounded-lg border border-border text-xs text-secondary hover:bg-hover transition-colors"
          @click="clearWorkspace"
        >
          更换文件夹
        </button>
      </div>

      <!-- Current workspace path display -->
      <div v-if="chatState.workspacePath.value" class="flex items-center gap-2 px-3 py-2 rounded-lg bg-code-bg">
        <Folder class="w-4 h-4 text-primary shrink-0" />
        <span class="text-sm font-mono text-text truncate">{{ chatState.workspacePath.value }}</span>
        <span class="ml-auto text-xs text-green-600 flex items-center gap-1">
          <CheckCircle class="w-3.5 h-3.5" />已就绪
        </span>
      </div>

      <!-- No workspace set -->
      <div v-else class="space-y-3">
        <!-- Import folder -->
        <button
          class="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-dashed border-border hover:border-primary hover:bg-primary/5 transition-colors group"
          @click="startImport"
        >
          <div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
            <FolderInput class="w-5 h-5 text-primary" />
          </div>
          <div class="text-left">
            <div class="text-sm font-medium text-text">导入已有文件夹</div>
            <div class="text-xs text-secondary mt-0.5">选择电脑上的一个文件夹作为工作目录</div>
          </div>
        </button>
        <input
          ref="folderInputRef"
          type="file"
          class="hidden"
          webkitdirectory
          @change="onFolderSelected"
        />

        <!-- Create folder -->
        <button
          class="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-dashed border-border hover:border-primary hover:bg-primary/5 transition-colors group"
          @click="showCreateDialog = true"
        >
          <div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
            <FolderPlus class="w-5 h-5 text-primary" />
          </div>
          <div class="text-left">
            <div class="text-sm font-medium text-text">创建新文件夹</div>
            <div class="text-xs text-secondary mt-0.5">选择路径并创建新的工作文件夹</div>
          </div>
        </button>
      </div>

      <!-- Error display -->
      <div
        v-if="chatState.workspaceError.value"
        class="px-3 py-2 rounded-lg bg-red-50 text-xs text-red-600 flex items-center gap-2"
      >
        <AlertCircle class="w-4 h-4 shrink-0" />
        {{ chatState.workspaceError.value }}
      </div>
    </div>

    <!-- Create folder dialog -->
    <div v-if="showCreateDialog" class="rounded-xl border border-border bg-white p-5 space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-medium text-text flex items-center gap-2">
          <FolderPlus class="w-4 h-4 text-primary" />
          创建新文件夹
        </h3>
        <button class="text-secondary hover:text-text transition-colors" @click="showCreateDialog = false">
          <X class="w-4 h-4" />
        </button>
      </div>

      <!-- Parent path -->
      <div>
        <label class="text-xs text-secondary mb-1.5 block">上级目录路径</label>
        <div class="flex gap-2">
          <input
            ref="parentPathInputRef"
            v-model="parentPath"
            class="flex-1 px-3 py-2 rounded-lg border border-border bg-white text-sm text-text outline-none focus:border-primary transition-colors font-mono"
            placeholder="D:\projects"
          />
          <button
            class="px-4 py-2 rounded-lg border border-border text-sm text-secondary hover:bg-hover transition-colors shrink-0"
            @click="pickParentFolder"
          >
            选择文件夹
          </button>
        </div>
        <p class="text-xs text-secondary mt-1.5">输入路径，或点击"选择文件夹"来选取</p>
      </div>
      <input
        ref="createFolderInputRef"
        type="file"
        class="hidden"
        webkitdirectory
        @change="onParentFolderPicked"
      />

      <!-- Folder name input -->
      <div>
        <label class="text-xs text-secondary mb-1.5 block">新文件夹名称</label>
        <input
          v-model="newFolderName"
          class="w-full px-3 py-2 rounded-lg border border-border bg-white text-sm text-text outline-none focus:border-primary transition-colors"
          placeholder="输入文件夹名称"
          @keydown.enter="doCreateFolder"
        />
      </div>

      <div class="flex gap-2 justify-end">
        <button
          class="px-4 py-2 rounded-lg border border-border text-sm text-secondary hover:bg-hover transition-colors"
          @click="showCreateDialog = false"
        >
          取消
        </button>
        <button
          :disabled="!newFolderName.trim() || !parentPath.trim()"
          class="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          @click="doCreateFolder"
        >
          创建并设为工作目录
        </button>
      </div>
    </div>

    <!-- Files list -->
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-medium text-text flex items-center gap-2">
          <FileIcon class="w-4 h-4" />
          生成的文件
        </h3>
        <span v-if="chatState.workspacePath.value" class="text-xs text-secondary">
          共 {{ sortedFiles.length }} 个文件
        </span>
      </div>

      <div v-if="sortedFiles.length > 0" class="grid gap-2">
        <button
          v-for="(item, idx) in sortedFiles"
          :key="idx"
          type="button"
          class="flex items-center justify-between p-4 rounded-xl border border-border bg-white text-left transition-colors hover:bg-hover focus:outline-none focus:border-primary"
          @click="openFilePreview(item)"
        >
          <div class="flex items-center gap-3 min-w-0">
            <FileIcon class="w-5 h-5 text-secondary shrink-0" />
            <div class="min-w-0">
              <div class="text-sm font-medium text-text truncate">{{ item.filename }}</div>
              <div class="text-xs text-secondary mt-0.5">
                {{ formatSize(item.size) }} · {{ formatDate(item.created_at) }}
              </div>
            </div>
          </div>
          <span class="flex items-center gap-2 text-xs text-secondary font-mono ml-4 truncate max-w-[240px]">
            <Eye class="w-3.5 h-3.5 shrink-0" />
            <span class="truncate">{{ item.path }}</span>
          </span>
        </button>
      </div>

      <!-- Empty state -->
      <div v-else class="text-center py-12 text-secondary text-sm">
        <FileIcon class="w-12 h-12 mx-auto mb-4 opacity-40" />
        <p v-if="chatState.workspacePath.value">工作文件夹中暂无生成的文件</p>
        <p v-else>请先设置工作文件夹</p>
        <p class="mt-1 text-xs">会话中的文件提案可一键保存到工作文件夹</p>
      </div>
    </div>

    <!-- File preview modal -->
    <Teleport to="body">
      <div
        v-if="previewOpen"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm px-4"
        @click.self="closeFilePreview"
      >
        <div class="w-full max-w-4xl max-h-[84vh] rounded-xl border border-border bg-white shadow-xl flex flex-col">
          <div class="flex items-center justify-between gap-4 px-5 py-4 border-b border-border">
            <div class="min-w-0">
              <div class="text-sm font-semibold text-text truncate">{{ selectedPreview?.filename || '文件预览' }}</div>
              <div class="text-xs text-secondary font-mono truncate mt-0.5">{{ selectedPreview?.path }}</div>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <button
                v-if="selectedPreview"
                class="px-3 py-1.5 rounded-lg border border-border text-xs text-secondary hover:bg-hover transition-colors"
                @click="copyPreviewContent"
              >
                复制内容
              </button>
              <button class="p-2 rounded-lg text-secondary hover:bg-hover hover:text-text transition-colors" @click="closeFilePreview">
                <X class="w-4 h-4" />
              </button>
            </div>
          </div>

          <div v-if="previewLoading" class="p-8 text-center text-sm text-secondary">
            正在读取文件...
          </div>
          <div v-else-if="previewError" class="p-5 text-sm text-red-600">
            {{ previewError }}
          </div>
          <div v-else-if="selectedPreview" class="min-h-0 flex-1 overflow-auto">
            <div
              v-if="selectedPreview.truncated"
              class="px-5 py-2 border-b border-border bg-primary/5 text-xs text-primary"
            >
              文件较大，仅显示前 {{ formatSize(selectedPreview.content.length) }} 内容。
            </div>
            <pre class="p-5 text-sm leading-relaxed font-mono whitespace-pre-wrap text-text">{{ selectedPreview.content }}</pre>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, inject, onMounted, watch } from 'vue'
import { Folder, FolderInput, FolderPlus, File as FileIcon, CheckCircle, AlertCircle, X, Eye } from 'lucide-vue-next'
import { readOutputFile } from '@/api/client'
import type { ChatState } from '@/composables/useChatState'
import type { CreatedFile, FilePreview } from '@/types'

const chatState = inject<ChatState>('chatState')!

const folderInputRef = ref<HTMLInputElement | null>(null)
const createFolderInputRef = ref<HTMLInputElement | null>(null)
const parentPathInputRef = ref<HTMLInputElement | null>(null)
const showCreateDialog = ref(false)
const parentPath = ref('')
const newFolderName = ref('')
const previewOpen = ref(false)
const previewLoading = ref(false)
const previewError = ref('')
const selectedPreview = ref<FilePreview | null>(null)

onMounted(() => {
  chatState.loadOutputFiles()
  if (chatState.workspacePath.value) {
    parentPath.value = chatState.workspacePath.value
  }
})

watch(() => chatState.workspacePath.value, () => {
  chatState.loadOutputFiles()
  if (chatState.workspacePath.value) {
    parentPath.value = chatState.workspacePath.value
  }
})

const sortedFiles = computed(() => {
  return [...chatState.outputFiles.value].reverse()
})

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr)
  return d.toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  })
}

/* ---- Import folder ---- */

function startImport() {
  folderInputRef.value?.click()
}

async function onFolderSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (!input.files?.length) return
  // Get the root folder path from the first file's webkitRelativePath
  const firstFile = input.files[0]
  if (!firstFile.webkitRelativePath) return

  // Extract the folder path (webkitRelativePath gives "folder/subfolder/file")
  const parts = firstFile.webkitRelativePath.split('/')
  const rootFolder = parts[0]

  // We need the full path from the user, so prompt for it
  // (browser security doesn't expose the actual filesystem path)
  const guessedPath = prompt(
    `已选择文件夹 "${rootFolder}"，请确认文件夹的完整路径（浏览器无法自动获取路径）：\n例如: D:\\projects\\${rootFolder}`,
    chatState.workspacePath.value || ''
  )
  if (!guessedPath) return

  try {
    await chatState.handleSetWorkspace(guessedPath)
  } catch {
    // Error is displayed via workspaceError
  }

  // Reset input so same folder can be re-selected
  input.value = ''
}

/* ---- Pick parent folder for creation ---- */

function pickParentFolder() {
  createFolderInputRef.value?.click()
}

async function onParentFolderPicked(e: Event) {
  const input = e.target as HTMLInputElement
  if (!input.files?.length) return
  const firstFile = input.files[0]
  if (!firstFile.webkitRelativePath) return

  const parts = firstFile.webkitRelativePath.split('/')
  const rootFolder = parts[0]

  const guessedPath = prompt(
    `已选择文件夹 "${rootFolder}"，请确认上级目录的完整路径：\n例如: D:\\projects`,
    parentPath.value || ''
  )
  if (!guessedPath) return
  parentPath.value = guessedPath
  input.value = ''
  // Focus the folder name input
  setTimeout(() => parentPathInputRef.value?.focus(), 100)
}

/* ---- Create folder ---- */

async function doCreateFolder() {
  if (!newFolderName.value.trim() || !parentPath.value.trim()) return
  try {
    await chatState.handleCreateFolder(parentPath.value.trim(), newFolderName.value.trim())
    showCreateDialog.value = false
    newFolderName.value = ''
  } catch {
    // Error is displayed via workspaceError
  }
}

/* ---- Clear workspace ---- */

function clearWorkspace() {
  chatState.clearWorkspace()
  chatState.loadOutputFiles()
}

async function openFilePreview(item: CreatedFile) {
  previewOpen.value = true
  previewLoading.value = true
  previewError.value = ''
  selectedPreview.value = null
  try {
    selectedPreview.value = await readOutputFile(item.path, chatState.workspacePath.value ?? undefined)
  } catch (error) {
    previewError.value = getPreviewError(error)
  } finally {
    previewLoading.value = false
  }
}

function closeFilePreview() {
  previewOpen.value = false
  previewLoading.value = false
  previewError.value = ''
  selectedPreview.value = null
}

async function copyPreviewContent() {
  if (!selectedPreview.value) return
  await navigator.clipboard.writeText(selectedPreview.value.content)
}

function getPreviewError(error: unknown): string {
  if (typeof error === 'object' && error !== null && 'response' in error) {
    const response = (error as { response?: { data?: { detail?: unknown }; status?: number } }).response
    const detail = response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) return detail
    if (response?.status) return `读取失败：HTTP ${response.status}`
  }
  if (error instanceof Error && error.message) return error.message
  return '读取文件失败'
}
</script>
