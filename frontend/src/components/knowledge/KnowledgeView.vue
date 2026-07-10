<template>
  <div class="max-w-content mx-auto px-6 py-8 space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-lg font-semibold text-text">知识库</h2>
      <p class="text-sm text-secondary mt-1">
        上传文档（PDF、DOCX、TXT、MD、HTML、XLSX、PPTX、CSV、EPUB、代码文件），系统将自动解析、分块并向量化存储。
      </p>
    </div>

    <!-- Upload Zone -->
    <UploadZone
      :uploading="chatState.uploading.value"
      :status="chatState.uploadStatus.value"
      @upload="handleUpload"
    />

    <!-- Search -->
    <div>
      <div class="flex gap-2">
        <div
          class="flex-1 flex items-center gap-2 px-4 py-2 rounded-xl border border-border bg-white transition-all duration-150 focus-within:border-primary focus-within:shadow-[0_0_0_1px_rgb(var(--color-primary))]"
        >
          <Search class="w-4 h-4 text-secondary flex-shrink-0" />
          <input
            v-model="chatState.searchQuery.value"
            type="text"
            placeholder="搜索知识库..."
            class="flex-1 bg-transparent text-sm text-text placeholder:text-secondary outline-none"
            @keydown.enter="chatState.handleSearch()"
          />
        </div>
        <button
          class="px-5 py-2 rounded-xl bg-primary text-white text-sm font-medium hover:bg-primary-hover transition-colors duration-150"
          @click="chatState.handleSearch()"
        >
          搜索
        </button>
      </div>

      <!-- Search results -->
      <div v-if="chatState.searchResults.value !== null" class="mt-4 space-y-2">
        <div class="text-xs font-medium text-secondary">
          搜索结果 ({{ chatState.searchResults.value.length }} 条)
        </div>
        <div
          v-if="chatState.searchResults.value.length === 0"
          class="text-sm text-secondary p-4 rounded-xl bg-hover"
        >
          未找到相关结果
        </div>
        <button
          v-for="r in chatState.searchResults.value"
          :key="r.chunk_id"
          type="button"
          class="w-full text-left p-4 rounded-xl border border-border bg-white space-y-1 transition-colors hover:bg-hover focus:outline-none focus:border-primary"
          @click="openDocPreview(r.document_id, r.filename)"
        >
          <div class="text-xs text-secondary flex items-center gap-2">
            <span>{{ r.filename }}</span>
            <span>·</span>
            <span>相似度: {{ (r.score * 100).toFixed(0) }}%</span>
          </div>
          <div class="text-sm text-text line-clamp-3">{{ r.content }}</div>
        </button>
      </div>
    </div>

    <!-- Document list -->
    <DocumentList @preview="openDocPreview" />

    <!-- File preview modal -->
    <Teleport to="body">
      <div
        v-if="previewOpen"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm px-4"
        @click.self="closePreview"
      >
        <div class="w-full max-w-4xl max-h-[84vh] rounded-xl border border-border bg-white shadow-xl flex flex-col">
          <div class="flex items-center justify-between gap-4 px-5 py-4 border-b border-border">
            <div class="min-w-0">
              <div class="text-sm font-semibold text-text truncate">{{ previewFilename }}</div>
              <div class="text-xs text-secondary font-mono truncate mt-0.5">{{ previewPath }}</div>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <button
                v-if="previewContent"
                class="px-3 py-1.5 rounded-lg border border-border text-xs text-secondary hover:bg-hover transition-colors"
                @click="copyPreview"
              >
                复制内容
              </button>
              <button class="p-2 rounded-lg text-secondary hover:bg-hover hover:text-text transition-colors" @click="closePreview">
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
          <div v-else-if="previewContent" class="min-h-0 flex-1 overflow-auto">
            <div
              v-if="previewTruncated"
              class="px-5 py-2 border-b border-border bg-primary/5 text-xs text-primary"
            >
              文件较大，仅显示前 1 MB 内容。
            </div>
            <pre class="p-5 text-sm leading-relaxed font-mono whitespace-pre-wrap text-text">{{ previewContent }}</pre>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, onMounted } from 'vue'
import { Search, X } from 'lucide-vue-next'
import UploadZone from '@/components/knowledge/UploadZone.vue'
import DocumentList from '@/components/knowledge/DocumentList.vue'
import { readKnowledgeDocument } from '@/api/client'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

const previewOpen = ref(false)
const previewLoading = ref(false)
const previewError = ref('')
const previewFilename = ref('')
const previewPath = ref('')
const previewContent = ref('')
const previewTruncated = ref(false)

onMounted(() => {
  chatState.loadDocs()
})

function handleUpload(files: File[]) {
  chatState.uploadFiles(files)
}

async function openDocPreview(docId: number, filename: string) {
  previewOpen.value = true
  previewLoading.value = true
  previewError.value = ''
  previewFilename.value = filename
  previewPath.value = ''
  previewContent.value = ''
  previewTruncated.value = false

  try {
    const data = await readKnowledgeDocument(docId)
    previewFilename.value = data.filename
    previewPath.value = data.path
    previewContent.value = data.content
    previewTruncated.value = data.truncated
  } catch (e: any) {
    previewError.value = e?.response?.data?.detail || e?.message || '读取文件失败'
  } finally {
    previewLoading.value = false
  }
}

function closePreview() {
  previewOpen.value = false
  previewContent.value = ''
  previewPath.value = ''
  previewFilename.value = ''
  previewError.value = ''
}

async function copyPreview() {
  if (!previewContent.value) return
  try {
    await navigator.clipboard.writeText(previewContent.value)
  } catch {
    // fallback silently
  }
}
</script>
