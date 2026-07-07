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
          class="flex-1 flex items-center gap-2 px-4 py-2 rounded-xl border border-border bg-white transition-all duration-150 focus-within:border-primary focus-within:shadow-[0_0_0_1px_#E86A33]"
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
        <div
          v-for="r in chatState.searchResults.value"
          :key="r.chunk_id"
          class="p-4 rounded-xl border border-border bg-white space-y-1"
        >
          <div class="text-xs text-secondary">
            {{ r.filename }} · 相似度: {{ (r.score * 100).toFixed(0) }}%
          </div>
          <div class="text-sm text-text line-clamp-3">{{ r.content }}</div>
        </div>
      </div>
    </div>

    <!-- Document list -->
    <DocumentList />
  </div>
</template>

<script setup lang="ts">
import { inject, onMounted } from 'vue'
import { Search } from 'lucide-vue-next'
import UploadZone from '@/components/knowledge/UploadZone.vue'
import DocumentList from '@/components/knowledge/DocumentList.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

onMounted(() => {
  chatState.loadDocs()
})

function handleUpload(files: File[]) {
  chatState.uploadFiles(files)
}
</script>
