<template>
  <div class="space-y-2">
    <div class="text-xs font-medium text-secondary">
      已索引文档 ({{ chatState.documents.value.length }})
    </div>

    <div
      v-if="chatState.documents.value.length === 0"
      class="text-sm text-secondary p-4 rounded-xl bg-hover"
    >
      暂无文档
    </div>

    <div
      v-for="doc in chatState.documents.value"
      :key="doc.id"
      class="group flex items-center justify-between p-4 rounded-xl border border-border bg-white transition-colors duration-150 hover:bg-hover"
    >
      <button
        type="button"
        class="min-w-0 flex-1 text-left"
        @click="$emit('preview', doc.id, doc.filename)"
      >
        <div class="text-sm font-medium text-text truncate">{{ doc.filename }}</div>
        <div class="text-xs text-secondary mt-0.5">
          {{ doc.file_type.toUpperCase() }} · {{ chatState.formatSize(doc.file_size) }} ·
          {{ chatState.formatDate(doc.created_at) }}
        </div>
      </button>
      <div class="flex items-center gap-2 ml-4 flex-shrink-0">
        <button
          class="p-1.5 rounded-lg text-secondary hover:text-primary hover:bg-primary/5 transition-colors opacity-0 group-hover:opacity-100"
          title="预览文件"
          @click="$emit('preview', doc.id, doc.filename)"
        >
          <Eye class="w-4 h-4" />
        </button>
        <button
          class="px-3 py-1.5 rounded-lg text-xs text-secondary border border-border hover:text-danger hover:border-danger/30 hover:bg-danger/5 transition-all duration-150"
          @click="chatState.handleDelete(doc.id, doc.filename)"
        >
          删除
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { inject } from 'vue'
import { Eye } from 'lucide-vue-next'
import type { ChatState } from '@/composables/useChatState'

defineEmits<{
  preview: [docId: number, filename: string]
}>()

const chatState = inject<ChatState>('chatState')!
</script>
