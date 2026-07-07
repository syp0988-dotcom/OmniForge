<template>
  <div>
    <!-- Drop zone -->
    <div
      class="relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-150"
      :class="
        localDragOver
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50'
      "
      @click="fileInputRef?.click()"
      @dragover.prevent="localDragOver = true"
      @dragleave.prevent="localDragOver = false"
      @drop.prevent="onDrop"
    >
      <input
        ref="fileInputRef"
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.txt,.md,.markdown,.html,.htm,.xlsx,.xls,.pptx,.csv,.epub,.py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.c,.cpp,.h,.hpp,.zip"
        class="hidden"
        @change="onFileSelect"
      />
      <input
        ref="folderInputRef"
        type="file"
        class="hidden"
        webkitdirectory
        @change="onFileSelect"
      />

      <div class="flex flex-col items-center gap-2">
        <Upload class="w-8 h-8 text-secondary" />
        <div class="text-sm text-text">{{ uploading ? '上传中...' : '拖拽文件或文件夹到此处' }}</div>
        <div class="text-xs text-secondary">支持 PDF、DOCX、TXT、MD、HTML、XLSX、PPTX、CSV、EPUB、代码文件、ZIP 压缩包</div>
        <div class="flex gap-3 mt-2">
          <button
            type="button"
            class="px-4 py-1.5 rounded-lg bg-primary text-white text-sm"
            @click.stop="fileInputRef?.click()"
          >
            选择文件
          </button>
          <button
            type="button"
            class="px-4 py-1.5 rounded-lg border border-border text-text text-sm"
            @click.stop="folderInputRef?.click()"
          >
            选择文件夹
          </button>
        </div>
      </div>
    </div>

    <!-- Status -->
    <div
      v-if="status"
      class="mt-3 text-sm p-3 rounded-xl"
      :class="
        status.includes('成功')
          ? 'bg-primary/5 text-primary'
          : 'bg-hover text-secondary'
      "
    >
      {{ status }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, inject } from 'vue'
import { Upload } from 'lucide-vue-next'
import type { ChatState } from '@/composables/useChatState'

const props = defineProps<{
  uploading: boolean
  status: string | null
}>()

const emit = defineEmits<{
  upload: [files: File[]]
}>()

const chatState = inject<ChatState>('chatState')!

const localDragOver = ref(false)
const fileInputRef = ref<HTMLInputElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)

function onFileSelect(e: Event) {
  const target = e.target as HTMLInputElement
  if (!target.files?.length) return
  emit('upload', Array.from(target.files))
  target.value = ''
}

async function onDrop(e: DragEvent) {
  localDragOver.value = false
  if (!e.dataTransfer?.items?.length) return
  const files = await chatState.collectFilesFromDrop(e.dataTransfer.items)
  if (files.length > 0) emit('upload', files)
}
</script>
