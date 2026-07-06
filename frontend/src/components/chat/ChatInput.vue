<template>
  <div class="w-full max-w-[900px] mx-auto">
    <div
      class="relative min-h-[140px] rounded-3xl border bg-white transition-all duration-150 ease-out"
      :class="[
        dragOver
          ? 'border-dashed border-primary bg-primary/5'
          : focused
            ? 'border-primary shadow-[0_0_0_1px_#E86A33]'
            : 'border-[#EAEAEA] hover:shadow-[0_2px_8px_rgba(0,0,0,0.04)]',
      ]"
      @dragover.prevent="onDragOver"
      @dragleave.prevent="onDragLeave"
      @drop.prevent="onDrop"
    >
      <!-- Textarea -->
      <textarea
        ref="textareaRef"
        v-model="text"
        placeholder="向 OmniForge 提问..."
        class="w-full h-[88px] resize-none bg-transparent px-5 pt-5 text-text placeholder:text-[#9B9B9B] outline-none text-sm leading-relaxed"
        @keydown="handleKeydown"
        @focus="focused = true"
        @blur="focused = false"
      ></textarea>

      <!-- Bottom bar -->
      <div class="flex items-center justify-between px-4 pb-4">
        <div class="flex items-center gap-1">
          <!-- Upload button -->
          <UploadButton @upload="handleUpload" />
        </div>

        <div class="flex items-center gap-2">
          <!-- Model selector -->
          <ModelSelector />
          <!-- Agent mode toggle -->
          <button
            class="px-3 py-1.5 text-xs font-medium rounded-full border border-border text-secondary hover:text-text hover:bg-hover transition-all duration-150"
          >
            Agent 模式
          </button>
          <!-- Voice (placeholder) -->
          <button
            class="w-8 h-8 rounded-full flex items-center justify-center text-secondary hover:text-text hover:bg-hover transition-all duration-150"
            title="语音输入（即将推出）"
          >
            <Mic class="w-4 h-4" />
          </button>
          <!-- Send / Stop -->
          <button
            v-if="!chatState.thinking.value"
            :disabled="!text.trim()"
            class="w-8 h-8 rounded-full flex items-center justify-center transition-all duration-150"
            :class="
              text.trim()
                ? 'bg-primary text-white hover:bg-primary-hover'
                : 'bg-hover text-secondary'
            "
            @click="sendMessage"
          >
            <ArrowUp class="w-4 h-4" />
          </button>
          <button
            v-else
            class="w-8 h-8 rounded-full flex items-center justify-center bg-red-500 text-white hover:bg-red-600 transition-all duration-150 animate-pulse"
            title="中断生成 (ESC)"
            @click="chatState.stopChat()"
          >
            <Square class="w-3.5 h-3.5 fill-current" />
          </button>
        </div>
      </div>
    </div>

    <!-- Upload status -->
    <div
      v-if="chatState.uploadStatus.value"
      class="mt-2 text-xs text-center"
      :class="
        chatState.uploadStatus.value.includes('成功')
          ? 'text-primary'
          : 'text-secondary'
      "
    >
      {{ chatState.uploadStatus.value }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, onMounted, onUnmounted } from 'vue'
import { Mic, ArrowUp, Square } from 'lucide-vue-next'
import UploadButton from './UploadButton.vue'
import ModelSelector from './ModelSelector.vue'
import type { ChatState } from '@/composables/useChatState'

const emit = defineEmits<{
  send: [text: string]
}>()

const chatState = inject<ChatState>('chatState')!

const text = ref('')
const focused = ref(false)
const dragOver = ref(false)
const textareaRef = ref<HTMLTextAreaElement | null>(null)

/* ---- Keybindings ---- */
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault()
    sendMessage()
    return
  }
  if (e.key === 'Enter' && e.shiftKey) {
    // Shift+Enter → newline (default behavior)
    return
  }
  if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
    e.preventDefault()
    sendMessage()
  }
}

function sendMessage() {
  const txt = text.value.trim()
  if (!txt) return
  emit('send', txt)
  chatState.handleSend(txt)
  text.value = ''
}

/* ---- Drag-and-drop upload ---- */
function onDragOver() {
  dragOver.value = true
}
function onDragLeave() {
  dragOver.value = false
}
async function onDrop(e: DragEvent) {
  dragOver.value = false
  if (!e.dataTransfer?.items?.length) return
  const files = await chatState.collectFilesFromDrop(e.dataTransfer.items)
  if (files.length > 0) {
    await chatState.uploadFiles(files)
  }
}

function handleUpload(files: File[]) {
  chatState.uploadFiles(files)
}

/* ---- Global ESC to stop ---- */
function onGlobalKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && chatState.thinking.value) {
    e.preventDefault()
    chatState.stopChat()
  }
}

onMounted(() => window.addEventListener('keydown', onGlobalKeydown))
onUnmounted(() => window.removeEventListener('keydown', onGlobalKeydown))

defineExpose({ focus: () => textareaRef.value?.focus() })
</script>
