<template>
  <Teleport to="body">
    <div
      v-if="chatState.showFolderReminder.value"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      @click.self="dismiss"
    >
      <div class="w-[440px] bg-white rounded-2xl shadow-xl p-6 animate-fade-in">
        <!-- Icon -->
        <div class="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
          <FolderOpen class="w-6 h-6 text-primary" />
        </div>

        <h3 class="text-lg font-semibold text-text text-center mb-2">未设置工作文件夹</h3>
        <p class="text-sm text-secondary text-center mb-6">
          创建文件前需要先选择一个工作文件夹。请导入已有文件夹或创建新文件夹来存放生成的文件。
        </p>

        <div class="space-y-2">
          <button
            class="w-full px-4 py-2.5 rounded-xl bg-primary text-white text-sm font-medium hover:bg-primary-hover transition-colors flex items-center justify-center gap-2"
            @click="goToProjects"
          >
            <FolderInput class="w-4 h-4" />
            选择或创建文件夹
          </button>
          <button
            class="w-full px-4 py-2.5 rounded-xl border border-border text-sm text-secondary hover:bg-hover transition-colors"
            @click="dismiss"
          >
            稍后提醒
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { inject } from 'vue'
import { FolderOpen, FolderInput } from 'lucide-vue-next'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

function dismiss() {
  chatState.showFolderReminder.value = false
}

function goToProjects() {
  chatState.showFolderReminder.value = false
  chatState.activeSection.value = 'projects'
}
</script>
