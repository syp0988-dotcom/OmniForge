<template>
  <div class="max-w-content mx-auto px-6 py-8 space-y-6">
    <div>
      <h2 class="text-lg font-semibold text-text">Artifacts</h2>
      <p class="text-sm text-secondary mt-1">查看和管理 Agent 生成的代码、文档和产物。</p>
    </div>

    <div v-if="displayItems.length > 0" class="grid gap-3">
      <div
        v-for="(item, idx) in displayItems"
        :key="item.title + '-' + idx"
        class="flex items-center justify-between p-5 rounded-xl border border-border bg-white transition-colors duration-150 hover:bg-hover"
      >
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 rounded-xl bg-code-bg flex items-center justify-center">
            <component :is="item.icon" class="w-5 h-5 text-secondary" />
          </div>
          <div>
            <div class="text-sm font-medium text-text">{{ item.title }}</div>
            <div class="text-xs text-secondary mt-0.5">
              {{ item.type }}<span v-if="item.size"> · {{ formatSize(item.size) }}</span> · {{ item.date }}
            </div>
          </div>
        </div>
        <span class="text-xs text-secondary font-mono">{{ item.path }}</span>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="text-center py-16 text-secondary text-sm">
      <FileIcon class="w-12 h-12 mx-auto mb-4 opacity-40" />
      <p>暂无 Agent 生成的文件</p>
      <p class="mt-1 text-xs">Agent 回复中的代码块可一键保存为文件</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, inject, onMounted } from 'vue'
import { Code, FileText, Brain, File as FileIcon } from 'lucide-vue-next'
import type { FunctionalComponent, SVGAttributes } from 'vue'
import type { ChatState } from '@/composables/useChatState'
import type { CreatedFile } from '@/types'

const chatState = inject<ChatState>('chatState')!

onMounted(() => {
  chatState.loadOutputFiles()
})

interface DisplayItem {
  icon: FunctionalComponent<SVGAttributes>
  title: string
  type: string
  size?: number
  date: string
  path: string
}

const staticDemoItems: DisplayItem[] = [
  { icon: Code, title: 'API 路由设计', type: '代码', date: '2026-07-04', path: 'outputs/api-routes.py' },
  { icon: FileText, title: '项目架构文档', type: '文档', date: '2026-07-03', path: 'outputs/architecture.md' },
  { icon: Brain, title: 'Agent 工作流分析', type: '报告', date: '2026-07-02', path: 'outputs/workflow-analysis.md' },
]

const displayItems = computed<DisplayItem[]>(() => {
  const files = chatState.outputFiles.value
  if (files.length === 0) return staticDemoItems
  return files.map((f: CreatedFile) => ({
    icon: FileIcon,
    title: f.filename,
    type: '生成文件',
    size: f.size,
    date: new Date(f.created_at).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }),
    path: f.path,
  }))
})

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
</script>
