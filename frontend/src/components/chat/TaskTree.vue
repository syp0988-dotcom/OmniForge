<template>
  <div class="task-tree border border-gray-200 rounded-xl overflow-hidden bg-white">
    <!-- Header -->
    <div class="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border-b border-gray-200">
      <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
      <span class="text-xs font-semibold text-gray-600 uppercase tracking-wider">执行计划</span>
      <span class="text-xs text-gray-400 ml-auto">
        {{ doneCount }}/{{ tasks.length }} 完成
      </span>
    </div>

    <!-- Task list -->
    <div class="divide-y divide-gray-100">
      <div
        v-for="task in tasks"
        :key="task.id"
        class="flex items-center gap-3 px-4 py-2.5 transition-colors"
        :class="rowClass(task)"
      >
        <!-- Status icon -->
        <span class="flex-shrink-0 w-5 h-5 flex items-center justify-center">
          <!-- todo -->
          <svg v-if="task.status === 'todo'" class="w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="9" stroke-width="2" />
          </svg>
          <!-- running -->
          <svg v-else-if="task.status === 'running'" class="w-4 h-4 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <!-- done -->
          <svg v-else-if="task.status === 'done'" class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
          </svg>
          <!-- failed -->
          <svg v-else-if="task.status === 'failed'" class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12" />
          </svg>
          <!-- blocked -->
          <svg v-else class="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </span>

        <!-- Tool badge -->
        <span
          class="flex-shrink-0 text-[10px] font-mono font-medium px-1.5 py-0.5 rounded border"
          :class="toolBadgeClass(task.tool)"
        >
          {{ task.tool }}
        </span>

        <!-- Title -->
        <span class="text-sm flex-1 min-w-0 truncate" :class="titleClass(task)">
          {{ task.title }}
        </span>

        <!-- Status text -->
        <span class="flex-shrink-0 text-[11px] font-medium" :class="statusTextClass(task)">
          {{ statusLabel(task) }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ExecutionTask } from '@/types'

const props = defineProps<{
  tasks: ExecutionTask[]
}>()

const doneCount = computed(() =>
  props.tasks.filter((t) => t.status === 'done' || t.status === 'skipped').length
)

function statusLabel(task: ExecutionTask): string {
  const map: Record<string, string> = {
    todo: '待执行',
    running: '执行中',
    done: '已完成',
    failed: '失败',
    blocked: '阻塞',
    skipped: '已跳过',
  }
  return map[task.status] || task.status
}

function rowClass(task: ExecutionTask): string {
  if (task.status === 'running') return 'bg-blue-50/60'
  if (task.status === 'done') return 'bg-green-50/40'
  if (task.status === 'failed') return 'bg-red-50/40'
  return ''
}

function toolBadgeClass(tool: string): string {
  const map: Record<string, string> = {
    filesystem: 'border-blue-200 text-blue-600 bg-blue-50',
    search: 'border-purple-200 text-purple-600 bg-purple-50',
    python: 'border-indigo-200 text-indigo-600 bg-indigo-50',
    git: 'border-gray-300 text-gray-600 bg-gray-50',
    browser: 'border-teal-200 text-teal-600 bg-teal-50',
    database: 'border-cyan-200 text-cyan-600 bg-cyan-50',
    mcp: 'border-pink-200 text-pink-600 bg-pink-50',
    composio: 'border-indigo-200 text-indigo-600 bg-indigo-50',
  }
  return map[tool] || 'border-gray-200 text-gray-500 bg-gray-50'
}

function titleClass(task: ExecutionTask): string {
  if (task.status === 'done' || task.status === 'skipped') return 'text-gray-400 line-through'
  if (task.status === 'failed') return 'text-red-700'
  if (task.status === 'running') return 'text-blue-700 font-medium'
  return 'text-gray-600'
}

function statusTextClass(task: ExecutionTask): string {
  const map: Record<string, string> = {
    running: 'text-blue-600',
    done: 'text-green-600',
    failed: 'text-red-600',
    blocked: 'text-yellow-600',
  }
  return map[task.status] || 'text-gray-400'
}
</script>

<style scoped>
@keyframes fade-in {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}
.task-tree {
  animation: fade-in 0.2s ease-out;
}
</style>
