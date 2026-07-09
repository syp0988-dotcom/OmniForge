<template>
  <div class="max-w-content mx-auto px-6 py-8 space-y-6">
    <div>
      <h2 class="text-lg font-semibold text-text">Agent 管理</h2>
      <p class="text-sm text-secondary mt-1">
        管理 OmniForge 中的 {{ chatState.agents.value.length }} 个智能 Agent，监控运行状态。
      </p>
    </div>

    <!-- Empty state when no agents loaded -->
    <div v-if="chatState.agents.value.length === 0" class="text-center py-16">
      <p class="text-secondary text-sm">Agent 列表为空</p>
      <p class="text-[#9B9B9B] text-xs mt-1">请检查后端 Agent 注册状态</p>
    </div>

    <div v-else class="grid gap-3">
      <div
        v-for="agent in chatState.agents.value"
        :key="agent.key"
        class="flex items-center justify-between p-5 rounded-xl border border-border bg-white transition-colors duration-150 hover:bg-hover"
      >
        <div class="flex items-center gap-4">
          <div
            class="w-10 h-10 rounded-xl flex items-center justify-center text-white font-bold text-sm"
            :style="{ backgroundColor: getColor(agent.key) }"
          >
            {{ getInitials(agent.name) }}
          </div>
          <div class="flex-1">
            <div class="flex items-center gap-2">
              <div class="text-sm font-medium text-text">{{ agent.name }}</div>
              <span
                class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                :class="agent.status === 'active'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-500'"
              >
                <span
                  class="w-1.5 h-1.5 rounded-full mr-1"
                  :class="agent.status === 'active' ? 'bg-green-500' : 'bg-gray-400'"
                />
                {{ agent.status === 'active' ? '活跃' : '未激活' }}
              </span>
            </div>
            <div class="text-xs text-secondary mt-0.5">{{ agent.description }}</div>
            <div class="flex gap-1 mt-1.5 flex-wrap">
              <span
                v-for="cap in agent.capabilities.slice(0, 3)"
                :key="cap"
                class="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600"
              >
                {{ cap }}
              </span>
              <span
                v-if="agent.capabilities.length > 3"
                class="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-400"
              >
                +{{ agent.capabilities.length - 3 }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { inject, onMounted } from 'vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

const AGENT_COLORS: Record<string, string> = {
  router: '#7C5CFC',
  planner: '#2563EB',
  knowledge: '#34C759',
  search: '#5AC8FA',
  answer: '#0F766E',
  memory: '#FF2D55',
  python: '#007AFF',
  report: '#8E8E93',
}

function getColor(key: string): string {
  return AGENT_COLORS[key] || '#8E8E93'
}

function getInitials(name: string): string {
  return name
    .split(/\s+/)
    .map((w) => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()
}

onMounted(() => {
  chatState.loadAgents()
})
</script>
