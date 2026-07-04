<template>
  <div class="border border-border rounded-xl p-4 bg-white shadow-sm">
    <!-- Header row -->
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-2 min-w-0">
        <FileIcon class="w-5 h-5 text-secondary shrink-0" />
        <span class="font-mono text-sm font-medium text-text truncate">{{ proposal.filename }}</span>
        <span class="text-[11px] px-1.5 py-0.5 rounded-full bg-code-bg text-secondary shrink-0">
          {{ proposal.language || 'text' }}
        </span>
      </div>
    </div>

    <!-- Preview -->
    <pre class="mt-3 p-3 rounded-lg bg-code-bg text-xs leading-relaxed overflow-x-auto max-h-28 text-text/80"><code>{{ proposal.preview }}</code></pre>

    <!-- Pending state: action buttons -->
    <div v-if="status === 'pending'" class="flex gap-2 mt-3">
      <button
        :disabled="loading"
        class="px-4 py-1.5 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        @click="$emit('create', proposal)"
      >
        <span v-if="loading" class="flex items-center gap-1.5">
          <svg class="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          创建中...
        </span>
        <span v-else>创建</span>
      </button>
      <button
        class="px-4 py-1.5 rounded-lg border border-border text-sm text-secondary hover:bg-hover transition-colors"
        @click="$emit('dismiss', proposal.suggestion_id)"
      >
        忽略
      </button>
    </div>

    <!-- Created state -->
    <div v-else-if="status === 'created'" class="flex items-center gap-1.5 mt-3 text-sm text-green-600">
      <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      已创建: outputs/{{ proposal.filename }}
    </div>

    <!-- Dismissed state -->
    <div v-else-if="status === 'dismissed'" class="mt-3 text-xs text-secondary italic">
      已忽略
    </div>
  </div>
</template>

<script setup lang="ts">
import { File as FileIcon } from 'lucide-vue-next'
import type { FileProposal } from '@/types'

defineProps<{
  proposal: FileProposal
  status: 'pending' | 'created' | 'dismissed'
  loading: boolean
}>()

defineEmits<{
  create: [proposal: FileProposal]
  dismiss: [suggestionId: string]
}>()
</script>
