<template>
  <!-- User message (right-aligned) -->
  <div v-if="msg.role === 'user'" class="flex justify-end">
    <div
      class="max-w-[760px] bg-[#F4F4F5] rounded-2xl px-5 py-3 animate-fade-in"
    >
      <div class="text-sm text-text leading-relaxed whitespace-pre-wrap">{{ msg.text }}</div>
    </div>
  </div>

  <!-- Agent message (left-aligned) -->
  <div v-else class="flex justify-start">
    <div class="max-w-[760px] w-full animate-slide-up">
      <!-- Header -->
      <div class="flex items-center gap-2 mb-3">
        <img
          :src="currentAvatar"
          alt="AI"
          class="w-12 h-12 rounded-md flex-shrink-0"
        />
        <span class="text-xs font-medium text-secondary">OmniForge</span>
      </div>
      <!-- Content -->
      <MarkdownRenderer :content="msg.text" />

      <!-- File proposal cards -->
      <div v-if="msg.proposals && msg.proposals.length > 0" class="mt-4 space-y-3">
        <FileProposalCard
          v-for="proposal in msg.proposals"
          :key="proposal.suggestion_id"
          :proposal="proposal"
          :status="getProposalStatus(proposal.suggestion_id)"
          :loading="loadingProposal === proposal.suggestion_id"
          @create="handleCreate"
          @dismiss="handleDismiss"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { inject, ref } from 'vue'
import type { Msg, FileProposal } from '@/types'
import type { ChatState } from '@/composables/useChatState'
import MarkdownRenderer from '@/components/markdown/MarkdownRenderer.vue'
import FileProposalCard from './FileProposalCard.vue'

const props = defineProps<{
  msg: Msg
}>()

const chatState = inject<ChatState>('chatState')!
const loadingProposal = ref<string | null>(null)

const currentAvatar = ref('/images/avatar-1.png')

function getProposalStatus(suggestionId: string): 'pending' | 'created' | 'dismissed' {
  return chatState.fileProposalStatuses.value[suggestionId] || 'pending'
}

async function handleCreate(proposal: FileProposal) {
  loadingProposal.value = proposal.suggestion_id
  await chatState.createOutputFile(proposal)
  loadingProposal.value = null
}

function handleDismiss(suggestionId: string) {
  chatState.dismissProposal(suggestionId)
}
</script>
