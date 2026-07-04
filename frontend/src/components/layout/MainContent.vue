<template>
  <main class="flex-1 flex flex-col h-full min-w-0">
    <!-- Content area -->
    <div class="flex-1 overflow-y-auto">
      <ChatView v-if="chatState.activeSection.value === 'chat'" />
      <KnowledgeView v-else-if="chatState.activeSection.value === 'knowledge'" />
      <AgentsView v-else-if="chatState.activeSection.value === 'agents'" />
      <ArtifactsView v-else-if="chatState.activeSection.value === 'artifacts'" />
      <ModelsSettings v-else-if="chatState.activeSection.value === 'settings'" />
    </div>
  </main>
</template>

<script setup lang="ts">
import { inject } from 'vue'
import ChatView from '@/components/chat/ChatView.vue'
import KnowledgeView from '@/components/knowledge/KnowledgeView.vue'
import AgentsView from '@/components/AgentsView.vue'
import ArtifactsView from '@/components/ArtifactsView.vue'
import ModelsSettings from '@/components/settings/ModelsSettings.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!
</script>
