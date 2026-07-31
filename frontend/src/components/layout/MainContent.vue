<template>
  <main class="flex-1 flex flex-col h-full min-w-0">
    <!-- Content area -->
    <div class="flex-1 overflow-y-auto">
      <KeepAlive>
        <component :is="currentView" />
      </KeepAlive>
    </div>
    <!-- Folder reminder modal (global, shown from any section) -->
    <FolderReminderModal />
  </main>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent, inject } from 'vue'
import ChatView from '@/components/chat/ChatView.vue'
import FolderReminderModal from '@/components/projects/FolderReminderModal.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

// Non-chat views are loaded on demand so the initial bundle stays small.
const KnowledgeView = defineAsyncComponent(() => import('@/components/knowledge/KnowledgeView.vue'))
const ProjectsView = defineAsyncComponent(() => import('@/components/projects/ProjectsView.vue'))
const ArtifactsView = defineAsyncComponent(() => import('@/components/ArtifactsView.vue'))
const ModelsSettings = defineAsyncComponent(() => import('@/components/settings/ModelsSettings.vue'))

const sectionMap: Record<string, any> = {
  chat: ChatView,
  knowledge: KnowledgeView,
  projects: ProjectsView,
  artifacts: ArtifactsView,
  settings: ModelsSettings,
}

const currentView = computed(() => {
  return sectionMap[chatState.activeSection.value] || ChatView
})
</script>
