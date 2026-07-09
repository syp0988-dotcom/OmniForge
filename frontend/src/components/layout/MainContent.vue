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
import { computed, inject } from 'vue'
import ChatView from '@/components/chat/ChatView.vue'
import KnowledgeView from '@/components/knowledge/KnowledgeView.vue'
import ProjectsView from '@/components/projects/ProjectsView.vue'
import ArtifactsView from '@/components/ArtifactsView.vue'
import ModelsSettings from '@/components/settings/ModelsSettings.vue'
import FolderReminderModal from '@/components/projects/FolderReminderModal.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

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
