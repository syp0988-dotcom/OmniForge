<template>
  <aside
    class="w-[300px] flex-shrink-0 h-full flex flex-col bg-sidebar-bg border-r border-border select-none"
  >
    <!-- Logo -->
    <div class="px-5 pt-5 pb-4">
      <div class="flex items-center gap-3">
        <img src="/images/logo.png" alt="OmniForge" class="w-12 h-12" />
        <span class="text-base font-semibold text-text">OmniForge</span>
      </div>
    </div>

    <!-- Segmented Control -->
    <SegmentedControl />

    <!-- Nav items -->
    <nav class="px-3 mt-3 space-y-0.5">
      <NavItem
        v-for="item in navItems"
        :key="item.label"
        :icon="item.icon"
        :label="item.label"
        :active="item.active"
        @click="item.action?.()"
      />
    </nav>

    <!-- Recent Chats -->
    <ChatHistory />

    <!-- Spacer -->
    <div class="flex-1 min-h-0" />
  </aside>
</template>

<script setup lang="ts">
import { computed, inject } from 'vue'
import { Plus, MessageCircle, Folder, BookOpen, Settings } from 'lucide-vue-next'
import SegmentedControl from '@/components/sidebar/SegmentedControl.vue'
import NavItem from '@/components/sidebar/NavItem.vue'
import ChatHistory from '@/components/sidebar/ChatHistory.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!

const navItems = computed(() => [
  { icon: MessageCircle, label: '对话', active: chatState.activeSection.value === 'chat', action: () => { chatState.activeSection.value = 'chat' } },
  { icon: Plus, label: '新对话', active: false, action: () => { chatState.newChat(); chatState.activeSection.value = 'chat' } },
  { icon: Folder, label: '项目', active: chatState.activeSection.value === 'projects', action: () => { chatState.activeSection.value = 'projects' } },
  { icon: BookOpen, label: '知识库', active: chatState.activeSection.value === 'knowledge', action: () => { chatState.activeSection.value = 'knowledge' } },
  { icon: Settings, label: '设置', active: chatState.activeSection.value === 'settings', action: () => { chatState.activeSection.value = 'settings' } },
])
</script>
