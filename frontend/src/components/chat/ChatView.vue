<template>
  <div class="flex-1 flex flex-col h-full">
    <!-- No messages → Welcome -->
    <template v-if="chatState.messages.value.length === 0">
      <WelcomeView />
    </template>

    <!-- Has messages → Chat -->
    <template v-else>
      <!-- Messages -->
      <div
        ref="scrollRef"
        class="flex-1 overflow-y-auto px-6"
      >
        <div class="max-w-[760px] mx-auto py-8 space-y-6">
          <MessageItem
            v-for="msg in chatState.messages.value"
            :key="msg.id"
            :msg="msg"
          />

          <!-- Task tree (execution plan) -->
          <TaskTree
            v-if="chatState.tasks.value.length > 0"
            :tasks="chatState.tasks.value"
          />

          <!-- Thinking indicator -->
          <ThinkingIndicator v-if="chatState.thinking.value" />
        </div>
      </div>

      <!-- Input (detached, at bottom) -->
      <div class="flex-shrink-0 px-6 pb-4 pt-2 bg-white">
        <ChatInput @send="onSend" />
      </div>
    </template>
  </div>
</template>

<script lang="ts">
export default { name: 'ChatView' }
</script>

<script setup lang="ts">
import { ref, inject, watch, nextTick, onActivated } from 'vue'
import WelcomeView from './WelcomeView.vue'
import ChatInput from './ChatInput.vue'
import MessageItem from './MessageItem.vue'
import TaskTree from './TaskTree.vue'
import ThinkingIndicator from './ThinkingIndicator.vue'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!
const scrollRef = ref<HTMLDivElement | null>(null)

function onSend(_text: string) {
  /* handled by composable */
}

/* Auto-scroll on new messages */
watch(
  () => chatState.messages.value.length,
  async () => {
    await nextTick()
    scrollRef.value?.scrollTo({
      top: scrollRef.value.scrollHeight,
      behavior: 'smooth',
    })
  },
)

/* Recover messages if state was lost during page switch / HMR */
onActivated(() => {
  chatState.recoverSessionMessages()
})
</script>
