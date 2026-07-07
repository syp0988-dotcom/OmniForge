<template>
  <div class="flex-1 flex flex-col items-center justify-center px-6 py-12">
    <!-- Logo -->
    <img src="/images/logo.png" alt="OmniForge" class="w-12 h-12 mb-6" />

    <!-- Greeting -->
    <h1 class="font-serif text-[48px] text-text leading-tight mb-1">
      {{ greeting }}
    </h1>
    <p class="text-base text-secondary mb-8">今天有什么可以帮你的？</p>

    <!-- Input -->
    <ChatInput />

    <!-- Quick actions -->
    <QuickActions class="mt-8" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ChatInput from './ChatInput.vue'
import QuickActions from './QuickActions.vue'

const greeting = computed(() => {
  const hour = new Date().getHours()
  if (hour < 12) return 'Good Morning'
  if (hour < 18) return 'Good Afternoon'
  return 'Good Evening'
})
</script>
