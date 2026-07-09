<template>
  <button
    class="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-text transition-colors duration-150 ease-out hover:bg-hover"
    :class="{ 'bg-hover': active }"
    @click="$emit('click')"
  >
    <component :is="icon" class="w-4 h-4 text-secondary flex-shrink-0" />
    <span>{{ label }}</span>
  </button>
</template>

<script setup lang="ts">
import type { Component } from 'vue'

defineProps<{
  icon: Component
  label: string
  active?: boolean
}>()

defineEmits<{
  click: []
}>()
</script>
