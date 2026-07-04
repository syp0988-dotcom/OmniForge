<template>
  <div class="relative">
    <button
      class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full border border-border text-secondary hover:text-text hover:bg-hover transition-all duration-150"
      @click="open = !open"
    >
      <Zap class="w-3.5 h-3.5" />
      <span>{{ currentModel }}</span>
      <ChevronDown
        class="w-3 h-3 transition-transform duration-150"
        :class="{ 'rotate-180': open }"
      />
    </button>

    <Transition name="dropdown">
      <div
        v-if="open"
        v-click-outside="() => (open = false)"
        class="absolute right-0 top-full mt-1 w-44 bg-white border border-border rounded-xl shadow-lg py-1 z-50"
      >
        <button
          v-for="model in models"
          :key="model.id"
          class="w-full flex items-center gap-2 px-3 py-2 text-sm text-text hover:bg-hover transition-colors duration-150"
          :class="{ 'bg-hover font-medium': model.name === currentModel }"
          @click="selectModel(model)"
        >
          <Zap class="w-3.5 h-3.5 text-secondary" />
          <span class="flex-1 text-left truncate">{{ model.name }}</span>
          <span
            v-if="model.is_active"
            class="w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0"
          />
        </button>
        <!-- Empty state -->
        <div
          v-if="models.length === 0"
          class="px-3 py-2 text-xs text-secondary text-center"
        >
          暂无模型配置
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Zap, ChevronDown } from 'lucide-vue-next'
import { getModels, activateModel } from '@/api/client'
import type { ModelConfig } from '@/types'

const open = ref(false)
const currentModel = ref('加载中...')
const models = ref<ModelConfig[]>([])

onMounted(async () => {
  try {
    models.value = await getModels()
    const active = models.value.find((m) => m.is_active)
    if (active) currentModel.value = active.name
    else if (models.value.length > 0) currentModel.value = models.value[0].name
    else currentModel.value = '默认模型'
  } catch {
    currentModel.value = '默认模型'
  }
})

async function selectModel(model: ModelConfig) {
  currentModel.value = model.name
  open.value = false
  if (!model.is_active) {
    try {
      await activateModel(model.id)
      models.value = await getModels()
    } catch { /* ignore */ }
  }
}

const vClickOutside = {
  mounted(el: HTMLElement, binding: { value: () => void }) {
    el.__clickOutside = (e: MouseEvent) => {
      if (!el.contains(e.target as Node)) binding.value()
    }
    document.addEventListener('click', el.__clickOutside)
  },
  unmounted(el: HTMLElement) {
    document.removeEventListener('click', el.__clickOutside)
  },
}
</script>

<style scoped>
.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 150ms ease-out;
}
.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
