<template>
  <div class="px-3 pb-3 pt-2 border-t border-border">
    <div
      class="relative flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors duration-150 ease-out hover:bg-hover"
      @click="open = !open"
    >
      <div
        class="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-medium flex-shrink-0"
      >
        {{ initials }}
      </div>
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium text-text truncate">开发者</div>
        <div class="text-xs text-secondary truncate">{{ modelName }}</div>
      </div>
      <ChevronDown
        class="w-3.5 h-3.5 text-secondary transition-transform duration-150"
        :class="{ 'rotate-180': open }"
      />
    </div>

    <!-- Dropdown -->
    <Transition name="dropdown">
      <div
        v-if="open"
        v-click-outside="() => (open = false)"
        class="absolute bottom-full mb-2 left-3 right-3 bg-white border border-border rounded-xl shadow-lg py-1 z-50"
      >
        <button
          v-for="item in menuItems"
          :key="item.label"
          class="w-full flex items-center gap-2 px-3 py-2 text-sm text-text hover:bg-hover transition-colors duration-150"
        >
          <component :is="item.icon" class="w-4 h-4 text-secondary" />
          {{ item.label }}
        </button>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ChevronDown, User, Settings, LogOut } from 'lucide-vue-next'
import { getModels } from '@/api/client'

const open = ref(false)
const initials = 'OF'
const modelName = ref('默认模型')

onMounted(async () => {
  try {
    const models = await getModels()
    const active = models.find((m) => m.is_active)
    if (active) modelName.value = active.name
  } catch { /* ignore */ }
})

const menuItems = [
  { icon: User, label: '个人资料' },
  { icon: Settings, label: '偏好设置' },
  { icon: LogOut, label: '退出登录' },
]

/* Simple click-outside directive */
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
