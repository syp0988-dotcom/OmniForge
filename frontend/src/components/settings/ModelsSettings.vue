<template>
  <div class="max-w-content mx-auto px-6 py-8 space-y-6">
    <div>
      <h2 class="text-lg font-semibold text-text">设置</h2>
      <p class="text-sm text-secondary mt-1">调整界面显示、默认回答来源和模型配置。</p>
    </div>

    <div class="grid gap-4 md:grid-cols-2">
      <section class="rounded-xl border border-border bg-white p-5 space-y-4">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
            <Palette class="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 class="text-sm font-semibold text-text">外观</h3>
            <p class="text-xs text-secondary mt-0.5">主题、密度和动效会立即生效。</p>
          </div>
        </div>

        <div class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-secondary mb-2">主题</label>
            <div class="grid grid-cols-3 gap-2">
              <button
                v-for="theme in themes"
                :key="theme.value"
                class="rounded-lg border px-3 py-2 text-left transition-colors"
                :class="chatState.themeMode.value === theme.value ? 'border-primary bg-primary/5' : 'border-border hover:bg-hover'"
                @click="chatState.themeMode.value = theme.value"
              >
                <span class="block text-xs font-medium text-text">{{ theme.label }}</span>
                <span class="mt-2 flex gap-1">
                  <span
                    v-for="swatch in theme.swatches"
                    :key="swatch"
                    class="h-3 w-3 rounded-full border border-border"
                    :style="{ backgroundColor: swatch }"
                  />
                </span>
              </button>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-secondary mb-1.5">显示密度</label>
              <select v-model="chatState.densityMode.value" class="settings-select">
                <option value="comfortable">舒适</option>
                <option value="compact">紧凑</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-secondary mb-1.5">动效</label>
              <select v-model="chatState.motionMode.value" class="settings-select">
                <option value="full">完整动效</option>
                <option value="reduced">减少动效</option>
              </select>
            </div>
          </div>
        </div>
      </section>

      <section class="rounded-xl border border-border bg-white p-5 space-y-4">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
            <Languages class="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 class="text-sm font-semibold text-text">语言与回答</h3>
            <p class="text-xs text-secondary mt-0.5">设置界面语言和默认回答来源。</p>
          </div>
        </div>

        <div class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-secondary mb-1.5">界面语言</label>
            <select v-model="chatState.languageMode.value" class="settings-select">
              <option value="zh-CN">简体中文</option>
              <option value="en-US">English</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-secondary mb-2">默认回答来源</label>
            <div class="grid grid-cols-3 gap-2">
              <button
                v-for="mode in sourceModes"
                :key="mode.value"
                class="rounded-lg border px-3 py-2 text-left transition-colors"
                :class="chatState.sourceMode.value === mode.value ? 'border-primary bg-primary/5 text-primary' : 'border-border text-text hover:bg-hover'"
                @click="chatState.sourceMode.value = mode.value"
              >
                <component :is="mode.icon" class="w-4 h-4 mb-2" />
                <span class="block text-xs font-medium">{{ mode.label }}</span>
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-lg font-semibold text-text">模型管理</h2>
        <p class="text-sm text-secondary mt-1">配置和管理 LLM 模型，支持 DeepSeek、OpenAI 等兼容接口。</p>
      </div>
      <button
        class="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-hover transition-colors duration-150"
        @click="openForm()"
      >
        添加模型
      </button>
    </div>

    <!-- Model list -->
    <div class="grid gap-3">
      <div
        v-for="model in models"
        :key="model.id"
        class="flex items-center justify-between p-5 rounded-xl border border-border bg-white transition-colors duration-150 hover:bg-hover"
      >
        <div class="flex items-center gap-4">
          <div
            class="w-10 h-10 rounded-xl flex items-center justify-center text-white font-bold text-sm"
            :class="providerColor(model.provider)"
          >
            {{ providerInitials(model.provider) }}
          </div>
          <div>
            <div class="flex items-center gap-2">
              <span class="text-sm font-medium text-text">{{ model.name }}</span>
              <span
                v-if="model.is_active"
                class="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium"
              >Active</span>
            </div>
            <div class="text-xs text-secondary mt-0.5">
              {{ model.provider }} · {{ model.model_name }}
            </div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button
            v-if="!model.is_active"
            class="px-3 py-1.5 rounded-lg border border-border text-xs text-text hover:bg-hover transition-colors duration-150"
            @click="activate(model.id)"
          >
            启用
          </button>
          <button
            class="p-1.5 rounded-lg text-secondary hover:text-text hover:bg-hover transition-colors duration-150"
            title="编辑"
            @click="openForm(model)"
          >
            <Edit class="w-4 h-4" />
          </button>
          <button
            class="p-1.5 rounded-lg text-secondary hover:text-red-500 hover:bg-red-50 transition-colors duration-150"
            title="删除"
            @click="remove(model)"
          >
            <Trash2 class="w-4 h-4" />
          </button>
        </div>
      </div>

      <!-- Empty state -->
      <div
        v-if="models.length === 0"
        class="flex flex-col items-center justify-center py-16 text-center"
      >
        <Brain class="w-12 h-12 text-secondary/40 mb-4" />
        <p class="text-sm text-secondary">还没有配置模型</p>
        <p class="text-xs text-secondary/60 mt-1">添加你的第一个模型配置开始使用</p>
      </div>
    </div>

    <!-- Add/Edit Modal -->
    <Transition name="modal">
      <div
        v-if="showForm"
        class="fixed inset-0 z-50 flex items-center justify-center"
      >
        <div class="absolute inset-0 bg-black/30" @click="closeForm" />
        <div class="relative bg-white rounded-2xl shadow-xl w-full max-w-lg mx-4 p-6">
          <div class="flex items-center justify-between mb-5">
            <h3 class="text-base font-semibold text-text">
              {{ editingId ? '编辑模型' : '添加模型' }}
            </h3>
            <button class="text-secondary hover:text-text" @click="closeForm">
              <X class="w-5 h-5" />
            </button>
          </div>

          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-secondary mb-1">名称</label>
              <input
                v-model="form.name"
                class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150"
                placeholder="例如：我的 DeepSeek"
              />
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-secondary mb-1">提供商</label>
                <select
                  v-model="form.provider"
                  class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150 bg-white"
                >
                  <option value="deepseek">DeepSeek</option>
                  <option value="openai">OpenAI</option>
                  <option value="custom">自定义</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-secondary mb-1">模型名</label>
                <input
                  v-model="form.model_name"
                  class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150"
                  placeholder="deepseek-chat"
                />
              </div>
            </div>

            <div>
              <label class="block text-xs font-medium text-secondary mb-1">API Base URL</label>
              <input
                v-model="form.base_url"
                class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150"
                placeholder="https://api.deepseek.com"
              />
            </div>

            <div>
              <label class="block text-xs font-medium text-secondary mb-1">API Key</label>
              <input
                v-model="form.api_key"
                type="password"
                class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150"
                placeholder="sk-..."
              />
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-secondary mb-1">
                  Temperature ({{ form.temperature }})
                </label>
                <input
                  v-model.number="form.temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  class="w-full accent-primary"
                />
              </div>
              <div>
                <label class="block text-xs font-medium text-secondary mb-1">Max Tokens</label>
                <input
                  v-model.number="form.max_tokens"
                  type="number"
                  class="w-full px-3 py-2 rounded-lg border border-border text-sm text-text outline-none focus:border-primary transition-colors duration-150"
                />
              </div>
            </div>
          </div>

          <div class="flex justify-end gap-2 mt-6">
            <button
              class="px-4 py-2 rounded-lg border border-border text-sm text-text hover:bg-hover transition-colors duration-150"
              @click="closeForm"
            >
              取消
            </button>
            <button
              class="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-hover disabled:opacity-50 transition-colors duration-150"
              :disabled="!form.name || !form.base_url || !form.model_name"
              @click="save"
            >
              {{ editingId ? '保存' : '添加' }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, inject } from 'vue'
import { Edit, Trash2, Brain, X, Palette, Languages, Sparkles, Search, BookOpen } from 'lucide-vue-next'
import {
  getModels,
  createModel,
  updateModel,
  deleteModel,
  activateModel,
} from '@/api/client'
import type { ModelConfig, SourceMode } from '@/types'
import type { ChatState } from '@/composables/useChatState'

const chatState = inject<ChatState>('chatState')!
const models = ref<ModelConfig[]>([])
const showForm = ref(false)
const editingId = ref<number | null>(null)

const themes = [
  { value: 'light', label: '浅色', swatches: ['#FFFFFF', '#2563EB', '#1A1A1A'] },
  { value: 'dark', label: '深色', swatches: ['#121418', '#60A5FA', '#F1F5F9'] },
  { value: 'ocean', label: '海雾', swatches: ['#FAFDFF', '#0F768C', '#13262D'] },
] as const

const sourceModes: Array<{
  value: SourceMode
  label: string
  icon: typeof Sparkles
}> = [
  { value: 'auto', label: '自动', icon: Sparkles },
  { value: 'web', label: '联网', icon: Search },
  { value: 'knowledge', label: '知识库', icon: BookOpen },
]

const form = ref({
  name: '',
  provider: 'deepseek',
  base_url: 'https://api.deepseek.com',
  api_key: '',
  model_name: '',
  temperature: 0.7,
  max_tokens: 4096,
})

onMounted(load)

async function load() {
  try {
    models.value = await getModels()
  } catch {
    /* silently fail */
  }
}

function openForm(model?: ModelConfig) {
  if (model) {
    editingId.value = model.id
    form.value = {
      name: model.name,
      provider: model.provider,
      base_url: model.base_url,
      api_key: '',
      model_name: model.model_name,
      temperature: model.temperature,
      max_tokens: model.max_tokens,
    }
  } else {
    editingId.value = null
    form.value = {
      name: '',
      provider: 'deepseek',
      base_url: 'https://api.deepseek.com',
      api_key: '',
      model_name: '',
      temperature: 0.7,
      max_tokens: 4096,
    }
  }
  showForm.value = true
}

function closeForm() {
  showForm.value = false
  editingId.value = null
}

async function save() {
  try {
    if (editingId.value) {
      const data: Record<string, unknown> = {}
      if (form.value.name) data.name = form.value.name
      if (form.value.provider) data.provider = form.value.provider
      if (form.value.base_url) data.base_url = form.value.base_url
      if (form.value.api_key) data.api_key = form.value.api_key
      if (form.value.model_name) data.model_name = form.value.model_name
      data.temperature = form.value.temperature
      data.max_tokens = form.value.max_tokens
      await updateModel(editingId.value, data)
    } else {
      await createModel({ ...form.value })
    }
    closeForm()
    await load()
  } catch {
    /* silently fail */
  }
}

async function activate(id: number) {
  try {
    await activateModel(id)
    await load()
  } catch {
    /* silently fail */
  }
}

async function remove(model: ModelConfig) {
  if (!confirm(`确定删除 "${model.name}"？`)) return
  try {
    await deleteModel(model.id)
    await load()
  } catch {
    /* silently fail */
  }
}

function providerColor(provider: string): string {
  const map: Record<string, string> = {
    deepseek: 'bg-blue-500',
    openai: 'bg-emerald-500',
    custom: 'bg-purple-500',
  }
  return map[provider] || 'bg-gray-500'
}

function providerInitials(provider: string): string {
  const map: Record<string, string> = {
    deepseek: 'DS',
    openai: 'AI',
    custom: 'CU',
  }
  return map[provider] || 'LLM'
}
</script>

<style scoped>
.settings-select {
  @apply w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text outline-none transition-colors focus:border-primary;
}

.modal-enter-active,
.modal-leave-active {
  transition: opacity 200ms ease-out;
}
.modal-enter-active > div:last-child,
.modal-leave-active > div:last-child {
  transition: transform 200ms ease-out;
}
.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}
.modal-enter-from > div:last-child,
.modal-leave-to > div:last-child {
  transform: scale(0.95) translateY(8px);
}
</style>
