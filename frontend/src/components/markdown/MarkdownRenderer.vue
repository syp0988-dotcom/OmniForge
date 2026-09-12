<template>
  <div class="markdown-body" ref="container" v-html="renderedHtml"></div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'

// The renderer lives in ./markdown.ts so its security-relevant configuration
// (html: false, link validation) can be unit-tested without a DOM.
import { renderMarkdown } from './markdown'

const props = defineProps<{
  content: string
}>()

const container = ref<HTMLDivElement | null>(null)

const renderedHtml = computed(() => renderMarkdown(props.content))

/* ---- Wire up copy buttons after render ---- */
watch(
  renderedHtml,
  () => {
    nextTick(() => {
      if (!container.value) return
      container.value.querySelectorAll('.copy-btn').forEach((el) => {
        // Avoid duplicate listeners
        if ((el as HTMLElement).dataset._listener) return
        ;(el as HTMLElement).dataset._listener = '1'
        el.addEventListener('click', () => {
          const code = el.getAttribute('data-code') || ''
          navigator.clipboard.writeText(code).then(() => {
            el.textContent = '已复制'
            setTimeout(() => {
              el.textContent = '复制'
            }, 2000)
          })
        })
      })
    })
  },
  { immediate: true },
)
</script>
