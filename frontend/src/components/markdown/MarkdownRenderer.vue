<template>
  <div class="markdown-body" ref="container" v-html="renderedHtml"></div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'

// Imported eagerly (synchronously): an async <script setup> (top-level await)
// returns a promise that cannot be rendered without a <Suspense> boundary,
// which silently prevents every AI reply from appearing. Bundle size is a
// reasonable trade-off for a chat reply renderer that must always display.
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js/lib/core'
import langPython from 'highlight.js/lib/languages/python'
import langJs from 'highlight.js/lib/languages/javascript'
import langTs from 'highlight.js/lib/languages/typescript'
import langJava from 'highlight.js/lib/languages/java'
import langGo from 'highlight.js/lib/languages/go'
import langBash from 'highlight.js/lib/languages/bash'
import langJson from 'highlight.js/lib/languages/json'
import langXml from 'highlight.js/lib/languages/xml'
import langCss from 'highlight.js/lib/languages/css'
import langSql from 'highlight.js/lib/languages/sql'
import langYaml from 'highlight.js/lib/languages/yaml'
import type { LanguageFn } from 'highlight.js'

const languages: Array<[string, LanguageFn]> = [
  ['python', langPython],
  ['javascript', langJs],
  ['typescript', langTs],
  ['java', langJava],
  ['go', langGo],
  ['bash', langBash],
  ['json', langJson],
  ['xml', langXml],
  ['css', langCss],
  ['sql', langSql],
  ['yaml', langYaml],
]
for (const [name, lang] of languages) {
  hljs.registerLanguage(name, lang)
}

const props = defineProps<{
  content: string
}>()

const container = ref<HTMLDivElement | null>(null)

/* ---- Markdown-It setup ---- */
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  breaks: true,
  highlight: (str: string, lang: string): string => {
    const language = lang && hljs.getLanguage(lang) ? lang : ''
    let highlighted: string
    if (language) {
      try {
        highlighted = hljs.highlight(str, {
          language,
          ignoreIllegals: true,
        }).value
      } catch {
        highlighted = md.utils.escapeHtml(str)
      }
    } else {
      highlighted = md.utils.escapeHtml(str)
    }
    const langLabel = language || 'text'
    return [
      '<div class="code-block">',
      '<div class="code-header">',
      `<span class="code-lang">${langLabel}</span>`,
      '<button class="copy-btn" data-code="' +
        md.utils.escapeHtml(str) +
        '">复制</button>',
      '</div>',
      '<div class="code-body"><pre><code class="hljs' +
        (language ? ` language-${language}` : '') +
        '">' +
        highlighted +
        '</code></pre></div>',
      '</div>',
    ].join('')
  },
})

const renderedHtml = computed(() => md.render(props.content))

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
