<template>
  <div class="markdown-body" ref="container" v-html="renderedHtml"></div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'

// Load the heavy markdown/highlighting stack on demand (first render),
// keeping it out of the initial page bundle.
const { default: MarkdownIt } = await import('markdown-it')
const { default: hljs } = await import('highlight.js/lib/core')
// Register only the languages most likely to appear in generated code.
const { default: langPython } = await import('highlight.js/lib/languages/python')
const { default: langJs } = await import('highlight.js/lib/languages/javascript')
const { default: langTs } = await import('highlight.js/lib/languages/typescript')
const { default: langJava } = await import('highlight.js/lib/languages/java')
const { default: langGo } = await import('highlight.js/lib/languages/go')
const { default: langBash } = await import('highlight.js/lib/languages/bash')
const { default: langJson } = await import('highlight.js/lib/languages/json')
const { default: langXml } = await import('highlight.js/lib/languages/xml')
const { default: langCss } = await import('highlight.js/lib/languages/css')
const { default: langSql } = await import('highlight.js/lib/languages/sql')
import type { LanguageFn } from 'highlight.js'
const { default: langYaml } = await import('highlight.js/lib/languages/yaml')
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
