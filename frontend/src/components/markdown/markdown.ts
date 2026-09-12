/**
 * Markdown renderer used for assistant replies.
 *
 * Kept as a plain module (instead of living inside the .vue component) so the
 * security-relevant configuration can be unit-tested without a DOM.
 *
 * Security: `html: false` is deliberate. Assistant output is untrusted input —
 * markdown-it escapes raw HTML instead of emitting it, and `validateLink`
 * (on by default) refuses `javascript:` / `data:` URLs, so a reply containing
 * `<img src=x onerror=alert(1)>` or `[x](javascript:alert(1))` cannot execute
 * script (security review H4).
 */

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

export function registerLanguages(): void {
  for (const [name, lang] of languages) {
    if (!hljs.getLanguage(name)) hljs.registerLanguage(name, lang)
  }
}

function highlightCode(md: MarkdownIt, str: string, lang: string): string {
  registerLanguages()
  const language = lang && hljs.getLanguage(lang) ? lang : ''
  let highlighted: string
  if (language) {
    try {
      highlighted = hljs.highlight(str, { language, ignoreIllegals: true }).value
    } catch {
      highlighted = md.utils.escapeHtml(str)
    }
  } else {
    highlighted = md.utils.escapeHtml(str)
  }
  const langLabel = md.utils.escapeHtml(language || 'text')
  return [
    '<div class="code-block">',
    '<div class="code-header">',
    `<span class="code-lang">${langLabel}</span>`,
    '<button class="copy-btn" data-code="' + md.utils.escapeHtml(str) + '">复制</button>',
    '</div>',
    '<div class="code-body"><pre><code class="hljs' +
      (language ? ` language-${language}` : '') +
      '">' +
      highlighted +
      '</code></pre></div>',
    '</div>',
  ].join('')
}

export function createMarkdownRenderer(): MarkdownIt {
  registerLanguages()
  const instance = new MarkdownIt({
    html: false,
    linkify: true,
    typographer: true,
    breaks: true,
  })
  // Assigned after construction so the highlighter can fall back to the
  // instance's own escaping helper.
  instance.options.highlight = (str: string, lang: string): string =>
    highlightCode(instance, str, lang)
  return instance
}

const md = createMarkdownRenderer()

export function renderMarkdown(content: string): string {
  return md.render(content)
}

export default md
