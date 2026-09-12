import { describe, expect, it } from 'vitest'

import { renderMarkdown } from './markdown'

/**
 * Security regression tests for security review H4 (stored XSS through
 * assistant replies rendered with `v-html`).
 */
describe('renderMarkdown', () => {
  it('escapes raw HTML instead of emitting it', () => {
    const html = renderMarkdown('<img src=x onerror="alert(1)">')

    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img')
  })

  it('escapes script tags and event handlers', () => {
    const html = renderMarkdown('<script>alert(1)</script>')

    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })

  it('refuses javascript: links', () => {
    const html = renderMarkdown('[click me](javascript:alert(1))')

    expect(html.toLowerCase()).not.toContain('href="javascript:')
  })

  it('still renders markdown features', () => {
    const html = renderMarkdown('# Title\n\n- item\n\n**bold**')

    expect(html).toContain('<h1>')
    expect(html).toContain('<li>')
    expect(html).toContain('<strong>bold</strong>')
  })

  it('highlights fenced code and escapes its content', () => {
    const html = renderMarkdown('```python\nprint("<b>hi</b>")\n```')

    expect(html).toContain('code-block')
    expect(html).toContain('language-python')
    expect(html).not.toContain('<b>hi</b>')
  })
})
