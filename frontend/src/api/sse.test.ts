import { describe, expect, it } from 'vitest'

import { readSseStream } from './sse'

async function readerFromText(text: string): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const encoder = new TextEncoder()
  const bytes = encoder.encode(text)
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // Split the payload into awkward chunks to exercise incremental parsing.
      const chunkSize = 7
      for (let i = 0; i < bytes.length; i += chunkSize) {
        controller.enqueue(bytes.slice(i, i + chunkSize))
      }
      controller.close()
    },
  })
  return stream.getReader()
}

describe('readSseStream', () => {
  it('parses events split across chunks', async () => {
    const payload = [
      'event: thinking',
      'data: {"phase": "分析问题"}',
      '',
      'event: done',
      'data: {"answer": "你好", "session_id": 7}',
      '',
      '',
    ].join('\n')

    const events: Array<[string, Record<string, unknown>]> = []
    await readSseStream(await readerFromText(payload), (event, data) => {
      events.push([event, data])
    })

    expect(events).toEqual([
      ['thinking', { phase: '分析问题' }],
      ['done', { answer: '你好', session_id: 7 }],
    ])
  })

  it('ignores malformed JSON but recovers on later events', async () => {
    const payload = [
      'event: broken',
      'data: {not json',
      '',
      'event: done',
      'data: {"answer": "ok"}',
      '',
    ].join('\n')

    const events: Array<[string, Record<string, unknown>]> = []
    await readSseStream(await readerFromText(payload), (event, data) => {
      events.push([event, data])
    })

    expect(events).toEqual([['done', { answer: 'ok' }]])
  })

  it('handles multi-line data blocks', async () => {
    const payload = [
      'event: text',
      'data: {"text": "line1"',
      'data: , "extra": 1}',
      '',
    ].join('\n')

    const events: Array<[string, Record<string, unknown>]> = []
    await readSseStream(await readerFromText(payload), (event, data) => {
      events.push([event, data])
    })

    expect(events).toEqual([['text', { text: 'line1', extra: 1 }]])
  })
})
