/**
 * Incremental SSE (Server-Sent Events) parsing.
 *
 * Extracted from the chat client so the buffering/event-dispatch logic can be
 * unit-tested without a browser or network.
 */

export type SseEventHandler = (event: string, data: Record<string, unknown>) => void

/**
 * Read an SSE stream from a fetch reader and invoke {@link onEvent} for every
 * complete event block. Malformed JSON payloads are ignored so partial or
 * interleaved chunks never break the stream.
 */
export async function readSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: SseEventHandler,
): Promise<void> {
  const decoder = new TextDecoder()
  let buffer = ''
  let currentEvent = ''
  let currentData = ''

  const dispatch = () => {
    if (!currentEvent) return
    try {
      const parsed = JSON.parse(currentData) as Record<string, unknown>
      onEvent(currentEvent, parsed)
    } catch {
      // Ignore malformed/incomplete events; the next chunk can still recover.
    } finally {
      currentEvent = ''
      currentData = ''
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // keep incomplete line

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        currentData += currentData ? `\n${line.slice(6).trim()}` : line.slice(6).trim()
      } else if (line === '' && currentEvent) {
        dispatch()
      }
    }
  }

  dispatch()
}
