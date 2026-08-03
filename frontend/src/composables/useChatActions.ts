import { postChatStream, createSession, deleteSession, renameSession, getSessionMessages } from '@/api/client'
import type { Msg, ExecutionTask } from '@/types'
import {
  messages, sessions, currentSessionId, thinking, activeSection, abortController,
  streamingPhase, streamingCategory, tasks, switchingSession,
  _MSG_KEY, _SID_KEY, _recoverFromStorage, _loadSessionMessages, _refreshSessions, _upsertSession,
  sourceMode,
} from './chatState'

export function useChatActions() {
  const handleSend = async (text: string) => {
    const localId = String(Date.now())
    const userMsg: Msg = { id: localId, role: 'user', text }
    messages.value = [...messages.value, userMsg]
    thinking.value = true
    streamingPhase.value = '发送中...'
    tasks.value = []

    // Create abort controller for this request
    abortController.value = new AbortController()
    const signal = abortController.value!.signal

    const history = messages.value
      .filter((m) => m.id !== localId)
      .map((m) => ({
        role: m.role === 'user' ? 'user' as const : 'assistant' as const,
        content: m.text,
      }))

    // Pre-create a placeholder agent message that gets filled progressively
    const agentMsgId = String(Date.now() + 1)
    const agentMsg: Msg = { id: agentMsgId, role: 'agent', text: '...' }
    messages.value = [...messages.value, agentMsg]

    try {
      // Streaming-only — no blocking fallback
      const result = await postChatStream(
        text,
        history,
        currentSessionId.value ?? undefined,
        sourceMode.value,
        (event, data) => {
          if (event === 'start') {
            streamingPhase.value = (data.phase as string) || '正在处理...'
          } else if (event === 'thinking') {
            streamingPhase.value = `分析中...`
            streamingCategory.value = (data.category as string) || ''
          } else if (event === 'planning') {
            streamingPhase.value = '制定计划...'
          } else if (event === 'searching') {
            streamingPhase.value = (data.phase as string) || '搜索中...'
          } else if (event === 'executing') {
            streamingPhase.value = (data.phase as string) || '执行中...'
          } else if (event === 'generating') {
            streamingPhase.value = '生成回答...'
          } else if (event === 'text') {
            // Progressive text delivery — replace placeholder on first text, then append
            const newText = (data.text as string) || ''
            if (agentMsg.text === '...') {
              agentMsg.text = newText
            } else {
              agentMsg.text += newText
            }
            // Trigger reactivity via splice (O(1) near-end, vs O(n) array spread)
            const idx = messages.value.findIndex((m) => m.id === agentMsgId)
            if (idx !== -1) {
              messages.value.splice(idx, 1, { ...agentMsg })
            }
          } else if (event === 'task_update') {
            tasks.value = (data.tasks as ExecutionTask[]) || []
          } else if (event === 'done') {
            // Final sync: any remaining running/todo tasks → mark done
            // Prevents UI from showing stuck state when answer already delivered
            tasks.value = tasks.value.map((t) =>
              t.status === 'running' || t.status === 'todo'
                ? { ...t, status: 'done' as const }
                : t,
            )
          } else if (event === 'tools') {
            streamingPhase.value = '加载工具列表...'
          } else if (event === 'cancelled') {
            // User cancelled — remove empty placeholder and show state
            messages.value = messages.value.filter((m) => m.id !== agentMsgId)
            streamingPhase.value = '已中断'
          } else if (event === 'reconnecting') {
            streamingPhase.value = `正在重连... (${data.attempt}/${data.maxRetries})`
          }
        },
        signal,
      )

      // Finalize: apply answer from done event if text events didn't deliver it
      const idx = messages.value.findIndex((m) => m.id === agentMsgId)
      const noRealContent = !agentMsg.text || agentMsg.text === '...'
      if (idx !== -1 && noRealContent) {
        if (result.answer) {
          agentMsg.text = result.answer
        } else if (result.degraded) {
          agentMsg.text = '系统处于受限模式，部分功能暂时不可用。'
        } else {
          agentMsg.text = '抱歉，我没有生成有效的回答。请重试或换个方式提问。'
        }
        messages.value = [...messages.value.slice(0, idx), { ...agentMsg }, ...messages.value.slice(idx + 1)]
      }

      // Check degraded mode flag
      if (result.degraded) {
        streamingPhase.value = '受限模式'
        streamingCategory.value = '系统部分功能不可用'
      }

      // Defensive: auto-finalize any remaining running/todo tasks.
      // If the backend didn't send a final task_update (edge case), this
      // prevents the UI from showing a stuck "待执行" state after the
      // answer has already been delivered.
      const lingering = tasks.value.some((t) => t.status === 'running' || t.status === 'todo')
      if (lingering) {
        tasks.value = tasks.value.map((t) =>
          t.status === 'running' || t.status === 'todo'
            ? { ...t, status: 'done' as const }
            : t,
        )
      }

      // Track the session id from streaming response
      if (result.session_id) {
        currentSessionId.value = result.session_id
      }

      // Update the current session in-place (move to top) instead of full refresh.
      // This prevents duplicates and avoids an extra network round-trip.
      const cid = currentSessionId.value
      if (cid) {
        const current = sessions.value.find(s => s.id === cid)
        if (current) {
          _upsertSession({ ...current, updated_at: new Date().toISOString() })
        } else {
          await _refreshSessions()
        }
      } else {
        await _refreshSessions()
      }
      streamingPhase.value = ''
    } catch (err) {
      // User aborted → clean up
      if (err instanceof DOMException && err.name === 'AbortError') {
        // Remove the empty placeholder
        messages.value = messages.value.filter((m) => m.id !== agentMsgId)
        streamingPhase.value = '已中断'
        return
      }
      // Streaming failed — update the placeholder with error message
      console.warn('[ChatState] SSE stream failed:', err, 'agentMsgId:', agentMsgId)
      agentMsg.text = '请求失败，请检查后端是否正常运行。'
      const idx = messages.value.findIndex((m) => m.id === agentMsgId)
      if (idx !== -1) {
        messages.value = [...messages.value.slice(0, idx), { ...agentMsg }, ...messages.value.slice(idx + 1)]
      }
    } finally {
      thinking.value = false
      abortController.value = null
      if (streamingPhase.value === '已中断') {
        setTimeout(() => { streamingPhase.value = '' }, 1000)
      } else {
        streamingPhase.value = ''
      }
    }
  }
  const stopChat = () => {
    if (abortController.value) {
      abortController.value.abort()
    }
  }
  const newChat = async () => {
    messages.value = []
    currentSessionId.value = null
    sessionStorage.removeItem(_MSG_KEY)
    sessionStorage.removeItem(_SID_KEY)
    try {
      const sess = await createSession()
      currentSessionId.value = sess.id
      _upsertSession(sess)
    } catch (e) {
      console.warn('Failed to create new session:', e)
    }
  }
  const recoverSessionMessages = async () => {
    // Case 1: completely empty — recover from storage/backend
    if (messages.value.length === 0) {
      if (_recoverFromStorage()) return
      const sid = currentSessionId.value
      if (sid != null) {
        try { await _loadSessionMessages(sid) } catch { /* ignore */ }
      }
      return
    }

    // Case 2: detect stale placeholder — the last agent message never got
    // a real response (SSE stream finished/errored while user was on another
    // page, leaving the "..." placeholder unresolved).
    const msgs = messages.value
    const lastMsg = msgs[msgs.length - 1]
    if (
      lastMsg &&
      lastMsg.role === 'agent' &&
      lastMsg.text === '...' &&
      !thinking.value
    ) {
      // Stream already ended — placeholder was never filled. Reload from
      // backend to get the persisted response (or remove placeholder if
      // nothing was saved).
      const sid = currentSessionId.value
      if (sid != null) {
        try {
          const remote = await getSessionMessages(sid)
          if (remote.length > 0) {
            messages.value = remote.map((m) => ({
              id: String(m.id),
              role: m.role as 'user' | 'agent',
              text: m.content,
            }))
            return
          }
        } catch { /* ignore */ }
      }
      // Backend had nothing — remove the stale placeholder so the user
      // can see their question and retry
      messages.value = msgs.filter((m) => m.id !== lastMsg.id)
      return
    }

    // Case 3: messages look intact, but verify sessionStorage is up to date
    // (covers edge case where storage was cleared by a crash)
  }
  const switchSession = async (sessionId: number) => {
    // Guard: don't switch to the already-active session
    if (currentSessionId.value === sessionId && !switchingSession.value) return
    // Guard: prevent concurrent session switches
    if (switchingSession.value) return
    // Abort any in-flight stream before switching
    if (abortController.value) {
      abortController.value.abort()
      abortController.value = null
    }
    thinking.value = false
    streamingPhase.value = ''
    tasks.value = []
    switchingSession.value = true
    try {
      activeSection.value = 'chat'
      await _loadSessionMessages(sessionId)
    } finally {
      switchingSession.value = false
    }
  }
  const deleteSessionById = async (sessionId: number) => {
    try {
      await deleteSession(sessionId)
      sessions.value = sessions.value.filter((s) => s.id !== sessionId)
      if (currentSessionId.value === sessionId) {
        if (sessions.value.length > 0) {
          await _loadSessionMessages(sessions.value[0].id)
        } else {
          messages.value = []
          currentSessionId.value = null
          const sess = await createSession()
          currentSessionId.value = sess.id
          _upsertSession(sess)
        }
      }
    } catch (e) {
      console.warn('Failed to delete session:', e)
    }
  }
  const renameSessionById = async (sessionId: number, title: string) => {
    try {
      await renameSession(sessionId, title)
      const idx = sessions.value.findIndex((s) => s.id === sessionId)
      if (idx !== -1) {
        sessions.value[idx] = { ...sessions.value[idx], title }
      }
    } catch (e) {
      console.warn('Failed to rename session:', e)
    }
  }

  return { handleSend, stopChat, newChat, recoverSessionMessages, switchSession, deleteSessionById, renameSessionById }
}
