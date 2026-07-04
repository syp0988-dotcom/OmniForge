import React, { useEffect, useRef, useState } from 'react'
import Thinking from './Thinking'
import WorkflowPanel from './WorkflowPanel'
import { postChat } from '../api/client'

type Msg = { id: string; role: 'user' | 'agent'; text: string }

function Message({ side = 'left', children }: { side?: 'left' | 'right'; children: React.ReactNode }) {
  return (
    <div className={`flex ${side === 'right' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[70%] p-4 rounded-lg ${side === 'right' ? 'bg-primary text-black' : 'bg-card text-text'}`}>
        {children}
      </div>
    </div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([
    { id: '1', role: 'agent', text: '欢迎使用 OmniForge。' }
  ])
  const [thinking, setThinking] = useState(false)
  const [workflow, setWorkflow] = useState<string[]>([])
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, thinking])

  // expose a simple send function for InputBox via window for now
  ;(window as any).omniforgeSend = async (text: string) => {
    const id = String(Date.now())
    setMessages(m => [...m, { id, role: 'user', text }])
    setThinking(true)
    try {
      const data = await postChat(text)
      const reply = data.reply || data.answer || '[no reply]'
      const wf = data.workflow || []
      setWorkflow(wf)
      setMessages(m => [...m, { id: String(Date.now()), role: 'agent', text: reply }])
    } catch (e) {
      setMessages(m => [...m, { id: String(Date.now()), role: 'agent', text: '请求失败，请检查后端。' }])
    } finally {
      setThinking(false)
    }
  }

  return (
    <div className="h-[70vh] overflow-auto" ref={containerRef}>
      {messages.map(m => (
        <Message key={m.id} side={m.role === 'user' ? 'right' : 'left'}>{m.text}</Message>
      ))}

      {thinking && <Thinking steps={[{ key: 'planner', label: 'Planning' }, { key: 'search', label: 'Searching' }, { key: 'report', label: 'Generating Report' }]} />}
    </div>
  )
}
