import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Chat from './components/Chat'
import InputBox from './components/InputBox'
import WorkflowPanel from './components/WorkflowPanel'
import { postChat } from './api/client'

type Msg = { id: string; role: 'user' | 'agent'; text: string }

type Section = 'chat' | 'history' | 'knowledge' | 'agents' | 'settings'

type DebugData = {
  category?: string
  workflow?: string[]
  search_results?: Array<{ title: string; url: string; snippet?: string }>
  router?: Record<string, unknown>
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([
    { id: '1', role: 'agent', text: '欢迎使用 OmniForge。' }
  ])
  const [thinking, setThinking] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Ready')
  const [activeSection, setActiveSection] = useState<Section>('chat')
  const [debugData, setDebugData] = useState<DebugData | null>(null)
  const [showDebug, setShowDebug] = useState(false)

  const handleSend = async (text: string) => {
    const id = String(Date.now())
    setMessages(current => [...current, { id, role: 'user', text }])
    setThinking(true)
    setStatusMessage('Thinking...')

    try {
      const data = await postChat(text)
      const reply = data.reply || '[no reply]'
      setDebugData(data.debug || null)
      setMessages(current => [...current, { id: String(Date.now()), role: 'agent', text: reply }])
      setStatusMessage('Ready')
    } catch (error) {
      setMessages(current => [...current, { id: String(Date.now()), role: 'agent', text: '请求失败，请检查后端。' }])
      setStatusMessage('Error sending message')
    } finally {
      setThinking(false)
    }
  }

  const renderSection = () => {
    switch (activeSection) {
      case 'history':
        return (
          <div className="space-y-4">
            <div className="text-lg font-semibold">聊天历史</div>
            <div className="text-sm text-muted">点击聊天历史可以快速查看过往对话和会话摘要。</div>
            <div className="grid gap-3">
              {messages.slice(-3).map(msg => (
                <div key={msg.id} className="p-4 bg-[#10101a] rounded-xl">
                  <div className="text-xs text-muted">{msg.role === 'user' ? '用户' : 'OmniForge'}</div>
                  <div className="mt-2 text-sm">{msg.text}</div>
                </div>
              ))}
            </div>
          </div>
        )
      case 'knowledge':
        return (
          <div className="space-y-4">
            <div className="text-lg font-semibold">知识库</div>
            <div className="text-sm text-muted">当前未连接知识库。可以在设置中新增来源、文档或语料。</div>
            <div className="grid gap-3">
              <div className="p-4 bg-[#10101a] rounded-xl">示例知识源：API 文档、内部手册、技术方案</div>
              <div className="p-4 bg-[#10101a] rounded-xl">示例知识源：产品需求说明、FAQ、调试日志</div>
            </div>
          </div>
        )
      case 'agents':
        return (
          <div className="space-y-4">
            <div className="text-lg font-semibold">Agent 管理</div>
            <div className="text-sm text-muted">管理 OmniForge 中的自定义智能 Agent，并配置它们的工作流。</div>
            <div className="grid gap-3">
              <div className="p-4 bg-[#10101a] rounded-xl flex items-center justify-between">
                <div>
                  <div className="font-medium">Code Assistant</div>
                  <div className="text-xs text-muted">擅长修复代码、生成测试用例和优化方案</div>
                </div>
                <button type="button" className="px-3 py-2 rounded-lg bg-primary text-black">管理</button>
              </div>
              <div className="p-4 bg-[#10101a] rounded-xl flex items-center justify-between">
                <div>
                  <div className="font-medium">Research Agent</div>
                  <div className="text-xs text-muted">擅长查找资料、总结背景和生成执行策略</div>
                </div>
                <button type="button" className="px-3 py-2 rounded-lg bg-primary text-black">管理</button>
              </div>
            </div>
          </div>
        )
      case 'settings':
        return (
          <div className="space-y-4">
            <div className="text-lg font-semibold">设置</div>
            <div className="text-sm text-muted">配置 API 端点、激活模式和工作流参数。</div>
            <div className="grid gap-3">
              <button type="button" className="w-full text-left p-4 rounded-xl bg-[#10101a] hover:bg-hover">更改模型与令牌限制</button>
              <button type="button" className="w-full text-left p-4 rounded-xl bg-[#10101a] hover:bg-hover">知识库源配置</button>
            </div>
          </div>
        )
      default:
        return <Chat messages={messages} thinking={thinking} />
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <div className="flex flex-1">
        <aside className="w-64 p-4 border-r border-[#19191b]">
          <Sidebar activeSection={activeSection} onSelect={setActiveSection} />
        </aside>
        <main className="flex-1 p-6">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm text-muted">{statusMessage}</div>
            <div className="flex gap-2">
              {activeSection !== 'chat' && (
                <button
                  type="button"
                  onClick={() => setActiveSection('chat')}
                  className="px-3 py-2 rounded-lg bg-primary text-black"
                >
                  返回聊天
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowDebug(current => !current)}
                className="px-3 py-2 rounded-lg bg-[#2d2d3a] text-white"
              >
                {showDebug ? '隐藏开发者模式' : '显示开发者模式'}
              </button>
            </div>
          </div>
          <div className="rounded-3xl bg-card h-full p-4">{renderSection()}</div>
        </main>
        {showDebug && (
          <aside className="w-96 p-4 border-l border-[#19191b] hidden lg:block">
            <div className="glass rounded-lg p-3 h-full overflow-auto">
              <WorkflowPanel debug={debugData} />
            </div>
          </aside>
        )}
      </div>
      <footer className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-[90%] max-w-6xl">
        <InputBox onSend={handleSend} />
      </footer>
    </div>
  )
}
