import React from 'react'

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
  return (
    <div className="h-[70vh] overflow-auto">
      <Message side="left">Agent: 这是一个示例回答，支持 Markdown，代码块，Mermaid 等渲染。</Message>
      <Message side="right">用户：请给我一个简短的答复。</Message>
      <Message side="left">Agent: (Thinking...) Planning → Searching → Generating</Message>
    </div>
  )
}
