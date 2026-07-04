import React from 'react'

export default function Header() {
  return (
    <header className="w-full py-3 px-6 border-b border-[#19191b] glass">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-sm text-muted">Model</div>
          <div className="px-3 py-1 bg-[#1f1b3a] rounded text-primary text-sm">DeepSeek • online</div>
        </div>
        <div className="text-sm text-muted">OmniForge</div>
      </div>
    </header>
  )
}
