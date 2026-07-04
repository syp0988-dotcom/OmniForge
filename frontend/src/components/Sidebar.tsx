import React from 'react'

export default function Sidebar() {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center text-black font-bold">OF</div>
        <div>
          <div className="text-white font-semibold">OmniForge</div>
          <div className="text-muted text-sm">Developer AI Workspace</div>
        </div>
      </div>

      <nav className="mt-6">
        <ul className="space-y-2 text-sm">
          <li className="p-2 rounded-lg hover:bg-hover">聊天历史</li>
          <li className="p-2 rounded-lg hover:bg-hover">知识库</li>
          <li className="p-2 rounded-lg hover:bg-hover">Agent 管理</li>
          <li className="p-2 rounded-lg hover:bg-hover">设置</li>
        </ul>
      </nav>
    </div>
  )
}
