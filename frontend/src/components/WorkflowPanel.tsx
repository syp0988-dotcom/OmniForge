import React from 'react'

type DebugItem = {
  category?: string
  workflow?: string[]
  search_results?: Array<{ title?: string; url?: string; snippet?: string }>
  router?: Record<string, unknown>
}

type WorkflowPanelProps = {
  debug?: DebugItem | null
}

export default function WorkflowPanel({ debug }: WorkflowPanelProps) {
  if (!debug) {
    return (
      <div className="h-full flex flex-col gap-4">
        <div className="text-sm text-muted">Developer Mode</div>
        <div className="p-4 bg-card rounded-lg">暂无调试数据，发送一条消息后可查看。</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col gap-4">
      <div>
        <div className="text-sm text-muted">Developer Mode</div>
        <div className="text-lg font-semibold">调试输出</div>
      </div>

      <div className="space-y-4 overflow-auto">
        <div className="p-3 bg-card rounded-lg">
          <div className="text-xs text-muted">Category</div>
          <div className="mt-2 text-sm">{debug.category || 'unknown'}</div>
        </div>

        <div className="p-3 bg-card rounded-lg">
          <div className="text-xs text-muted">Workflow</div>
          <div className="mt-2 text-sm">{Array.isArray(debug.workflow) ? debug.workflow.join(' → ') : 'N/A'}</div>
        </div>

        <div className="p-3 bg-card rounded-lg">
          <div className="text-xs text-muted">Router</div>
          <pre className="mt-2 text-sm whitespace-pre-wrap">{JSON.stringify(debug.router || {}, null, 2)}</pre>
        </div>

        <div className="p-3 bg-card rounded-lg">
          <div className="text-xs text-muted">Search Results</div>
          {debug.search_results && debug.search_results.length > 0 ? (
            <ul className="mt-2 space-y-2 text-sm">
              {debug.search_results.map((item, index) => (
                <li key={index} className="rounded-lg border border-[#2b2b3b] p-2">
                  <div className="font-medium">{item.title || 'Untitled'}</div>
                  <div className="text-xs text-muted">{item.url}</div>
                  <div className="mt-1">{item.snippet}</div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="mt-2 text-sm text-muted">No search results captured.</div>
          )}
        </div>
      </div>
    </div>
  )
}
