import React, { useState } from 'react'
import { motion } from 'framer-motion'
import Thinking from './Thinking'

const defaultSteps = [
  { key: 'planner', label: 'Planning', done: false },
  { key: 'search', label: 'Searching', done: false },
  { key: 'knowledge', label: 'Reading Knowledge', done: false },
  { key: 'python', label: 'Running Python', done: false },
  { key: 'report', label: 'Generating Report', done: false }
]

export default function WorkflowPanel({ workflow = [] }: { workflow?: string[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const steps = defaultSteps.map(s => ({ ...s, done: workflow.includes(s.key) }))

  return (
    <div className="h-full flex flex-col gap-4">
      <div className="text-sm text-muted">Workflow</div>
      <Thinking steps={steps} />

      <div className="mt-2">
        {steps.map(s => (
          <motion.div key={s.key} layout className="mb-2">
            <div
              onClick={() => setExpanded(expanded === s.key ? null : s.key)}
              className="flex items-center justify-between p-2 rounded-lg hover:bg-hover cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${s.done ? 'bg-success' : 'bg-[#444]'}`} />
                <div className="text-sm">{s.label}</div>
              </div>
              <div className="text-xs text-muted">{s.done ? '✓' : '○'}</div>
            </div>

            {expanded === s.key && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-3 bg-card rounded mt-2">
                <div className="text-xs text-muted">Prompt</div>
                <pre className="text-sm whitespace-pre-wrap">(prompt content)</pre>
                <div className="text-xs text-muted mt-2">Output</div>
                <pre className="text-sm whitespace-pre-wrap">(output content)</pre>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  )
}
