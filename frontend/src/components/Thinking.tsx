import React from 'react'
import { motion } from 'framer-motion'

type Step = { key: string; label: string; done?: boolean }

export default function Thinking({ steps }: { steps: Step[] }) {
  return (
    <div className="p-3 bg-card rounded-lg glass">
      <div className="flex flex-col gap-2">
        {steps.map((s, i) => (
          <motion.div
            key={s.key}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.07 }}
            className="flex items-center gap-3"
          >
            <div className={`w-5 h-5 rounded-full flex items-center justify-center ${s.done ? 'bg-success' : 'bg-[#444]'}`} />
            <div className="text-sm">{s.label}</div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
