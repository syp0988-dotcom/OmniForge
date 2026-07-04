import React, { useCallback, useEffect, useRef, useState } from 'react'

export default function InputBox() {
  const [value, setValue] = useState('')
  const taRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    taRef.current?.focus()
  }, [])

  const send = useCallback(() => {
    const txt = value.trim()
    if (!txt) return
    ;(window as any).omniforgeSend && (window as any).omniforgeSend(txt)
    setValue('')
  }, [value])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="bg-card glass rounded-lg p-3 shadow-lg">
      <div className="flex items-start gap-3">
        <textarea
          ref={taRef}
          rows={1}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="按 / 或 @ 呼出快捷命令，Shift+Enter 换行，Enter 发送"
          className="w-full resize-none bg-transparent outline-none text-text placeholder:text-muted"
        />
        <button onClick={send} className="ml-2 of-btn-primary px-4 py-2 rounded-lg">发送</button>
      </div>
    </div>
  )
}
