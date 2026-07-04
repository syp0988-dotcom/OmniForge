import React from 'react'

export default function InputBox() {
  return (
    <div className="bg-card glass rounded-lg p-3 shadow-lg">
      <div className="flex items-start gap-3">
        <textarea
          rows={1}
          placeholder="按 / 或 @ 呼出快捷命令，Shift+Enter 换行，Enter 发送"
          className="w-full resize-none bg-transparent outline-none text-text placeholder: text-muted"
        />
        <button className="ml-2 bg-primary text-black px-4 py-2 rounded-lg">发送</button>
      </div>
    </div>
  )
}
