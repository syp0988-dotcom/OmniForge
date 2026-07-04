import React from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Chat from './components/Chat'
import InputBox from './components/InputBox'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1">
        <aside className="w-64 p-4 border-r border-[#19191b]">
          <Sidebar />
        </aside>
        <main className="flex-1 p-6">
          <Chat />
        </main>
        <aside className="w-96 p-4 border-l border-[#19191b] hidden lg:block">
          <div className="glass rounded-lg p-3 h-full">Workflow / Artifacts</div>
        </aside>
      </div>
      <footer className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-[90%] max-w-6xl">
        <InputBox />
      </footer>
    </div>
  )
}
