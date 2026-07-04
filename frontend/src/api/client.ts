import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export async function postChat(message: string) {
  const resp = await axios.post(`${API_BASE}/chat`, { message })
  return resp.data
}
