import { useState, useRef, useEffect } from 'react'

function formatMarkdown(text) {
  let s = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  s = s.replace(/^### (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;margin-top:8px;">$1</div>')
  s = s.replace(/^## (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;font-size:15px;margin-top:8px;">$1</div>')
  s = s.replace(/^# (.+)$/gm, '<div style="font-weight:700;color:#83b4aa;font-size:16px;margin-top:8px;">$1</div>')
  s = s.replace(/\*\*(.+?)\*\*/g, '<b style="color:#e2e8f0;">$1</b>')
  s = s.replace(/\*(.+?)\*/g, '<i>$1</i>')
  s = s.replace(/^[-•] (.+)$/gm, '<div style="padding-left:12px;">• $1</div>')
  s = s.replace(/\n/g, '<br>')
  return s
}

export default function ChatPanel({ onAnalyze, onLoadOpening }) {
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Добро пожаловать в SFEDUCASTLING! Сделайте первый ход, и я начну анализ вашей партии.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const addMessage = (role, text) => {
    setMessages(prev => [...prev, { role, text }])
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    addMessage('user', text)
    setLoading(true)
    try {
      const data = await onAnalyze()
      addMessage('ai', data.message || 'Нет ответа от GigaChat.')
    } catch {
      addMessage('ai', 'Ошибка соединения с сервером.')
    }
    setLoading(false)
  }

  return (
    <div className="chat-section">
      <div className="chat-header">
        <div className="icon">⚡</div>
        <h2>Анализ GigaChat</h2>
        <span className="badge">AI</span>
      </div>
      <div className="status-bar">
        <span className="dot"></span>
        <span>Maia Chess Engine подключён</span>
      </div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div className="chat-msg" key={i}>
            <div className="msg-avatar" style={msg.role === 'user' ? { background: '#0d3550' } : undefined}>
              {msg.role === 'user' ? '👤' : '⚡'}
            </div>
            <div
              className="msg-body"
              style={msg.role === 'user' ? { borderRadius: '12px 0 12px 12px' } : undefined}
              dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.text) }}
            />
          </div>
        ))}
        {loading && (
          <div className="chat-msg">
            <div className="msg-avatar">⚡</div>
            <div className="msg-body">Думаю...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Задайте вопрос по позиции..."
        />
        <button onClick={handleSend} disabled={loading}>Отправить</button>
      </div>
    </div>
  )
}
