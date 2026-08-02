import { useState, useRef, useEffect } from "react";

function formatMarkdown(text) {
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // ОБНОВЛЕНО: Меняем цвета маркдауна под новую светлую тему
  s = s.replace(
    /^### (.+)$/gm,
    '<div style="font-weight:700;color:#79b180;margin-top:8px;">$1</div>',
  );
  s = s.replace(
    /^## (.+)$/gm,
    '<div style="font-weight:700;color:#79b180;font-size:15px;margin-top:8px;">$1</div>',
  );
  s = s.replace(
    /^# (.+)$/gm,
    '<div style="font-weight:700;color:#79b180;font-size:16px;margin-top:8px;">$1</div>',
  );
  s = s.replace(/\*\*(.+?)\*\*/g, '<b style="color:#333;">$1</b>'); // Темный текст вместо белого
  s = s.replace(/\*(.+?)\*/g, "<i>$1</i>");
  s = s.replace(/^[-•] (.+)$/gm, '<div style="padding-left:12px;">• $1</div>');
  s = s.replace(/\n/g, "<br>");
  return s;
}

export default function ChatPanel({ onAnalyze, onLoadOpening }) {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: "Добро пожаловать в SFEDUCASTLING! Сделайте первый ход, и я начну анализ вашей партии.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const addMessage = (role, text) => {
    setMessages((prev) => [...prev, { role, text }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    addMessage("user", text);
    setLoading(true);
    try {
      const data = await onAnalyze();
      addMessage("ai", data.message || "Нет ответа от GigaChat.");
    } catch {
      addMessage("ai", "Ошибка соединения с сервером.");
    }
    setLoading(false);
  };

  return (
    <div className="chat-section">
      <div className="chat-header">
        <div className="chat-avatar">
          <img src="\bot-icon.svg" alt="Maia 2" />
        </div>
        <div className="chat-title-block">
          <h2>ЧАТ</h2>
          <span className="chat-subtitle">Шахматный ассистент</span>
        </div>
      </div>

      <div className="status-bar">
        <span className="dot"></span>
        <span>Maia Chess Engine подключён</span>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div
            className={`chat-msg ${msg.role === "user" ? "user" : ""}`}
            key={i}
          >
            <div className="msg-avatar">
              {msg.role === "user" ? "👤" : "⚡"}
            </div>

            <div
              className="msg-body"
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
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Задайте вопрос по позиции..."
        />
        <button onClick={handleSend} disabled={loading}>
          Отправить
        </button>
      </div>
    </div>
  );
}
