import { useState, useRef, useEffect } from "react";
import { sendChatMessage, fetchChatMessages } from "../api.js";

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
  let s = escapeHtml(text);

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
  s = s.replace(/\*\*(.+?)\*\*/g, '<b style="color:#333;">$1</b>');
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
  const [lastIndex, setLastIndex] = useState(0);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Poll backend for new messages every 3s
  useEffect(() => {
    const poll = async () => {
      try {
        const data = await fetchChatMessages(lastIndex);
        if (data.messages && data.messages.length > 0) {
          const newMsgs = data.messages.map((m) => ({
            role: m.role === "user" ? "user" : "ai",
            text: m.text,
          }));
          setMessages((prev) => [...prev, ...newMsgs]);
          setLastIndex((prev) => prev + data.messages.length);
        }
      } catch {
        // silent — server may be off
      }
    };
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [lastIndex]);

  const addMessage = (role, text) => {
    setMessages((prev) => [...prev, { role, text }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    addMessage("user", text);

    // Sanitize: strip HTML tags before sending to backend
    const sanitized = text.replace(/<[^>]*>/g, "");

    // Store user message on backend
    try {
      await sendChatMessage(sanitized, "user");
    } catch {
      // continue even if backend is off
    }

    setLoading(true);
    try {
      const data = await onAnalyze();
      const reply = data.message || "Нет ответа от GigaChat.";
      addMessage("ai", reply);
      // Store assistant reply on backend so other clients can see it
      try {
        await sendChatMessage(reply.replace(/<[^>]*>/g, ""), "assistant");
      } catch {
        // ok
      }
    } catch {
      addMessage("ai", "Ошибка соединения с сервером.");
    }
    setLoading(false);
  };

  return (
    <div className="chat-section">
      <div className="chat-header">
        <div className="chat-avatar">
          <img src="\bot-icon.svg" alt="Maia 3" />
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
