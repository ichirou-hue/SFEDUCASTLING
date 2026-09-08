import { useState, useRef, useEffect } from "react";
import { askLLM } from "../api.js";
import MiniBoard from "./MiniBoard.jsx";

const GREETING_ONLY = /^(привет|здравствуй|хай|hello|hi|hey|пока|до свидания|спасибо|благодарю)$/i;
const CHESS_MARKERS = /позиц|ход|фигур|ферз|конь|ладь|пешк|слон|рокир|мат|шах|взят|игр|доск|парти|elo|rating|уровн|помоги|объясни|что делать|чем ход|лучш|ход\b|uci|fen|дебют/i;

function classifyLocal(text) {
  if (GREETING_ONLY.test(text.trim())) return "greeting";
  if (CHESS_MARKERS.test(text)) return "chess";
  return "unknown";
}

const STOCKFISH_GREEN = "rgba(0, 150, 50, 0.85)";
const MAIA_ORANGE = "rgba(220, 120, 20, 0.9)";

/*
 * Стрелки для мини-доски подсказки Stockfish:
 * зелёная — объективно лучший ход, оранжевая — ход Maia3,
 * если движки разошлись.
 */
function buildHintArrows(hintMessage) {
  const arrows = [];

  if (hintMessage?.stockfishArrow?.from && hintMessage?.stockfishArrow?.to) {
    arrows.push({ ...hintMessage.stockfishArrow, color: STOCKFISH_GREEN });
  }

  if (
    hintMessage?.maiaArrow?.from &&
    hintMessage?.maiaArrow?.to &&
    (!hintMessage.stockfishArrow ||
      hintMessage.maiaArrow.from !== hintMessage.stockfishArrow.from ||
      hintMessage.maiaArrow.to !== hintMessage.stockfishArrow.to)
  ) {
    arrows.push({ ...hintMessage.maiaArrow, color: MAIA_ORANGE, role: "maia" });
  }

  return arrows;
}

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

export default function ChatPanel({ onAnalyze, onLoadOpening, hintMessage, currentFen, currentMoves }) {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: "Добро пожаловать в SFEDUCASTLING! Сделайте первый ход, и я начну анализ вашей партии.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const lastHintMessageIdRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /*
   * ============================================================
   * ПОДСКАЗКА STOCKFISH
   * ============================================================
   *
   * hintMessage приходит из ChessBoard через родительский компонент.
   *
   * Он появляется только после нажатия кнопки "Лучший ход".
   */

  useEffect(() => {
  if (!hintMessage?.text) {
    return;
  }

  /*
   * Защита от повторного добавления одного и того же
   * сообщения при повторном рендере компонента.
   */
  if (
    lastHintMessageIdRef.current ===
    hintMessage.id
  ) {
    return;
  }

  lastHintMessageIdRef.current =
    hintMessage.id;

  setMessages((prev) => [
    ...prev,
    {
      role: "ai",
      text: hintMessage.text,
      fen: hintMessage.fen || null,
      arrow: hintMessage.stockfishArrow || null,
      arrows: buildHintArrows(hintMessage),
    },
  ]);
  }, [hintMessage]);

  const addMessage = (role, text, fen = null, arrows = null) => {
    setMessages((prev) => [...prev, { role, text, fen, arrows }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    addMessage("user", text);

    const sanitized = text.replace(/<[^>]*>/g, "");
    const type = classifyLocal(sanitized);

    setLoading(true);
    try {
      const data = await askLLM(
        sanitized,
        currentFen || "",
        currentMoves || [],
        type === "greeting",
      );
      const reply = data.reply || "Не понял вопрос. Спросите о шахматах!";
      addMessage(
        "ai",
        reply,
        data.fen || null,
        data.arrow
          ? [{ ...data.arrow, color: STOCKFISH_GREEN }]
          : null,
      );
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

            <div className="msg-content">
              <div
                className="msg-body"
                dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.text) }}
              />

              {msg.fen && (
                <MiniBoard
                  fen={msg.fen}
                  width={230}
                  lastMove={msg.lastMove}
                  arrows={msg.arrows || (msg.arrow ? [msg.arrow] : null)}
                />
              )}
            </div>
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
