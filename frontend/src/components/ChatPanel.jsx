import { useState, useRef, useEffect, useLayoutEffect } from "react";
import { askLLM, fetchOpeningByFen } from "../api.js";
import MiniBoard from "./MiniBoard.jsx";
import ChatCollapsedStrip from "./ChatCollapsedStrip.jsx";

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

export default function ChatPanel({
  onAnalyze,
  onLoadOpening,
  hintMessage,
  currentFen,
  currentMoves,
  isComparing,
  collapsed = false,
  onToggleCollapsed,
}) {
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
  const rootRef = useRef(null);
  // Ширина панели в развёрнутом виде: с неё начинается анимация сворачивания.
  const naturalWidthRef = useRef(0);
  // Ширина, на которой анимацию прервали повторным кликом: следующий
  // проезд начинаем ровно оттуда, без скачка.
  const interruptedRef = useRef(0);
  const firstRunRef = useRef(true);

  /*
   * Плавное сворачивание/разворачивание: ширина колонки едет к новому
   * значению, а строка пересчитывается на каждом кадре - доска и шкала
   * оценки съезжают вместе с ней. Ширина в развёрнутом виде задана
   * флексом и зависит от окна, поэтому запоминаем её замером, а не
   * константой.
   */
  useEffect(() => {
    const el = rootRef.current;

    if (!el || collapsed) {
      return undefined;
    }

    const remember = () => {
      if (el.classList.contains("chat-section--sliding")) return;
      const width = el.getBoundingClientRect().width;
      if (width > 0) naturalWidthRef.current = width;
    };

    remember();

    const observer = new ResizeObserver(remember);
    observer.observe(el);

    return () => observer.disconnect();
  }, [collapsed]);

  useLayoutEffect(() => {
    const el = rootRef.current;

    if (!el) {
      return undefined;
    }

    // Первый рендер не анимируем: панель и так уже нужной ширины.
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return undefined;
    }

    // Снимаем остатки прошлой анимации, иначе замерим не ту ширину.
    el.classList.remove("chat-section--sliding", "chat-section--reveal");
    el.style.flex = "";
    el.style.width = "";
    el.style.minWidth = "";
    el.style.maxWidth = "";

    const target = el.getBoundingClientRect().width;
    const from =
      interruptedRef.current ||
      (collapsed ? naturalWidthRef.current : 48);

    interruptedRef.current = 0;

    if (!from || Math.abs(target - from) < 1) {
      return undefined;
    }

    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = reduceMotion ? 0 : 340;

    /*
     * На время анимации ширину задаём инлайном: у свёрнутого класса
     * жёсткие min/max-width 48px, у развёрнутого - min-width 320px,
     * они бы не дали панели проехать промежуточные значения.
     */
    el.classList.add("chat-section--sliding");
    el.style.flex = "0 0 auto";
    el.style.minWidth = "0px";
    el.style.maxWidth = "none";
    el.style.width = `${target}px`;

    const animation = el.animate(
      [{ width: `${from}px` }, { width: `${target}px` }],
      { duration, easing: "cubic-bezier(0.4, 0, 0.2, 1)" }
    );

    const finish = () => {
      el.style.flex = "";
      el.style.width = "";
      el.style.minWidth = "";
      el.style.maxWidth = "";
      el.classList.remove("chat-section--sliding");
      el.classList.add("chat-section--reveal");

      if (!collapsed) {
        naturalWidthRef.current = el.getBoundingClientRect().width;
      }

      window.setTimeout(() => {
        el.classList.remove("chat-section--reveal");
      }, 260);
    };

    animation.addEventListener("finish", finish, { once: true });

    return () => {
      if (animation.playState === "running") {
        interruptedRef.current = el.getBoundingClientRect().width;
      }

      animation.cancel();
    };
  }, [collapsed]);


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

  /*
   * Дебют из базы знаний: определяем дебют по текущей позиции и
   * показываем карточку прямо в чате.
   */
  const handleShowOpening = async () => {
    if (loading) return;
    const fen = (currentFen || "").trim();
    if (!fen) {
      addMessage("ai", "Позиция на доске ещё не определена.");
      return;
    }

    setLoading(true);
    try {
      const data = await fetchOpeningByFen(fen);
      if (data.error) {
        addMessage("ai", data.error);
      } else if (!data.opening) {
        addMessage("ai", data.message || "Дебют не найден в базе знаний.");
      } else {
        const o = data.opening;
        /*
         * Название показываем по-русски. Если бэкенд не смог перевести
         * какое-то редкое имя, name_ru не будет — тогда остаётся
         * оригинальное английское.
         */
        const title = o.name_ru || o.name;
        const lines = [`**${title}**${o.eco ? ` (${o.eco})` : ""}`];
        if (o.name_ru && o.name && o.name_ru !== o.name) {
          lines.push(`*${o.name}*`);
        }
        if (o.pgn) {
          lines.push(`Продолжение: ${o.pgn}`);
        }
        addMessage("ai", lines.join("\n"));
      }
    } catch {
      addMessage("ai", "Не удалось загрузить данные о дебюте.");
    }
    setLoading(false);
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

  /*
   * Свёрнутый чат: узкая стеклянная полоса во всю высоту колонки.
   * История сообщений не теряется: компонент не размонтируется,
   * меняется только разметка.
   */
  if (collapsed) {
    return (
      <div className="chat-section chat-section--collapsed" ref={rootRef}>
        <ChatCollapsedStrip onExpand={onToggleCollapsed} />
      </div>
    );
  }

  return (
    <div className="chat-section" ref={rootRef}>
      <div className="chat-header">
        <div className="chat-avatar">
          <img src="\bot-icon.svg" alt="Maia 3" />
        </div>
        <div className="chat-title-block">
          <h2 className="chat-panel-title">Шахматный ассистент</h2>
        </div>
        <button
          type="button"
          className="chat-collapse-btn"
          onClick={onToggleCollapsed}
          title="Свернуть чат"
          aria-label="Свернуть чат"
          aria-expanded={true}
        >
          »
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div
            className={`chat-msg ${msg.role === "user" ? "user" : ""}`}
            key={i}
          >
            <div className={`msg-avatar ${msg.role === "ai" ? "msg-avatar--ai" : ""}`}>
              {msg.role === "user" ? "♟" : "♞"}
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
            <div className="msg-avatar msg-avatar--ai">♞</div>
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

      <div className="chat-bottom-actions">
        <button
          type="button"
          className="chat-opening-btn"
          onClick={handleShowOpening}
          disabled={loading}
          title="Определить дебют по текущей позиции"
        >
          ♟ Определить дебют
        </button>
        <button
          type="button"
          className="chat-opening-btn"
          onClick={onAnalyze}
          disabled={isComparing}
          title="Показать лучший ход в текущей позиции"
        >
          {isComparing ? "Анализ..." : "Подсказка"}
        </button>
      </div>
    </div>
  );
}
