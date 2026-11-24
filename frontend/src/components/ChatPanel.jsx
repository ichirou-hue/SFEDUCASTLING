import { useState, useRef, useEffect, useLayoutEffect, useMemo } from "react";
import { Chess } from "chess.js";
import {
  askLLM,
  checkOpeningMove,
  fetchOpeningByFen,
  fetchOpenings,
} from "../api.js";
import { isStartingPosition } from "../chessStatus.js";
import formatMarkdown from "../formatMarkdown.js";
import useChatAutoScroll from "../useChatAutoScroll.js";
import MiniBoard from "./MiniBoard.jsx";
import ChatCollapsedStrip from "./ChatCollapsedStrip.jsx";
import ChatCollapseIcon from "./ChatCollapseIcon.jsx";

const GREETING_ONLY = /^(привет|здравствуй|хай|hello|hi|hey|пока|до свидания|спасибо|благодарю)$/i;
const CHESS_MARKERS = /позиц|ход|фигур|ферз|конь|ладь|пешк|слон|рокир|мат|шах|взят|игр|доск|парти|elo|rating|уровн|помоги|объясни|что делать|чем ход|лучш|ход\b|uci|fen|дебют/i;

function classifyLocal(text) {
  if (GREETING_ONLY.test(text.trim())) return "greeting";
  if (CHESS_MARKERS.test(text)) return "chess";
  return "unknown";
}

const STOCKFISH_GREEN = "rgba(0, 150, 50, 0.85)";
const MAIA_ORANGE = "rgba(220, 120, 20, 0.9)";

// Ширина свёрнутой полосы чата: должна совпадать с третьей колонкой
// .main-area в style.css.
const COLLAPSED_CHAT_WIDTH = 48;

/*
 * Стрелки для мини-доски подсказки Stockfish:
 * зелёная - объективно лучший ход, оранжевая - ход Maia3,
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

/*
 * Каталог дебютов для панели чата. База отдаёт ~3800 записей (все линии),
 * поэтому в списке по умолчанию показываем только семейства - самую
 * короткую линию каждого базового названия (149 записей). Поиск идёт по
 * всем линиям, но результаты ограничены, чтобы не рендерить тысячи кнопок.
 */
function buildOpeningCatalog(openings) {
  const families = new Map();
  const lines = new Map();

  for (const opening of openings) {
    const name = String(opening?.name || "");
    if (!name) continue;
    const moveCount = Array.isArray(opening.moves) ? opening.moves.length : 0;

    const base = name.split(":")[0].trim();
    const currentBase = families.get(base);
    if (!currentBase || moveCount < currentBase.moveCount) {
      families.set(base, { opening, moveCount });
    }

    const key = name.toLowerCase();
    const currentLine = lines.get(key);
    if (!currentLine || moveCount < currentLine.moveCount) {
      lines.set(key, { opening, moveCount });
    }
  }

  const title = (opening) => opening.name_ru || opening.name || "";
  const familyList = [...families.values()]
    .map((item) => item.opening)
    .sort((a, b) => title(a).localeCompare(title(b), "ru"));

  return {
    families: familyList,
    lines: [...lines.values()].map((item) => item.opening),
  };
}

/* UCI-ход из базы знаний (e2e4) → SAN для показа человеку (e4). */
function uciToSan(fen, uci) {
  try {
    const game = new Chess(fen);
    const move = game.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.slice(4, 5) || "q",
    });
    return move?.san || uci;
  } catch {
    return uci;
  }
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
  const messagesListRef = useChatAutoScroll(messages);
  const lastHintMessageIdRef = useRef(null);
  const rootRef = useRef(null);
  // Ширина панели в развёрнутом виде: с неё начинается анимация сворачивания.
  const naturalWidthRef = useRef(0);
  // Ширина, на которой анимацию прервали повторным кликом: следующий
  // проезд начинаем ровно оттуда, без скачка.
  const interruptedRef = useRef(0);
  // Предыдущее состояние чата: первый рендер (и повторный вызов эффекта
  // в StrictMode) не анимируем - панель уже нужной ширины.
  const prevCollapsedRef = useRef(collapsed);

  // Каталог дебютов: грузим лениво при первом открытии (ответ ~3 МБ),
  // поэтому держим его в состоянии и переиспользуем.
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [catalog, setCatalog] = useState(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState(null);
  const [catalogQuery, setCatalogQuery] = useState("");

  /*
   * Плавное сворачивание/разворачивание: ширина колонки едет к новому
   * значению, а строка пересчитывается на каждом кадре - доска и шкала
   * оценки съезжают вместе с ней. Ширина в развёрнутом виде задана
   * грид-треком и зависит от окна, поэтому запоминаем её замером, а не
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
    const layout = el?.parentElement;

    if (!el || !layout) {
      return undefined;
    }

    /*
     * Возвращаем панель и сетку в исходное состояние. Ширину колонки
     * анимируем не у самой панели, а у грид-контейнера: только так
     * соседняя колонка с доской пересчитывается на каждом кадре.
     */
    const reset = () => {
      el.classList.remove("chat-section--sliding", "chat-section--reveal");
      el.style.flex = "";
      el.style.width = "";
      el.style.minWidth = "";
      el.style.maxWidth = "";
      delete el.dataset.chatTargetWidth;
      layout.style.gridTemplateColumns = "";
    };

    // Первый рендер не анимируем: панель и так уже нужной ширины.
    // Сравнение с прошлым состоянием заодно гасит повторный запуск
    // layout-эффекта в StrictMode.
    if (prevCollapsedRef.current === collapsed) {
      return undefined;
    }
    prevCollapsedRef.current = collapsed;

    if (window.matchMedia("(max-width: 1450px)").matches) {
      reset();
      return undefined;
    }

    // Снимаем остатки прошлой анимации, иначе замерим не ту ширину.
    reset();

    /*
     * Целевую ширину нельзя брать из замера в свёрнутом состоянии:
     * в первом кадре колонка ещё может быть старой ширины. Поэтому
     * для сворачивания она известна (48px), а для разворачивания
     * берём реально отрисованную ширину развёрнутой панели.
     */
    const measured = el.getBoundingClientRect().width;
    const target = collapsed ? COLLAPSED_CHAT_WIDTH : measured;
    const natural = collapsed ? naturalWidthRef.current : measured;
    const from =
      interruptedRef.current || (collapsed ? natural : COLLAPSED_CHAT_WIDTH);

    interruptedRef.current = 0;

    if (!from || !target || Math.abs(target - from) < 1) {
      if (!collapsed && measured > 0) {
        naturalWidthRef.current = measured;
      }
      return undefined;
    }

    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = reduceMotion ? 0 : 340;
    const easing = "cubic-bezier(0.4, 0, 0.2, 1)";

    /*
     * Первую колонку (история ходов) оставляем как есть, средняя
     * остаётся гибкой и забирает освободившуюся ширину, а третья
     * едет между значениями. Это тот же приём, что и на странице
     * задач (см. PuzzlesPage.jsx).
     */
    const firstCol = getComputedStyle(layout).gridTemplateColumns.split(" ")[0];
    const track = (width) => `${firstCol} minmax(0, 1fr) ${width}px`;

    el.classList.add("chat-section--sliding");
    // По этому значению Chessboard считает итоговый размер доски,
    // пока колонки ещё едут.
    el.dataset.chatTargetWidth = String(Math.round(target));

    const animation = layout.animate(
      [
        { gridTemplateColumns: track(from) },
        { gridTemplateColumns: track(target) },
      ],
      { duration, easing }
    );

    const finish = () => {
      reset();
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
        /*
         * Ширину прерванной анимации берём у грид-колонки, а не у
         * панели: при переключении разметки старый DOM-узел уже
         * может быть откреплён.
         */
        const columns = getComputedStyle(layout).gridTemplateColumns.split(" ");
        const current = parseFloat(columns[columns.length - 1]);
        if (current > 0) {
          interruptedRef.current = current;
        }
      }

      animation.cancel();
    };
  }, [collapsed]);


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
   * Поиск по каталогу: пустой запрос показывает семейства, иначе ищем по
   * русскому/английскому названию, ECO и линии. Ограничение в 60 строк
   * достаточно для выбора мышью и не заставляет браузер рисовать всё.
   */
  const catalogResults = useMemo(() => {
    if (!catalog) return [];
    const query = catalogQuery.trim().toLocaleLowerCase("ru");
    if (!query) return catalog.families;

    const matched = catalog.lines.filter((opening) =>
      [opening.name, opening.name_ru, opening.eco, opening.pgn]
        .filter(Boolean)
        .join(" ")
        .toLocaleLowerCase("ru")
        .includes(query),
    );
    return matched.slice(0, 60);
  }, [catalog, catalogQuery]);

  const openCatalog = async () => {
    setCatalogOpen(true);
    if (catalog || catalogLoading) return;

    setCatalogLoading(true);
    setCatalogError(null);
    try {
      const data = await fetchOpenings();
      const openings = Array.isArray(data?.openings) ? data.openings : [];
      if (openings.length === 0) {
        setCatalogError(data?.error || "Каталог дебютов пуст.");
        return;
      }
      setCatalog(buildOpeningCatalog(openings));
    } catch {
      setCatalogError("Не удалось загрузить каталог дебютов.");
    } finally {
      setCatalogLoading(false);
    }
  };

  const handlePickOpening = (opening) => {
    setCatalogOpen(false);
    setCatalogQuery("");

    const title = opening.name_ru || opening.name;
    const lines = [`**${title}**${opening.eco ? ` (${opening.eco})` : ""}`];
    if (opening.name_ru && opening.name && opening.name_ru !== opening.name) {
      lines.push(`*${opening.name}*`);
    }
    if (opening.pgn) {
      lines.push(`Линия: ${opening.pgn}`);
    }
    // Финальная позиция линии - на мини-доске в сообщении.
    addMessage("ai", lines.join("\n"), opening.fen || null);
  };

  // Escape закрывает каталог, не сворачивая сам чат.
  useEffect(() => {
    if (!catalogOpen) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === "Escape") setCatalogOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [catalogOpen]);

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

    if (isStartingPosition(fen)) {
      addMessage(
        "ai",
        "Партия ещё не начата. Сделайте первый ход — и я определю дебют.",
      );
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
         * какое-то редкое имя, name_ru не будет - тогда остаётся
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
        /*
         * check-move добавляет книжные продолжения из всех дебютов,
         * покрывающих позицию. Это подсказка «что ещё играют по теории»,
         * которой нет в карточке отдельного дебюта. Ошибка проверки не
         * должна ломать основную карточку.
         */
        try {
          const theory = await checkOpeningMove(fen);
          const nextMoves = (theory?.next_moves || [])
            .map((uci) => uciToSan(fen, uci))
            .filter(Boolean);
          if (theory?.in_theory && nextMoves.length > 0) {
            lines.push(`По теории можно продолжить: ${nextMoves.join(", ")}.`);
          }
        } catch {
          // Карточка дебюта уже собрана - без этой строки она корректна.
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
          <span className="chat-avatar-glyph" aria-hidden="true"><span>♘</span></span>
        </div>
        <div className="chat-title-block">
          <h2 className="chat-panel-title">Шахматный гуру</h2>
        </div>
        <div className="chat-header-actions">
          <button
            type="button"
            className="chat-collapse-btn chat-catalog-toggle"
            onClick={openCatalog}
            title="Каталог дебютов"
            aria-label="Открыть каталог дебютов"
            aria-expanded={catalogOpen}
          >
            ♜
          </button>
          <button
            type="button"
            className="chat-collapse-btn"
            onClick={onToggleCollapsed}
            title="Свернуть чат"
            aria-label="Свернуть чат"
            aria-expanded={true}
          >
            <ChatCollapseIcon />
          </button>
        </div>
      </div>

      {catalogOpen && (
        <div className="chat-catalog" role="dialog" aria-label="Каталог дебютов">
          <div className="chat-catalog-header">
            <strong>Каталог дебютов</strong>
            <button
              type="button"
              className="chat-catalog-close"
              onClick={() => setCatalogOpen(false)}
              title="Закрыть каталог"
              aria-label="Закрыть каталог"
            >
              ×
            </button>
          </div>
          <input
            type="text"
            className="chat-catalog-search"
            value={catalogQuery}
            onChange={(e) => setCatalogQuery(e.target.value)}
            placeholder="Название, ECO или ходы"
            aria-label="Поиск дебюта"
          />
          <div className="chat-catalog-list">
            {catalogLoading && (
              <div className="chat-catalog-note">Загружаем каталог…</div>
            )}
            {catalogError && (
              <div className="chat-catalog-note chat-catalog-note--error">
                {catalogError}
              </div>
            )}
            {!catalogLoading && !catalogError && catalogResults.length === 0 && (
              <div className="chat-catalog-note">Ничего не найдено.</div>
            )}
            {catalogResults.map((opening) => (
              <button
                type="button"
                className="chat-catalog-item"
                key={`${opening.eco}-${opening.name}-${opening.pgn}`}
                onClick={() => handlePickOpening(opening)}
                title={opening.pgn || opening.name}
              >
                <span className="chat-catalog-eco">{opening.eco}</span>
                <span className="chat-catalog-name">
                  {opening.name_ru || opening.name}
                </span>
                <span className="chat-catalog-line">{opening.pgn}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div
        className="chat-messages"
        ref={messagesListRef}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
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
