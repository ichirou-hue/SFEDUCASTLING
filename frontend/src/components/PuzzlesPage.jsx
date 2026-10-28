import { useState, useEffect, useId, useRef, useCallback, useLayoutEffect } from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  askLLM,
  fetchAdaptivePuzzles,
  fetchPuzzles,
  getAccessToken,
  recordPuzzleAttempt,
} from "../api.js";
import MiniBoard from "./MiniBoard.jsx";
import ChatCollapsedStrip from "./ChatCollapsedStrip.jsx";
import PuzzleRegisterGate from "./PuzzleRegisterGate.jsx";
import PuzzlePreviewGate from "./PuzzlePreviewGate.jsx";
import SiteFooter from "./SiteFooter.jsx";
import { getCourseTopicForPuzzle, getCourseTopicMeta } from "../learningLinks.js";
import "./LearningCrossLinks.css";

const PIECE_IMAGES = {
  wK: "/pieces/white_king.svg",
  wQ: "/pieces/white_queen.svg",
  wR: "/pieces/white_rook.svg",
  wB: "/pieces/white_bishop.svg",
  wN: "/pieces/white_knight.svg",
  wP: "/pieces/white_pawn.svg",
  bK: "/pieces/black_king.svg",
  bQ: "/pieces/black_queen.svg",
  bR: "/pieces/black_rook.svg",
  bB: "/pieces/black_bishop.svg",
  bN: "/pieces/black_knight.svg",
  bP: "/pieces/black_pawn.svg",
};

const customPieces = Object.fromEntries(
  Object.entries(PIECE_IMAGES).map(([code, src]) => [
    code,
    () => <img src={src} style={{ width: "100%", height: "100%" }} />,
  ]),
);

// Пауза перед автопереключением на следующую задачу после решения (мс).
const AUTO_ADVANCE_DELAY_MS = 1800;

const THEME_LABELS = {
  mateIn1: "Мат в 1 ход",
  mateIn2: "Мат в 2 хода",
  mateIn3: "Мат в 3 хода",
  mateIn4: "Мат в 4 хода",
  fork: "Вилка",
  pin: "Связка",
  skewer: "Сквозной удар",
  discoveredAttack: "Открытая атака",
  discoveredCheck: "Шах с открывания",
  deflection: "Отвлечение",
  attraction: "Притяжение",
  sacrifice: "Жертва",
  endgame: "Эндшпиль",
  middlegame: "Миттельшпиль",
  opening: "Дебют",
  promotion: "Превращение пешки",
  backRankMate: "Мат по последней горизонтали",
  trappedPiece: "Запертая фигура",
  intermezzo: "Промежуточный ход",
  clearance: "Освобождение клетки",
  quietMove: "Тихий ход",
  hangingPiece: "Висячая фигура",
  capturingDefender: "Взятие защитника",
  advantage: "Преимущество",
  crushing: "Разгром",
  exposedKing: "Обнажённый король",
  master: "Мастерская позиция",
  mate: "Мат",
  advancedPawn: "Продвинутая пешка",
  defensiveMove: "Защитный ход",
  kingsideAttack: "Атака на королевском фланге",
  queensideAttack: "Атака на ферзевом фланге",
  bishopEndgame: "Слоновый эндшпиль",
  rookEndgame: "Ладейный эндшпиль",
  knightEndgame: "Коневой эндшпиль",
  queenEndgame: "Ферзевый эндшпиль",
  pawnEndgame: "Пешечный эндшпиль",
  mateIn5: "Мат в 5 ходов",
  smotheredMate: "Спёртый мат",
  enPassant: "Взятие на проходе",
  doubleCheck: "Двойной шах",
  underPromotion: "Тихое превращение",
};

/* Приоритет тем: чем специфичнее тактический приём, тем выше приоритет.
   Это нужно, чтобы над доской показывалась суть задачи, а не общий
   «Эндшпиль»/«Преимущество». */
const THEME_PRIORITY = [
  "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "mate",
  "smotheredMate", "backRankMate", "doubleCheck",
  "fork", "pin", "skewer", "discoveredAttack", "discoveredCheck",
  "sacrifice", "deflection", "attraction", "intermezzo", "quietMove",
  "promotion", "underPromotion", "trappedPiece", "clearance",
  "capturingDefender", "hangingPiece", "kingsideAttack", "queensideAttack",
  "exposedKing", "enPassant",
];

const THEME_ICONS = {
  mateIn1: "♚", mateIn2: "♚", mateIn3: "♚", mateIn4: "♚", mate: "♚",
  fork: "♞", pin: "♝", skewer: "♝",
  discoveredAttack: "♜", discoveredCheck: "♜",
  deflection: "♟", attraction: "♛", sacrifice: "♛",
  endgame: "♔", middlegame: "♝", opening: "♘",
  promotion: "♟", backRankMate: "♜",
  hangingPiece: "♞", capturingDefender: "♝",
  advantage: "♟", crushing: "♛", exposedKing: "♚",
  quietMove: "♟", master: "♛",
  advancedPawn: "♟", defensiveMove: "♟", kingsideAttack: "♚",
  bishopEndgame: "♝", rookEndgame: "♜", knightEndgame: "♞",
};

function getThemeInfo(puzzle) {
  const themes = puzzle.themes || [];
  const byPriority = themes
    .filter((t) => THEME_PRIORITY.includes(t))
    .sort((a, b) => THEME_PRIORITY.indexOf(a) - THEME_PRIORITY.indexOf(b));
  const pick = byPriority[0] || themes.find((t) => THEME_ICONS[t]) || themes[0];
  if (pick) {
    return {
      key: pick,
      label: THEME_LABELS[pick] || pick,
      icon: THEME_ICONS[pick] || "♟",
    };
  }
  return { key: null, label: "Тактика", icon: "♟" };
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

export default function PuzzlesPage({ user, onRegister, onOpenOnboarding }) {
  // Свой id доски - иначе доски делят один dnd-менеджер и мини-доски
  // рисуют у курсора чужую перетаскиваемую фигуру (см. Chessboard.jsx).
  const boardId = useId();

  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedTopicSlug = searchParams.get("topic");
  const requestedTopic = getCourseTopicMeta(requestedTopicSlug);
  const explicitAdaptiveMode = searchParams.get("adaptive") === "1";
  const allPuzzlesMode = searchParams.get("mode") === "all";
  const ratingMinParam = searchParams.get("rating_min");
  const ratingMin =
    ratingMinParam != null && /^\d+$/.test(ratingMinParam)
      ? Number(ratingMinParam)
      : null;
  const adaptiveMode =
    explicitAdaptiveMode ||
    (Boolean(user && getAccessToken()) &&
      !allPuzzlesMode &&
      !requestedTopicSlug &&
      ratingMin === null);

  const [puzzles, setPuzzles] = useState([]);
  const [adaptiveMeta, setAdaptiveMeta] = useState(null);
  // Пробный режим до анкеты: сколько задач доступно и не исчерпан ли лимит.
  const [previewInfo, setPreviewInfo] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [boardWidth, setBoardWidth] = useState(520);

  const [fen, setFen] = useState("");
  const [lastMove, setLastMove] = useState(null);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalMovesForSelected, setLegalMovesForSelected] = useState([]);
  const [waitingForOpponent, setWaitingForOpponent] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [resultMessage, setResultMessage] = useState(null);
  const [movesLeft, setMovesLeft] = useState(0);
  const [solvedCount, setSolvedCount] = useState(0);
  /*
   * Серия подряд правильно решённых задач (конвенция chess.com):
   * решение без ошибок - +1, любая неверная ставка обнуляет серию.
   * bestStreak - рекорд текущей сессии страницы.
   */
  const [streak, setStreak] = useState(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [hintArrow, setHintArrow] = useState(null);
  const [boardArrows, setBoardArrows] = useState([]);
  const [boardOrientation, setBoardOrientation] = useState("white");
  const [puzzleSolved, setPuzzleSolved] = useState(false);
  const [regGate, setRegGate] = useState(false);
  // Гейт анкеты на последнем ходу последней пробной задачи.
  const [previewGate, setPreviewGate] = useState(false);

  // Гейт показывается на предпоследнем ходу только незарегистрированным.
  const showRegGate = regGate && !user;

  const [chatMessages, setChatMessages] = useState([
    {
      role: "ai",
      text: "Задайте вопрос по текущей позиции, и я помогу разобраться!",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  /*
   * Чат можно свернуть в узкую полосу (как на главной): колонка чата
   * в гриде сужается, а освободившаяся ширина уходит доске.
   */
  const [chatCollapsed, setChatCollapsed] = useState(false);

  /*
   * Плавное сворачивание чата: едет третья колонка грида, поэтому
   * вместе с ней плавно расширяется колонка доски, а сама доска
   * съезжает к середине. Ширину развёрнутой колонки запоминаем
   * замером - она зависит от окна.
   */
  const layoutRef = useRef(null);
  const chatPanelRef = useRef(null);
  const naturalColRef = useRef(0);
  const interruptedColRef = useRef(0);
  // Ширина колонки доски в целевом состоянии: нужна расчёту размера,
  // пока колонки ещё едут.
  const boardColTargetRef = useRef(0);

  useEffect(() => {
    const layout = layoutRef.current;
    const chat = chatPanelRef.current;

    if (!layout || !chat || chatCollapsed) {
      return undefined;
    }

    const remember = () => {
      if (layout.classList.contains("puzzles-layout--chat-sliding")) return;
      const width = chat.getBoundingClientRect().width + 24;
      if (width > 24) naturalColRef.current = width;
    };

    remember();

    const observer = new ResizeObserver(remember);
    observer.observe(chat);

    return () => observer.disconnect();
    // loading в зависимостях: до готовой раскладки панели в DOM нет,
    // и запомнить ширину можно только после её появления.
  }, [chatCollapsed, loading]);

  useLayoutEffect(() => {
    const layout = layoutRef.current;
    const chat = chatPanelRef.current;

    if (!layout || !chat) {
      return undefined;
    }

    // Снимаем остатки прошлой анимации, иначе замерим не ту ширину.
    layout.classList.remove(
      "puzzles-layout--chat-sliding",
      "puzzles-layout--chat-reveal"
    );
    layout.style.gridTemplateColumns = "";
    chat.style.minWidth = "";
    chat.style.maxWidth = "";

    /*
     * Документ уже в целевом состоянии: запоминаем ширину колонки
     * доски - по ней посчитается размер самой доски (эффект ниже).
     */
    const boardCol = layout.querySelector(".puzzle-board-section");
    boardColTargetRef.current = boardCol ? boardCol.clientWidth : 0;

    /* На узком экране колонки идут друг под другом (см. style.css),
       и анимировать ширину нечего: состояние переключается сразу. */
    if (
      typeof window.matchMedia === "function" &&
      window.matchMedia("(max-width: 1150px)").matches
    ) {
      interruptedColRef.current = 0;
      return undefined;
    }

    const targetCol = chatCollapsed
      ? 72
      : chat.getBoundingClientRect().width + 24;
    const fromCol =
      interruptedColRef.current ||
      (chatCollapsed ? naturalColRef.current : 72);

    interruptedColRef.current = 0;

    if (!fromCol || Math.abs(targetCol - fromCol) < 1) {
      return undefined;
    }

    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = reduceMotion ? 0 : 340;
    const easing = "cubic-bezier(0.4, 0, 0.2, 1)";

    // Средняя колонка остаётся гибкой: она и забирает освободившуюся
    // ширину, пока крайняя колонка едет.
    const [infoCol] = getComputedStyle(layout).gridTemplateColumns.split(" ");
    const track = (width) =>
      `${infoCol} minmax(0, 1fr) ${width}px`;

    layout.classList.add("puzzles-layout--chat-sliding");
    // Свёрнутая панель ограничена 48px, развёрнутая - 320px: на время
    // проезда ограничения снимаем, иначе панель не тянется за колонкой.
    chat.style.minWidth = "0";
    chat.style.maxWidth = "none";

    const animation = layout.animate(
      [
        { gridTemplateColumns: track(fromCol) },
        { gridTemplateColumns: track(targetCol) },
      ],
      { duration, easing }
    );

    const finish = () => {
      layout.style.gridTemplateColumns = "";
      chat.style.minWidth = "";
      chat.style.maxWidth = "";
      layout.classList.remove("puzzles-layout--chat-sliding");
      layout.classList.add("puzzles-layout--chat-reveal");

      if (!chatCollapsed) {
        naturalColRef.current = chat.getBoundingClientRect().width + 24;
      }

      window.setTimeout(() => {
        layout.classList.remove("puzzles-layout--chat-reveal");
      }, 260);
    };

    animation.addEventListener("finish", finish, { once: true });

    return () => {
      if (animation.playState === "running") {
        interruptedColRef.current = chat.getBoundingClientRect().width + 24;
      }

      animation.cancel();
    };
  }, [chatCollapsed]);

  const gameRef = useRef(new Chess());
  const solutionMovesRef = useRef([]);
  const solutionIndexRef = useRef(0);
  const puzzlesRef = useRef([]);
  // Уникальные задачи, решённые за текущую загрузку страницы. Повторное
  // решение той же задачи не приближает пробный лимит.
  const solvedPuzzleIdsRef = useRef(new Set());
  const loadSeqRef = useRef(0);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    // Замеряем только готовую раскладку: на экране загрузки её нет
    // в DOM (другая ветка), querySelector вернул бы null и размер
    // посчитался бы от окна — с переполнением и без пересчёта.
    if (loading) return;
    const updateSize = () => {
      // Ширина — по формуле от окна (колонки грида фиксированы).
      // Свёрнутый чат занимает 48px полосы + 24px отступа, остальное
      // достаётся доске (см. .puzzles-layout--chat-collapsed).
      const isMobile = window.matchMedia("(max-width: 760px)").matches;
      const infoW = isMobile ? 0 : 230;
      const chatReserve = chatCollapsed ? 64 : 560;
      const hGaps = isMobile ? 48 : 64; // gaps + паддинги + скроллбар
      const layoutEl = document.querySelector(".puzzles-layout");
      const availW = isMobile
        ? (layoutEl?.clientWidth || window.innerWidth) - hGaps
        : window.innerWidth - infoW - chatReserve - hGaps;
      // Высота — видимая область страницы (без топбара и футера), а не
      // высота самой раскладки: она теперь растягивается по содержимому,
      // и замер её clientHeight дал бы петлю (доска растит саму себя).
      const belowBoard = 175;
      const shellEl = document.querySelector(".puzzles-page-shell");
      const footerEl = document.querySelector(
        ".puzzles-page-shell > .site-footer"
      );
      const viewH = shellEl
        ? shellEl.clientHeight - (footerEl?.offsetHeight || 0)
        : window.innerHeight;
      const availH = viewH - belowBoard;
      // Страховка от переполнения колонки: доска никогда не шире
      // реальной ширины своей грид-колонки минус рамка обёртки (38px).
      // Иначе доска залезает на инфо-панель и чат (так и было).
      const colEl = document.querySelector(".puzzle-board-section");
      /*
       * Пока колонка чата едет, живой замер колонки доски показывает
       * промежуточную ширину - берём целевую, её запомнила анимация.
       */
      const targetCol = boardColTargetRef.current;
      boardColTargetRef.current = 0;
      const boardChrome = isMobile ? 22 : 38;
      const colW = targetCol
        ? targetCol - boardChrome
        : colEl
          ? colEl.clientWidth - boardChrome
          : Infinity;
      const size = Math.floor(
        Math.min(Math.max(availW, isMobile ? 240 : 280), Math.max(availH, 280), colW),
      );
      setBoardWidth(Math.max(isMobile ? 240 : 320, Math.min(size, 780)));
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [loading, chatCollapsed]);

  const countPlayerMoves = useCallback((movesStr) => {
    const moves = movesStr.trim().split(/\s+/).filter(Boolean);
    return Math.ceil(moves.length / 2);
  }, []);

  const loadPuzzle = useCallback(
    (index) => {
      const puzzleList = puzzlesRef.current;
      if (!puzzleList[index]) return;
      const puzzle = puzzleList[index];

      const seq = loadSeqRef.current + 1;
      loadSeqRef.current = seq;

      const game = new Chess(puzzle.fen);
      gameRef.current = game;
      setFen(game.fen());
      setSelectedSquare(null);
      setLegalMovesForSelected([]);
      setWaitingForOpponent(false);
      setShowSolution(false);
      setResultMessage(null);
      setLastMove(null);
      setHintArrow(null);
      setBoardArrows([]);
      setPuzzleSolved(false);
      setRegGate(false);
      setPreviewGate(false);

      const turn = puzzle.fen.split(" ")[1];
      setBoardOrientation(turn === "b" ? "black" : "white");

      const moves = (puzzle.moves || "").trim().split(/\s+/).filter(Boolean);
      solutionMovesRef.current = moves;
      solutionIndexRef.current = 0;
      setMovesLeft(countPlayerMoves(puzzle.moves));
    },
    [countPlayerMoves],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSolvedCount(0);
    solvedPuzzleIdsRef.current = new Set();
    // Новый набор задач - серия начинается заново.
    setStreak(0);
    setBestStreak(0);
    setAdaptiveMeta(null);
    setPreviewInfo(null);

    const request = adaptiveMode
      ? fetchAdaptivePuzzles(20)
      : fetchPuzzles(20, requestedTopic?.slug || null, ratingMin);

    request
      .then((data) => {
        if (cancelled) return;
        const qs = data.puzzles || [];
        const preview = data.preview === true;
        setPreviewInfo(
          preview
            ? {
                locked: Boolean(data.preview_locked),
                remaining: Number(data.preview_remaining ?? 0),
                limit: Number(data.preview_limit ?? 3),
              }
            : null,
        );
        if (qs.length === 0) {
          puzzlesRef.current = [];
          setPuzzles([]);
          if (!(preview && data.preview_locked)) {
            setError(
              adaptiveMode
                ? "Пока не удалось подобрать персональную тренировку."
                : requestedTopic
                  ? `Для темы «${requestedTopic.title}» пока нет доступных пазлов.`
                  : ratingMin != null
                    ? `Пока нет доступных пазлов с рейтингом от ${ratingMin}.`
                    : "Нет доступных задач",
            );
          }
          return;
        }
        if (adaptiveMode) {
          setAdaptiveMeta({
            weakTopics: data.weak_topics || [],
            strongTopics: data.strong_topics || [],
            allocation: data.allocation || null,
            playerLevel: data.player_level || null,
          });
        }
        puzzlesRef.current = qs;
        setPuzzles(qs);
        setCurrentIndex(0);
      })
      .catch((e) => {
        if (cancelled) return;
        setError("Ошибка загрузки задач: " + (e.response?.data?.detail || e.message));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [adaptiveMode, ratingMin, requestedTopic?.slug, requestedTopic?.title]);

  useEffect(() => {
    if (puzzles.length > 0 && puzzles[currentIndex]) {
      loadPuzzle(currentIndex);
    }
  }, [currentIndex, puzzles, loadPuzzle]);

  const currentPuzzle = puzzles[currentIndex];
  const themeInfo = currentPuzzle
    ? getThemeInfo(currentPuzzle)
    : { key: null, label: "Тактика", icon: "♟" };
  // Сторона, за которую нужно найти ходы, — это очередь хода из FEN
  // (в lichess-паззлах игрок всегда ходит первым). boardOrientation
  // выставляется по тому же признаку в loadPuzzle.
  const sideToMove = currentPuzzle?.fen?.split(" ")?.[1] === "b" ? "b" : "w";

  // Показываем персонализацию, только когда профиль настоящий с обеих
  // сторон: есть подтверждённые попытками слабые И сильные темы.
  // Без истории бэкенд присылает фабрикат (см. rank_topic_profile).
  const observedWeakTopics = (adaptiveMeta?.weakTopics || []).filter(
    (topic) => topic.observed || (topic.attempts || 0) > 0,
  );
  const observedStrongTopics = (adaptiveMeta?.strongTopics || []).filter(
    (topic) => topic.observed || (topic.attempts || 0) > 0,
  );
  const hasRealProfile =
    observedWeakTopics.length > 0 && observedStrongTopics.length > 0;
  const relatedCourseTopic =
    getCourseTopicMeta(currentPuzzle?.adaptive_topic) ||
    requestedTopic ||
    getCourseTopicForPuzzle(currentPuzzle);

  // Пробный доступ исчерпан: либо бэкенд уже вернул пусто (лимит на сегодня
  // выбран), либо на третьей решённой задаче открыт гейт анкеты.
  const previewLocked =
    Boolean(previewInfo) && (previewInfo.locked || previewGate);

  // Гейт анкеты показываем на последнем ходу ТРЕТЬЕЙ решённой за сегодня
  // задачи, а не на последней задаче в списке. previewInfo.remaining —
  // сколько ещё можно решить сегодня с учётом прошлых заходов, поэтому
  // в этой сессии гейт включается после (remaining - 1) новых решений.
  // Какую именно из оставшихся задач пользователь решит третьей — не важно.
  const previewGateAfterSolved = previewInfo
    ? Math.max(0, previewInfo.remaining - 1)
    : null;
  const isPreviewGatePuzzle =
    Boolean(previewInfo) &&
    !previewInfo.locked &&
    puzzles.length > 0 &&
    solvedCount >= previewGateAfterSolved &&
    !(currentPuzzle?.id && solvedPuzzleIdsRef.current.has(currentPuzzle.id));

  // «Позже» в пробном гейте: если задача на доске есть — просто закрываем
  // напоминание, доска остаётся на последнем ходу без сделанного хода.
  // Если задач нет (дневной лимит выбран на загрузке) — уходим на главную.
  const handlePreviewLater = useCallback(() => {
    if (puzzlesRef.current.length > 0) {
      setPreviewGate(false);
      return;
    }
    navigate("/");
  }, [navigate]);

  const savePuzzleProgress = useCallback(
    (correct) => {
      if (!currentPuzzle?.id || !getAccessToken()) return;
      recordPuzzleAttempt(currentPuzzle.id, correct)
        .then(() => {
          window.dispatchEvent(new Event("sfedu-progress-updated"));
        })
        .catch(() => {
          // Ошибка статистики не должна мешать решению самой задачи.
        });
    },
    [currentPuzzle?.id],
  );

  const handleMove = useCallback(
    (sourceSquare, targetSquare) => {
      if (waitingForOpponent || showSolution || showRegGate || previewGate) return false;

      const expectedMove = solutionMovesRef.current[solutionIndexRef.current];
      if (!expectedMove) return false;

      const expectedCore = expectedMove.substring(0, 4);
      const moveCore = sourceSquare + targetSquare;

      if (moveCore === expectedCore) {
        // Третья решённая за сегодня задача: не даём дорешить без анкеты.
        // Вместо завершающего (решающего) хода показываем окно с анкетой.
        if (
          isPreviewGatePuzzle &&
          solutionIndexRef.current === solutionMovesRef.current.length - 1
        ) {
          setPreviewGate(true);
          return true;
        }

        const game = gameRef.current;
        const promo = expectedMove.length > 4 ? expectedMove[4] : "q";
        const move = game.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: promo,
        });
        if (!move) return false;

        setFen(game.fen());
        setLastMove({ from: sourceSquare, to: targetSquare });
        setSelectedSquare(null);
        setLegalMovesForSelected([]);
        setHintArrow(null);
        setBoardArrows([]);
        solutionIndexRef.current++;

        const totalMoves = solutionMovesRef.current.length;
        const remaining = totalMoves - solutionIndexRef.current;
        setMovesLeft(Math.ceil(remaining / 2));

        if (solutionIndexRef.current >= totalMoves) {
          setResultMessage({ type: "success", text: "Правильно! Задача решена!" });
          if (currentPuzzle?.id) {
            solvedPuzzleIdsRef.current.add(currentPuzzle.id);
          }
          setSolvedCount(solvedPuzzleIdsRef.current.size);
          // Серия растёт: задача решена без неверных ставок
          // (ошибка в ветке ниже уже обнулила счётчик).
          const nextStreak = streak + 1;
          setStreak(nextStreak);
          setBestStreak((b) => Math.max(b, nextStreak));
          setPuzzleSolved(true);
          savePuzzleProgress(true);
          return true;
        }

        // Предпоследний ход игрока сделан (остался ответ соперника и финальный
        // ход) — незарегистрированный обязан зарегистрироваться, дальше нельзя.
        if (remaining === 2 && !user) setRegGate(true);

        if (
          solutionIndexRef.current % 2 === 1 &&
          solutionIndexRef.current < totalMoves
        ) {
          setWaitingForOpponent(true);
          const seqAtStart = loadSeqRef.current;
          setTimeout(() => {
            if (seqAtStart !== loadSeqRef.current) return;
            const oppUci =
              solutionMovesRef.current[solutionIndexRef.current];
            if (oppUci) {
              const oppFrom = oppUci.substring(0, 2);
              const oppTo = oppUci.substring(2, 4);
              const oppPromo =
                oppUci.length > 4 ? oppUci[4] : undefined;
              const oppMove = game.move({
                from: oppFrom,
                to: oppTo,
                promotion: oppPromo || "q",
              });
              if (oppMove) {
                setFen(game.fen());
                setLastMove({ from: oppFrom, to: oppTo });
                solutionIndexRef.current++;
                const rem = totalMoves - solutionIndexRef.current;
                setMovesLeft(Math.ceil(rem / 2));
              }
            }
            setWaitingForOpponent(false);
          }, 400);
        }

        return true;
      }

      // Ошибка рвёт серию: счётчик обнуляется, а если серия была,
      // сообщаем об этом в сообщении над кнопками.
      if (streak > 0) {
        setResultMessage({
          type: "error",
          text: `Неверный ход. Серия из ${streak} прервана.`,
        });
      } else {
        setResultMessage({
          type: "error",
          text: "Неверный ход. Попробуйте ещё раз.",
        });
      }
      setStreak(0);
      savePuzzleProgress(false);
      return false;
    },
    [
      waitingForOpponent,
      showSolution,
      showRegGate,
      previewGate,
      isPreviewGatePuzzle,
      currentPuzzle?.id,
      user,
      savePuzzleProgress,
      streak,
    ],
  );

  const onSquareClick = useCallback(
    (square) => {
      if (waitingForOpponent || showSolution || showRegGate || previewGate) return;

      const game = gameRef.current;
      const piece = game.get(square);

      if (selectedSquare) {
        if (piece && piece.color === game.turn()) {
          if (square === selectedSquare) {
            setSelectedSquare(null);
            setLegalMovesForSelected([]);
            return;
          }
          setSelectedSquare(square);
          const moves = game.moves({ square, verbose: true });
          setLegalMovesForSelected(moves.map((m) => m.to));
          return;
        }

        const success = handleMove(selectedSquare, square);
        if (success) return;

        setSelectedSquare(null);
        setLegalMovesForSelected([]);
        return;
      }

      if (piece && piece.color === game.turn()) {
        setSelectedSquare(square);
        const moves = game.moves({ square, verbose: true });
        setLegalMovesForSelected(moves.map((m) => m.to));
      }
    },
    [selectedSquare, waitingForOpponent, showSolution, showRegGate, previewGate, handleMove],
  );

  const handleShowSolution = () => {
    if (!currentPuzzle) return;
    const seqAtStart = loadSeqRef.current;
    const game = new Chess(currentPuzzle.fen);
    const moves = solutionMovesRef.current;

    setShowSolution(true);
    setResultMessage(null);
    setHintArrow(null);

    let i = 0;
    const playNext = () => {
      if (seqAtStart !== loadSeqRef.current) return;
      if (i >= moves.length) return;
      const uci = moves[i];
      const from = uci.substring(0, 2);
      const to = uci.substring(2, 4);
      const promo = uci.length > 4 ? uci[4] : undefined;
      const move = game.move({ from, to, promotion: promo || "q" });
      if (move) {
        setFen(game.fen());
        setLastMove({ from, to });
        i++;
        if (i < moves.length) {
          setTimeout(playNext, 600);
        }
      }
    };

    setTimeout(playNext, 300);
  };

  const handleShowHint = () => {
    if (!currentPuzzle) return;
    const nextUci = solutionMovesRef.current[solutionIndexRef.current];
    if (!nextUci) return;

    const from = nextUci.substring(0, 2);
    const to = nextUci.substring(2, 4);

    setHintArrow({ from, to });
    setBoardArrows([[from, to, "rgba(0, 150, 50, 0.85)"]]);
    setResultMessage({
      type: "hint",
      text: `Подсказка: попробуйте ход с ${from} на ${to}`,
    });
  };

  const handleNextPuzzle = () => {
    if (currentIndex < puzzles.length - 1) {
      setCurrentIndex((i) => i + 1);
    }
  };

  // Автопереключение после решения: даём полюбоваться оверлеем «Задача
  // решена!» и переходим к следующей. На последней задаче таймер не
  // ставим — оверлей остаётся, пользователь решает сам. Ручное навигирование
  // (кнопки «→»/«←») меняет currentIndex → таймер отменяется, а новая
  // задача сбрасывает puzzleSolved в loadPuzzle.
  useEffect(() => {
    if (!puzzleSolved) return undefined;
    if (currentIndex >= puzzles.length - 1) return undefined;

    const timer = setTimeout(() => {
      setCurrentIndex((i) => i + 1);
    }, AUTO_ADVANCE_DELAY_MS);

    return () => clearTimeout(timer);
  }, [puzzleSolved, currentIndex, puzzles.length]);

  const handlePrevPuzzle = () => {
    if (currentIndex > 0) {
      setCurrentIndex((i) => i - 1);
    }
  };

  const handleRestartPuzzle = () => {
    loadPuzzle(currentIndex);
  };

  const handleChatSend = async () => {
    const text = chatInput.trim();
    if (!text || chatLoading) return;
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", text }]);

    setChatLoading(true);
    try {
      const data = await askLLM(text, fen, [], false);
      const reply = data.reply || "Не понял вопрос.";
      setChatMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: reply,
          fen: data.fen || null,
          arrows: data.arrow ? [data.arrow] : null,
        },
      ]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: "ai", text: "Ошибка соединения с сервером." },
      ]);
    }
    setChatLoading(false);
  };

  const customSquareStyles = {};

  // Подсветка последнего хода — базовая. Точки возможных ходов накладываем
  // поверх неё, иначе клетка, совпавшая с ходом соперника, теряет точку.
  if (lastMove) {
    customSquareStyles[lastMove.from] = {
      backgroundColor: "rgba(201, 169, 110, 0.45)",
    };
    customSquareStyles[lastMove.to] = {
      backgroundColor: "rgba(201, 169, 110, 0.6)",
    };
  }

  if (selectedSquare) {
    customSquareStyles[selectedSquare] = {
      backgroundColor: "rgba(201, 169, 110, 0.55)",
    };
  }

  for (const sq of legalMovesForSelected) {
    const targetPiece = gameRef.current.get(sq);
    const dotStyle = targetPiece
      ? {
          backgroundImage:
            "radial-gradient(circle, transparent 48%, rgba(100,80,40,0.25) 50%, rgba(100,80,40,0.25) 80%, transparent 82%)",
          backgroundPosition: "center",
          backgroundSize: "100% 100%",
          backgroundRepeat: "no-repeat",
        }
      : {
          backgroundImage:
            "radial-gradient(circle, rgba(100,80,40,0.2) 15%, transparent 16%)",
          backgroundPosition: "center",
          backgroundSize: "100% 100%",
          backgroundRepeat: "no-repeat",
        };
    customSquareStyles[sq] = { ...(customSquareStyles[sq] || {}), ...dotStyle };
  }

  if (loading) {
    return (
      <div className="puzzles-page">
        <div className="puzzles-loading">Загрузка задач...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="puzzles-page">
        <div className="puzzles-error">{error}</div>
      </div>
    );
  }

  // Пробные задачи кончились ещё на загрузке: отдельная страница без доски.
  if (previewLocked && puzzles.length === 0) {
    return (
      <div className="puzzles-page">
        <PuzzlePreviewGate
          onOpenOnboarding={onOpenOnboarding}
          onLater={handlePreviewLater}
        />
        <SiteFooter />
      </div>
    );
  }

  return (
    <div className="puzzles-page-shell">
      <div
        ref={layoutRef}
        className={`puzzles-layout${
          chatCollapsed ? " puzzles-layout--chat-collapsed" : ""
        }`}
      >
      <div className="puzzle-info-panel">
        <div className="puzzle-theme-label">Тема задачи:</div>
        <div className="puzzle-theme-name">{themeInfo.label}</div>
        <div className="puzzle-theme-icon">{themeInfo.icon}</div>

        <div className="puzzle-instruction">
          Найдите
          <br />
          лучший ход в
          <br />
          позиции
        </div>

        {currentPuzzle && (
          <div className="puzzle-turn-indicator">
            Ход {currentPuzzle.fen.split(" ")[1] === "b" ? "чёрных" : "белых"}
          </div>
        )}
        <div className="puzzle-side">
          <span className={`puzzle-side-dot puzzle-side-dot--${sideToMove}`} aria-hidden="true" />
          Играете за {sideToMove === "b" ? "чёрных" : "белых"}
        </div>

        <div className="puzzle-moves-left-label">
          Осталось
          <br />
          найти ходов:
        </div>
        <div className="puzzle-moves-left-count">{movesLeft}</div>

        {currentPuzzle && (
          <div className="puzzle-rating">
            Рейтинг: {currentPuzzle.rating}
          </div>
        )}

        {/*
         * Серия подряд правильно решённых задач. Показываем всегда
         * (в том числе ноль), чтобы блок не прыгал при обнулении.
         */}
        <div className="puzzle-streak">
          <div className="puzzle-streak-label">Серия:</div>
          <div
            className={`puzzle-streak-value ${
              streak > 0 ? "puzzle-streak-value--hot" : ""
            }`}
          >
            🔥 {streak}
          </div>
          {bestStreak > 0 && (
            <div className="puzzle-streak-best">
              рекорд сессии: {bestStreak}
            </div>
          )}
        </div>

        {adaptiveMode && hasRealProfile && (
          <div className="learning-topic-filter learning-topic-filter--adaptive">
            Персональная тренировка
            {adaptiveMeta?.playerLevel && (
              <strong>
                {adaptiveMeta.playerLevel.provisional ? "Предварительный уровень" : "Уровень"}: {adaptiveMeta.playerLevel.title}
              </strong>
            )}
            {currentPuzzle?.adaptive_difficulty != null && (
              <span className="adaptive-topic-list">
                Целевая сложность: {currentPuzzle.adaptive_difficulty} из 3
                {currentPuzzle.difficulty_match === false && " · задача вне целевого диапазона"}
              </span>
            )}
            <strong>Подбор по уровню и темам</strong>
            {adaptiveMeta?.weakTopics?.length > 0 && (
              <div className="adaptive-topic-list">
                Слабые: {adaptiveMeta.weakTopics.map((topic) => topic.title).join(", ")}
              </div>
            )}
            <button type="button" onClick={() => navigate("/puzzles?mode=all")}>
              Обычный режим
            </button>
          </div>
        )}

        {adaptiveMode && !hasRealProfile && (
          <div className="learning-topic-filter learning-topic-filter--adaptive">
            <button type="button" onClick={() => navigate("/puzzles")}>
              Обычный режим
            </button>
          </div>
        )}

        {!adaptiveMode && requestedTopic && (
          <div className="learning-topic-filter">
            Практика по теме
            <strong>
              {requestedTopic.icon} {requestedTopic.title}
            </strong>
            <button type="button" onClick={() => navigate("/puzzles?mode=all")}>
              Показать все пазлы
            </button>
          </div>
        )}

        {!adaptiveMode && ratingMin != null && (
          <div className="learning-topic-filter">
            Подбор по рейтингу
            <strong>От {ratingMin}</strong>
            <button type="button" onClick={() => navigate("/puzzles?mode=all")}>
              Показать все пазлы
            </button>
          </div>
        )}

        {relatedCourseTopic && (
          <button
            type="button"
            className="learning-crosslink learning-crosslink--compact"
            onClick={() =>
              navigate(
                `/training?module=${encodeURIComponent(relatedCourseTopic.slug)}`,
              )
            }
          >
            Пройти урок по теме →
          </button>
        )}

        <div className="puzzle-nav-buttons">
          <button
            className="puzzle-nav-btn"
            onClick={handlePrevPuzzle}
            disabled={currentIndex === 0}
          >
            ←
          </button>
          <span className="puzzle-nav-index">
            {currentIndex + 1} / {puzzles.length}
          </span>
          <button
            className="puzzle-nav-btn"
            onClick={handleNextPuzzle}
            disabled={currentIndex >= puzzles.length - 1}
          >
            →
          </button>
        </div>
      </div>

      <div className="puzzle-board-section">
        <div className={`puzzle-board-wrapper ${puzzleSolved ? "puzzle-board-wrapper--solved" : ""}`} data-logo-anchor>
          <ReactChessboard
            id={boardId}
            position={fen}
            onPieceDrop={handleMove}
            onSquareClick={onSquareClick}
            boardWidth={boardWidth}
            animationDuration={200}
            boardOrientation={boardOrientation}
            showBoardNotation={true}
            customNotationStyle={{
              fontSize: "12px",
              fontFamily: "Cormorant Garamond, Georgia, serif",
              fontWeight: "500",
              color: "#4a3220",
            }}
            areArrowsAllowed={true}
            customArrows={boardArrows}
            customArrowColor="rgba(0, 150, 50, 0.85)"
            customPieces={customPieces}
            customBoardStyle={{
              borderRadius: "15px",
              backgroundImage:
                "linear-gradient(0deg, rgba(74, 178, 45, 0.43) 0%, rgba(74, 178, 45, 0.43) 100%), url(/textures/green-marble.png)",
              backgroundPosition: "-0.111px 0px",
              backgroundSize: "100.027% 100%",
              backgroundRepeat: "no-repeat",
              backgroundColor: "lightgray",
            }}
            customDarkSquareStyle={{
              boxShadow:
                "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
              backgroundColor: "rgba(223, 239, 252, 0.20)",
            }}
            customLightSquareStyle={{
              boxShadow:
                "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
              backgroundImage: "url(/textures/white-marble.png)",
              backgroundPosition: "50%",
              backgroundSize: "cover",
              backgroundRepeat: "no-repeat",
              backgroundColor: "rgba(255, 255, 255, 0.75)",
            }}
            customSquareStyles={customSquareStyles}
          />
          {puzzleSolved && (
            <div className="puzzle-solved-overlay">
              <div className="puzzle-solved-checkmark">✓</div>
              <div className="puzzle-solved-text">Задача решена!</div>
              {streak > 0 && (
                <div className="puzzle-solved-streak">
                  🔥 Серия: {streak}
                </div>
              )}
              {currentIndex < puzzles.length - 1 && (
                <div className="puzzle-solved-hint">
                  Следующая задача откроется автоматически…
                </div>
              )}
            </div>
          )}
        </div>

        {resultMessage && (
          <div className={`puzzle-result puzzle-result--${resultMessage.type}`}>
            {resultMessage.text}
          </div>
        )}

        <div className="puzzle-controls">
          <button
            className="puzzle-ctrl-btn puzzle-ctrl-btn--restart"
            onClick={handleRestartPuzzle}
          >
            Сброс
          </button>
          <button
            className="puzzle-ctrl-btn puzzle-ctrl-btn--solution"
            onClick={handleShowSolution}
            disabled={showSolution}
          >
            Посмотреть решение
          </button>
          <button
            className="puzzle-ctrl-btn puzzle-ctrl-btn--hint"
            onClick={handleShowHint}
          >
            Подсказка
          </button>
        </div>
      </div>

      {chatCollapsed ? (
        <div
          className="puzzle-chat-section puzzle-chat-section--collapsed"
          ref={chatPanelRef}
        >
          <ChatCollapsedStrip onExpand={() => setChatCollapsed(false)} />
        </div>
      ) : (
        <div className="puzzle-chat-section" ref={chatPanelRef}>
          <div className="chat-header">
            <div className="chat-avatar">
              <img src="/bot-icon.svg" alt="Ассистент" />
            </div>
            <div className="chat-title-block">
              <h2>ЧАТ ПО ЗАДАЧЕ</h2>
              <span className="chat-subtitle">Шахматный ассистент</span>
            </div>
            <button
              type="button"
              className="chat-collapse-btn"
              onClick={() => setChatCollapsed(true)}
              title="Свернуть чат"
              aria-label="Свернуть чат"
              aria-expanded={true}
            >
              »
            </button>
          </div>

          <div className="chat-messages">
            {chatMessages.map((msg, i) => (
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
                      width={220}
                      lastMove={msg.lastMove}
                      arrows={msg.arrows || null}
                    />
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="chat-msg">
                <div className="msg-avatar msg-avatar--ai">♞</div>
                <div className="msg-body">Думаю...</div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleChatSend()}
              placeholder="Введите ваше сообщение..."
            />
            <button onClick={handleChatSend} disabled={chatLoading}>
              ➤
            </button>
          </div>
        </div>
      )}
      <PuzzleRegisterGate open={showRegGate} onRegister={onRegister} />
      </div>
      {previewLocked && puzzles.length > 0 && (
        <PuzzlePreviewGate
          onOpenOnboarding={onOpenOnboarding}
          onLater={handlePreviewLater}
        />
      )}
      <SiteFooter />
    </div>
  );
}
