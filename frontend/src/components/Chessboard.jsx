import {
  useState,
  useRef,
  useCallback,
  useEffect,
  forwardRef,
  useId,
  useImperativeHandle,
} from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";

import {
  fetchMaiaMove,
  fetchEval,
  fetchCompareMoves,
  saveMoveToDataset,
  fetchExplainMove,
  finishGame,
  reviewGame,
  getStoredUser,
} from "../api.js";

/*
 * id залогиненного пользователя (число) или null для анонима.
 * Ходы и партии анонимов уходят в БД с user_id = NULL.
 * Используем getStoredUser из api.js: ключ хранения - gigachess_user.
 */
function getCurrentUserId() {
  const id = getStoredUser()?.id;
  return Number.isFinite(id) ? id : null;
}

let gameId = "game_" + Date.now();

/*
 * Имена файлов фигур в public/pieces. Ключи - как в chess.js
 * (piece.type). Нужны, чтобы рисовать иконки съеденного материала
 * и «призрак» фигуры на клетке предхода.
 */
const PIECE_FILE_NAMES = {
  p: "pawn",
  n: "knight",
  b: "bishop",
  r: "rook",
  q: "queen",
  k: "king",
};

/*
 * Небольшая задержка перед выполнением предхода: даём доске
 * проиграть анимацию ответа Maia3, чтобы игрок успел увидеть,
 * на что именно он отвечает.
 */
const PREMOVE_EXECUTION_DELAY = 220;

/*
 * Порядок иконок в плашке перевеса: от пешки к ферзю, как в списке
 * съеденных фигур на chess.com.
 */
const MATERIAL_PIECE_ORDER = ["p", "n", "b", "r", "q"];

/*
 * Начальная позиция: по ней понимаем, что партия только началась -
 * нужно для первого хода Maia3, когда играем чёрными.
 */
const START_POSITION =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/*
 * История ходов в SAN -> список UCI от начальной позиции.
 * Нужен и для сохранения партии, и для разбора: сервер работает
 * только с UCI. Если история начинается не с начальной позиции
 * (загружен FEN), разбор прекращается на первом же несоответствии.
 */
function toUciMoves(sanMoves) {
  const replay = new Chess();
  const moves = [];

  for (const san of sanMoves) {
    try {
      const move = replay.move(san);
      if (move) {
        moves.push(move.from + move.to + (move.promotion || ""));
      }
    } catch {
      break;
    }
  }

  return moves;
}

/*
 * Подписи для итогов разбора: что именно произошло с ходами игрока.
 * Порядок - от лучших к худшим, чтобы строка читалась слева направо.
 */
const REVIEW_COUNT_LABELS = [
  ["best", "лучших"],
  ["excellent", "отличных"],
  ["good", "хороших"],
  ["inaccuracy", "неточностей"],
  ["mistake", "ошибок"],
  ["blunder", "зевков"],
];

function formatAccuracy(accuracy) {
  if (accuracy == null) return "-";
  return `${accuracy.toFixed(1).replace(".", ",")}%`;
}

/*
 * Меньше этого числа оцененных ходов - процент строить рано: на выборке
 * в три-четыре хода он гуляет на несколько пунктов от прогона к прогону,
 * а один ход решает всё. Порог тот же по смыслу, что MIN_SCORED_MOVES
 * в точности по этапам.
 */
const MIN_RELIABLE_MOVES = 8;

/*
 * Описание итога партии для карточки. На мате ход принадлежит
 * проигравшей стороне: если это сторона игрока - поражение, иначе победа.
 */
function describeGameOver(g, playerColor) {
  if (g.isCheckmate()) {
    return g.turn() === playerColor
      ? {
          outcome: "loss",
          reason: `Мат - победили ${playerColor === "w" ? "чёрные" : "белые"}`,
        }
      : { outcome: "win", reason: "Мат! Вы победили" };
  }
  if (g.isStalemate()) {
    return { outcome: "draw", reason: "Пат - ничья" };
  }
  if (g.isThreefoldRepetition()) {
    return { outcome: "draw", reason: "Троекратное повторение - ничья" };
  }
  if (g.isInsufficientMaterial()) {
    return { outcome: "draw", reason: "Недостаточно материала - ничья" };
  }
  if (g.isDrawByFiftyMoves()) {
    return { outcome: "draw", reason: "Правило 50 ходов - ничья" };
  }
  return { outcome: "draw", reason: "Ничья" };
}

function GameReviewSummary({ review }) {
  const counts = review.counts || {};
  const parts = REVIEW_COUNT_LABELS.filter(
    ([key]) => (counts[key] || 0) > 0,
  ).map(([key, label]) => `${label}: ${counts[key]}`);

  /*
   * Подсказка, сколько точности не хватает до следующей звезды:
   * без неё «ноль звёзд» выглядит как поломка, а не как результат.
   * Потолок по результату партии учитываем: за поражение больше одной
   * звезды не дают, и обещать «до 2 звёзд» нельзя.
   */
  const thresholds = review.thresholds || {};
  const stars = Number.isFinite(review.stars) ? review.stars : 0;
  const cap = Number.isFinite(review.star_cap) ? review.star_cap : 3;
  const nextStar = stars < cap ? stars + 1 : null;
  const nextThreshold =
    nextStar !== null && thresholds[String(nextStar)] != null
      ? thresholds[String(nextStar)]
      : null;

  /*
   * Почему звёзд меньше, чем позволяет точность. Без этой строки
   * «1 звезда за 100% точности» выглядит как ошибка подсчёта.
   */
  const capNote =
    stars >= cap && review.outcome === "loss"
      ? "За поражение максимум одна звезда"
      : stars >= cap && review.outcome === "draw"
        ? "За ничью максимум две звезды"
        : null;

  return (
    <div className="gameover-review">
      <div
        className="gameover-stars"
        aria-label={`Оценка партии: ${stars} из 3 звёзд`}
      >
        {[1, 2, 3].map((star) => (
          <span
            key={star}
            className={`gameover-star${
              star <= stars ? " gameover-star--on" : ""
            }`}
            aria-hidden="true"
          >
            ★
          </span>
        ))}
      </div>
      <div className="gameover-review-accuracy">
        Точность: {formatAccuracy(review.accuracy)}
      </div>
      <div className="gameover-review-detail">
        {parts.length > 0 ? parts.join(" · ") : "Ходы игрока не разобраны"}
      </div>
      {nextThreshold !== null && review.accuracy != null && (
        <div className="gameover-review-note">
          До {nextStar} {nextStar === 1 ? "звезды" : "звёзд"} нужно{" "}
          {String(nextThreshold).replace(".", ",")}% точности
        </div>
      )}
      {capNote && <div className="gameover-review-note">{capNote}</div>}
      {review.truncated && (
        <div className="gameover-review-note">
          Разобраны первые {review.moves_analyzed} ходов
        </div>
      )}
      {review.moves_decided > 0 && (
        <div className="gameover-review-note">
          {/*
           * Точность по ходам из решённых позиций не считается: когда
           * партия уже выиграна или проиграна, любой ход сохраняет те же
           * шансы и получает 100 %. Без этой строки «45 % за 17 ходов»
           * выглядит как ошибка подсчёта.
           */}
          В точность вошли {review.moves_scored ?? 0} из{" "}
          {review.moves_analyzed ?? 0} ходов: остальные сделаны в уже
          решённой позиции
        </div>
      )}
      {Number.isFinite(review.moves_scored) &&
        review.moves_scored < MIN_RELIABLE_MOVES && (
          <div className="gameover-review-note">
            {/*
             * Процент по трём-четырём ходам гуляет на несколько пунктов
             * от прогона к прогону: движок ищет 300 мс, и оценка соседних
             * позиций чуть разная. Честнее предупредить, чем показывать
             * «55,8 %» как приговор всей игре.
             */}
            Оценено мало ходов - точность ориентировочная, смотрите на
            разметку ошибок
          </div>
        )}
      <div className="gameover-review-note">
        Оценка каждого хода есть в истории партии - листайте ходы стрелками
      </div>
    </div>
  );
}

let audioCtx = null;

function playMoveSound() {
  if (!audioCtx)
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  const now = audioCtx.currentTime;

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  const filter = audioCtx.createBiquadFilter();

  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1200, now);
  filter.frequency.exponentialRampToValueAtTime(300, now + 0.08);

  osc.type = "triangle";
  osc.frequency.setValueAtTime(180, now);
  osc.frequency.exponentialRampToValueAtTime(60, now + 0.06);

  gain.gain.setValueAtTime(0.5, now);
  gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);

  osc.connect(filter);
  filter.connect(gain);
  gain.connect(audioCtx.destination);

  osc.start(now);
  osc.stop(now + 0.12);
}

const ChessboardComponent = forwardRef(function ChessboardComponent(
  {
    onStateChange,
    chatCollapsed = false,
    evalMode = "material",
    onToggleEvalMode,
  },
  ref,
) {
  const gameRef = useRef(new Chess());

  /*
   * Уникальный id доски.
   *
   * Все доски react-chessboard на странице делят один dnd-менеджер
   * (библиотека держит его синглтоном на window), а внутренний
   * CustomDragLayer рисует «фигуру под курсором» только тогда, когда
   * item.id совпадает с id доски. По умолчанию id = 0 у всех досок,
   * поэтому при перетаскивании фигуры на главной доске такую же фигуру
   * рисовала и мини-доска из подсказки - фигура «раздваивалась».
   */
  const boardId = useId();

  const [fen, setFen] = useState(gameRef.current.fen());
  const [moveHistory, setMoveHistory] = useState([]);
  const [turn, setTurn] = useState("w");

  /*
   * Сколько ходов сделал сам игрок. Нужен для фиксации уровня Maia3:
   * за белых это первый ход партии, за чёрных - второй (первый делает
   * Maia3), поэтому привязываться к длине истории нельзя.
   */
  const [playerMovesPlayed, setPlayerMovesPlayed] = useState(0);

  /*
   * Итог партии для оверлея «Победа / Поражение / Ничья»:
   * null - партия идёт, иначе { outcome: "win"|"loss"|"draw", reason }.
   * Игрок играет выбранной стороной (см. playerColor), поэтому мат при
   * ходе игрока - поражение, мат при ходе соперника - победа.
   */
  const [gameResult, setGameResult] = useState(null);
  /*
   * Карточку итога можно закрыть кликом по фону, чтобы рассмотреть
   * финальную позицию. Признак хранится отдельно от gameResult: иначе
   * карточка закрывалась бы вместе с кнопкой «Разобрать партию».
   */
  const [resultCardDismissed, setResultCardDismissed] = useState(false);

  const [boardFlipped, setBoardFlipped] = useState(false);
  const [elo, setElo] = useState(1500);

  /*
   * Уровень Maia3 фиксируется после первого хода игрока: за белых это
   * первый ход партии, за чёрных - второй, после ответа Maia3. Иначе
   * можно доиграть на 800 и перед матом поставить 2600, а победа
   * засчитается как победа над 2600.
   */
  const levelLocked = playerMovesPlayed > 0;

  /*
   * Уровень, с которым реально шла партия. Запоминаем его, пока игрок
   * ещё не сделал первый ход; на завершении партии отправляем именно
   * это значение, а не текущее состояние слайдера.
   */
  const gameLevelRef = useRef(elo);

  useEffect(() => {
    if (!levelLocked) {
      gameLevelRef.current = elo;
    }
  }, [levelLocked, elo]);

  /*
   * Сторона игрока: "w" - белые, "b" - чёрные. При открытии главной
   * страницы показываем меню выбора (та же карточка, что и итог партии),
   * поэтому по умолчанию выбраны белые, а меню открыто.
   */
  const [playerColor, setPlayerColor] = useState("w");
  const [sidePickerOpen, setSidePickerOpen] = useState(true);

  const [isAiThinking, setIsAiThinking] = useState(false);
  const [maiaError, setMaiaError] = useState("");

  const [lastMove, setLastMove] = useState(null);

  const [positionSnapshots, setPositionSnapshots] = useState([]);
  const [viewIndex, setViewIndex] = useState(-1);
  const [isViewMode, setIsViewMode] = useState(false);

  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalMovesForSelected, setLegalMovesForSelected] = useState([]);

  /*
   * Клетка и фигура, с которой началось перетаскивание: { square, piece }.
   * react-chessboard прячет фигуру на исходной клетке (opacity: 0),
   * поэтому подсвечиваем её и рисуем полупрозрачного «призрака».
   */
  const [dragFrom, setDragFrom] = useState(null);

  /*
   * Предход: ход, который игрок ставит заранее, пока думает Maia3.
   * Храним максимум один предход - новый заменяет предыдущий.
   *
   * Ref нужен для отложенного выполнения: к моменту ответа Maia3
   * замыкание эффекта может содержать устаревшее состояние.
   */
  const [premove, setPremove] = useState(null);
  const premoveRef = useRef(null);

  const [boardWidth, setBoardWidth] = useState(560);
  const boardBoxRef = useRef(null);

  const [evalScore, setEvalScore] = useState(null);

  /*
   * Запрос оценки позиции у движка. Оценка приходит от лица белых,
   * поэтому ни доска, ни шкала её не пересчитывают. Ответ без оценки
   * (движок недоступен) старое значение не затирает.
   */
  const fetchEvalForPosition = useCallback((positionFen) => {
    fetchEval(positionFen)
      .then((data) => {
        if (data.evaluation) {
          setEvalScore(data.evaluation);
        }
      })
      .catch(() => {});
  }, []);

  /*
   * Разбор партии по запросу: до нажатия кнопки ничего не считаем.
   * review - ответ сервера (звёзды, точность, счётчики ходов),
   * reviewError - текст ошибки, reviewLoading - идёт разбор.
   */
  const [review, setReview] = useState(null);
  const [reviewError, setReviewError] = useState("");
  const [reviewLoading, setReviewLoading] = useState(false);

  /*
   * Две независимые стрелки:
   *
   * Stockfish -> объективно лучший ход.
   * Maia3     -> ход, который выбрал бы человек заданного Elo.
   */
  const [stockfishArrow, setStockfishArrow] = useState(null);
  const [maiaArrow, setMaiaArrow] = useState(null);

  /*
   * Данные последнего сравнения.
   * Нужны, чтобы при необходимости вывести информацию
   * рядом с доской.
   */
  const [compareData, setCompareData] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  /*
   * Номер партии для отложенного ответа Maia3: если начали новую партию
   * или загрузили другой FEN, пока движок думал, пришедший ход уже
   * неактуален и применяться не должен.
   */
  const gameGenRef = useRef(0);

  /*
   * Первый ход Maia3 при игре за чёрных запускаем один раз на партию:
   * иначе повторные рендеры дёргали бы движок заново.
   */
  const aiFirstMoveGenRef = useRef(-1);

  /*
   * Ключ доски. react-chessboard держит позицию в своём внутреннем
   * состоянии (currentPosition) и обновляет её эффектом от пропа position
   * через анимацию длиной animationDuration. При резкой внешней смене
   * позиции (новая партия после мата, загрузка FEN) это состояние может
   * остаться от прежней позиции - фигуры не обновляются. Смена key
   * пересоздаёт доску, и она монтируется сразу с нужной позицией
   * (currentPosition берётся из пропа при монтировании).
   */
  const [boardKey, setBoardKey] = useState(0);

  const game = gameRef.current;

  /*
   * ============================================================
   * ПРЕДХОДЫ
   * ============================================================
   *
   * Предход проверяется только по геометрии фигуры: занятость клеток
   * и шахи на момент постановки не учитываются. Иначе нельзя было бы
   * поставить типовой предход-перебой на клетку, где сейчас стоит своя
   * фигура, или ход пешкой, путь которой пока перекрыт.
   *
   * Окончательную легальность проверяет chess.js после ответа Maia3:
   * если ход невозможен, processPlayerMove вернёт false и предход
   * будет снят.
   */
  const setPremoveMove = useCallback((next) => {
    premoveRef.current = next;
    setPremove(next);
  }, []);

  const clearPremove = useCallback(() => {
    premoveRef.current = null;
    setPremove(null);
  }, []);

  const isPremoveGeometryLegal = useCallback(
    (from, to) => {
      if (!from || !to || from === to) return false;

      const piece = game.get(from);

      if (!piece || piece.color !== playerColor) return false;

      /*
       * Короля нельзя «съесть» ни в каком предходе: такая клетка
       * не может быть целью нормального хода.
       */
      const targetPiece = game.get(to);
      if (targetPiece?.type === "k") return false;

      const dx = to.charCodeAt(0) - from.charCodeAt(0);
      const dy = Number(to[1]) - Number(from[1]);
      const absX = Math.abs(dx);
      const absY = Math.abs(dy);

      switch (piece.type) {
        case "p": {
          const direction = piece.color === "w" ? 1 : -1;
          const startRank = piece.color === "w" ? "2" : "7";

          if (dx === 0 && dy === direction) return true;

          if (
            dx === 0 &&
            dy === 2 * direction &&
            from[1] === startRank
          ) {
            return true;
          }

          return absX === 1 && dy === direction;
        }
        case "n":
          return (
            (absX === 1 && absY === 2) ||
            (absX === 2 && absY === 1)
          );
        case "b":
          return absX === absY && absX > 0;
        case "r":
          return (dx === 0 || dy === 0) && absX + absY > 0;
        case "q":
          return (
            (dx === 0 || dy === 0 || absX === absY) &&
            absX + absY > 0
          );
        case "k": {
          if (absX <= 1 && absY <= 1 && absX + absY > 0) {
            return true;
          }

          /*
           * Рокировка: только с начальной клетки короля и только
           * при сохранённом праве на неё. Саму возможность рокировки
           * (пути, шах) окончательно проверит chess.js.
           */
          if (absY !== 0 || absX !== 2) return false;

          const rank = piece.color === "w" ? "1" : "8";
          if (from !== `e${rank}`) return false;

          const kingSide = to === `g${rank}`;
          const queenSide = to === `c${rank}`;

          if (!kingSide && !queenSide) return false;

          const castling =
            game.fen().split(" ")[2] || "-";
          const right =
            piece.color === "w"
              ? kingSide
                ? "K"
                : "Q"
              : kingSide
                ? "k"
                : "q";

          return castling.includes(right);
        }
        default:
          return false;
      }
    },
    [game, playerColor],
  );

  /*
   * Все клетки, куда геометрически может пойти выбранная фигура.
   * Используется для подсветки кликового предхода.
   */
  const getPremoveTargets = useCallback(
    (from) => {
      const targets = [];

      for (let file = 0; file < 8; file += 1) {
        for (let rank = 1; rank <= 8; rank += 1) {
          const to = "abcdefgh"[file] + rank;
          const targetPiece = game.get(to);

          /*
           * Клик по своей фигуре - это выбор другой фигуры,
           * а не взятие. Кольцо на такой клетке только путало бы;
           * перебой мышью всё равно ставится перетаскиванием.
           */
          if (targetPiece?.color === playerColor) continue;

          if (isPremoveGeometryLegal(from, to)) {
            targets.push(to);
          }
        }
      }

      return targets;
    },
    [game, playerColor, isPremoveGeometryLegal],
  );

  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        boardFlipped,
      }));
    }
  }, [boardFlipped, onStateChange]);

  /*
   * Пока открыта карточка выбора стороны, история ходов должна быть
   * недоступна: иначе поверх «За кого играем?» можно откатить партию
   * назад. Флаг уходит в общее состояние страницы, чтобы панель истории
   * погасила стрелки и клики по ходам.
   */
  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        sidePickerOpen,
      }));
    }
  }, [sidePickerOpen, onStateChange]);

  /*
   * Фиксация уровня Maia3 тоже живёт в общем состоянии страницы:
   * чипы уровней отрисовываются в левой колонке, вне доски.
   */
  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        levelLocked,
      }));
    }
  }, [levelLocked, onStateChange]);

  useEffect(() => {
    if (!boardBoxRef.current) return;

    const el = boardBoxRef.current;
    const wrapper = el.closest(".board-wrapper");
    const mainEl = document.querySelector(".main-area");

    const report = () => {
      if (typeof onStateChange === "function") {
        onStateChange((prev) => ({
          ...prev,
          boardHeight: el.offsetHeight,
        }));
      }

      /*
       * История и чат должны быть той же высоты, что и карточка доски.
       * Высота карточки зависит от размера доски и внутренних панелей,
       * поэтому передаём её в CSS через переменную, а не дублируем
       * в правилах.
       */
      if (mainEl && wrapper) {
        mainEl.style.setProperty(
          "--play-panel-height",
          `${Math.round(wrapper.getBoundingClientRect().height)}px`,
        );
      }
    };

    report();

    const ro = new ResizeObserver(report);
    ro.observe(el);
    if (wrapper) ro.observe(wrapper);

    return () => ro.disconnect();
  }, [onStateChange]);

  /*
   * Размер доски.
   */
  const updateBoardSize = useCallback(() => {
    const isMobile = window.matchMedia("(max-width: 760px)").matches;
    const isStacked = window.matchMedia("(max-width: 1450px)").matches;

    /*
     * Ширину берём у самой колонки доски, а не вычитаем чат из окна.
     * Колонка забирает всё место, оставшееся от истории, шкалы и чата,
     * и её ширина не зависит от размера доски (flex-basis: 0 в CSS).
     * Поэтому первый замер сразу совпадает с итоговой раскладкой и доска
     * не ждёт, пока чат свернут и развернут.
     */
    const boardChromeW = isMobile ? 22 : 66; // карточка (22 mobile / 38 desktop) + паддинги колонки
    const mainEl = document.querySelector(".main-area");
    const boardSectionEl = document.querySelector(".board-section");
    const boardStageEl = document.querySelector(".board-stage");
    const chatEl = document.querySelector(".chat-section");
    const containerW = mainEl?.clientWidth || window.innerWidth;

    /*
     * Пока чат едет, его текущая ширина ещё не итоговая. Берём целевую
     * из инлайнового width (его ставит ChatPanel) и переносим разницу на
     * колонку доски: так доска сразу получает финальный размер и плавно
     * переезжает вместе с чатом, без скачка в конце анимации.
     */
    let chatTargetW = chatEl ? chatEl.getBoundingClientRect().width : 0;
    if (chatEl?.classList.contains("chat-section--sliding")) {
      const inlineW = parseFloat(chatEl.style.width);
      if (inlineW > 0) {
        chatTargetW = chatEl.classList.contains("chat-section--collapsed")
          ? 48
          : inlineW;
      }
    }

    const sizingArea = isStacked ? boardSectionEl : boardStageEl;
    const sectionW = sizingArea
      ? sizingArea.clientWidth +
        (chatEl ? chatEl.getBoundingClientRect().width - chatTargetW : 0)
      : containerW;
    const evalAndGapW = isStacked ? 0 : 38 + 12;
    const availW = isStacked
      ? containerW - boardChromeW
      : sectionW - boardChromeW - evalAndGapW;

    /*
     * Высота - замером живых элементов, а не гаданием по window:
     * clientHeight .main-area уже исключает топбар, футер, отступы
     * и хром браузера. Из него вычитаем всё, что лежит в карточке
     * доски кроме самой доски: верхнюю панель, кнопки, карточку
     * сравнения, ФЕН-панель, паддинги и рамку. Именно замером, потому
     * что карточка сравнения появляется только после подсказки и
     * добавляет ~100px - иначе доска перестаёт помещаться на экране.
     *
     * Через scrollHeight, а не offsetHeight: когда содержимого больше,
     * чем места, карточка сжимается по высоте колонки и offsetHeight
     * перестаёт отражать реальную высоту содержимого.
     */
    const boardBox = boardBoxRef.current;
    const wrapper = boardBox
      ? boardBox.closest(".board-wrapper")
      : null;

    const measuredChrome =
      wrapper && boardBox
        ? wrapper.getBoundingClientRect().height -
          boardBox.getBoundingClientRect().height
        : 0;

    // Минимум - панель противника + кнопки + ФЕН-панель + рамка
    // с паддингами, когда карточки сравнения ещё нет. Реальный хром
    // карточки — около 100px, поэтому не занижаем доску лишними 50px.
    const chromeH = measuredChrome > 0 ? measuredChrome : 104;
    const pageViewport = mainEl?.parentElement;
    const availableHeight = pageViewport
      ? pageViewport.clientHeight - 22
      : (mainEl?.clientHeight || window.innerHeight) - 12;
    const availH = availableHeight - chromeH;

    const size = Math.round(Math.min(availW, availH));

    const maxBoardWidth = isMobile ? 720 : isStacked ? 860 : 1150;
    setBoardWidth(Math.max(isMobile ? 240 : 260, Math.min(size, maxBoardWidth)));
  }, []);

  useEffect(() => {
    updateBoardSize();

    window.addEventListener("resize", updateBoardSize);

    /*
     * Топбар, футер, подгрузка шрифтов и анимация чата меняют ширину
     * или высоту колонки уже после первого замера - пересчитываем по
     * изменению .main-area и самой колонки доски. Петли нет: ширина
     * колонки задана флексом и от размера доски не зависит.
     */
    const mainEl = document.querySelector(".main-area");
    const boardSectionEl = document.querySelector(".board-section");
    const boardStageEl = document.querySelector(".board-stage");
    let observer;

    if (mainEl && typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => {
        /*
         * Пока чат анимируется, колонка пересчитывается каждый кадр,
         * и замеры в этот момент врут. Окончательный размер посчитает
         * таймер ниже, когда анимация закончится.
         */
        if (document.querySelector(".chat-section--sliding")) {
          return;
        }

        updateBoardSize();
      });
      observer.observe(mainEl);
      if (boardStageEl) observer.observe(boardStageEl);
      else if (boardSectionEl) observer.observe(boardSectionEl);
    }

    // Чат едет 340 мс; после этого раскладка устоялась.
    const settleTimer = window.setTimeout(updateBoardSize, 380);

    return () => {
      window.removeEventListener("resize", updateBoardSize);
      window.clearTimeout(settleTimer);

      if (observer) {
        observer.disconnect();
      }
    };
  }, [updateBoardSize, chatCollapsed]);

  /*
   * Карточка сравнения появляется/меняет высоту после подсказки -
   * пересчитываем размер доски, чтобы карточка и ФЕН-панель влезали.
   */
  useEffect(() => {
    updateBoardSize();
  }, [compareData, updateBoardSize]);

  const updateStatus = useCallback(() => {
    const g = gameRef.current;
    setTurn(g.turn());
  }, []);

  /*
   * Автосохранение завершённой партии: как только на доске
   * мат / пат / ничья, отправляем всю партию в /api/game/finish -
   * бэкенд в фоне классифицирует ходы через Stockfish и
   * сохраняет games + game_moves.
   */
  const finishSentRef = useRef(false);

  /*
   * Показать карточку итога сразу после хода. Раньше карточка зависела
   * только от эффекта на moveHistory: если эффект по какой-то причине
   * не срабатывал, партия визуально «не заканчивалась». Теперь состояние
   * выставляется прямо в момент хода, а эффект ниже отвечает только
   * за отправку партии на сервер.
   */
  const showGameOverCard = useCallback(() => {
    const g = gameRef.current;

    if (!g || !g.isGameOver()) return;

    /*
     * Признак закрытия карточки здесь НЕ сбрасываем: иначе она снова
     * всплывала бы над доской, когда при разборе листаешь до последнего
     * хода. Новую партию и так открывает handleNewGame.
     */
    setGameResult((current) => current ?? describeGameOver(g, playerColor));
  }, [playerColor]);

  useEffect(() => {
    if (!moveHistory || moveHistory.length === 0) {
      finishSentRef.current = false;
      return;
    }

    const g = gameRef.current;

    if (!g || !g.isGameOver()) return;

    // Карточку показываем независимо от того, отправлена ли партия.
    showGameOverCard();

    if (finishSentRef.current) return;

    /*
     * Итог этой партии считаем обработанным: отправка на сервер идёт
     * один раз, чтобы повторный рендер не создал дубль партии.
     */
    finishSentRef.current = true;

    if (g.isCheckmate() && g.turn() !== playerColor && typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        maiaVictory: {
          id: `${gameGenRef.current}-${Date.now()}`,
          elo: gameLevelRef.current ?? elo,
          userId: getCurrentUserId(),
        },
      }));
    }

    let result = "*";
    let status = "game_over";

    if (g.isCheckmate()) {
      status = "checkmate";
      result = g.turn() === "w" ? "0-1" : "1-0";
    } else if (g.isStalemate()) {
      status = "stalemate";
      result = "1/2-1/2";
    } else {
      status = "draw";
      result = "1/2-1/2";
    }

    /*
     * Партию отправляем, только если ходы разворачиваются из начальной
     * позиции: после загрузки FEN история начинается с середины и
     * повтор собрать нельзя.
     */
    const uciMoves = toUciMoves(moveHistory);

    if (uciMoves.length === 0) return;

    finishGame({
      moves: uciMoves,
      elo: gameLevelRef.current ?? elo,
      engine: "maia3",
      result,
      status,
      playerColor,
    }).catch(() => {});
  }, [moveHistory, fen, elo, playerColor, onStateChange, showGameOverCard]);

  /*
   * Разбор партии по кнопке в итоговой карточке. Считает сервер:
   * Stockfish проходится по ходам игрока и возвращает точность,
   * звёзды и счётчики удачных/плохих ходов.
   */
  const handleReviewGame = useCallback(() => {
    if (reviewLoading) return;

    const uciMoves = toUciMoves(moveHistory);

    if (uciMoves.length === 0) {
      setReviewError("Партию, начатую с загруженной позиции, разобрать нельзя");
      return;
    }

    setReviewError("");
    setReviewLoading(true);

    /*
     * Разбор идёт до 40 секунд. Если за это время начали новую партию
     * или загрузили FEN, ответ уже не про текущую партию: сверяем номер
     * поколения и молча выбрасываем его.
     */
    const gen = gameGenRef.current;

    reviewGame(uciMoves, playerColor)
      .then((data) => {
        if (gen === gameGenRef.current) {
          setReview(data);
        }
      })
      .catch((error) => {
        if (gen !== gameGenRef.current) return;

        const detail = error?.response?.data?.detail;
        setReviewError(
          typeof detail === "string" && detail
            ? detail
            : "Не удалось разобрать партию, попробуйте ещё раз",
        );
      })
      .finally(() => {
        if (gen === gameGenRef.current) {
          setReviewLoading(false);
        }
      });
  }, [moveHistory, playerColor, reviewLoading]);

  /*
   * Кнопка под доской. Открывает карточку итога и сразу запускает разбор,
   * если он ещё не готов: карточку можно случайно закрыть кликом по фону,
   * и раньше вместе с ней пропадала единственная кнопка разбора.
   */
  const openReviewCard = useCallback(() => {
    setResultCardDismissed(false);
    if (!review && !reviewLoading) {
      handleReviewGame();
    }
  }, [review, reviewLoading, handleReviewGame]);

  /*
   * Пересчитываем FEN позиции по истории SAN.
   */
  const getFenForIndex = useCallback(
    (index) => {
      const START_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

      if (index === -1) return START_FEN;

      const tempGame = new Chess();

      for (let i = 0; i <= index; i++) {
        if (moveHistory[i]) {
          tempGame.move(moveHistory[i]);
        }
      }

      return tempGame.fen();
    },
    [moveHistory],
  );

  /*
   * Добавление хода в историю.
   */
  const buildMoveHistory = useCallback(
    (newMoveSan) => {
      setMoveHistory((prev) => {
        const safePrev = prev || [];

        const validHistory =
          viewIndex === -1
            ? safePrev
            : safePrev.slice(0, viewIndex + 1);

        return [...validHistory, newMoveSan];
      });
    },
    [viewIndex],
  );

  /*
   * Снимок позиции.
   */
  const takeSnapshot = useCallback(() => {
    setPositionSnapshots((prev) => {
      const safePrev = prev || [];

      const validSnapshots =
        viewIndex === -1
          ? safePrev
          : safePrev.slice(0, viewIndex + 2);

      return [...validSnapshots, game.fen()];
    });

    setViewIndex(-1);
    setIsViewMode(false);
  }, [viewIndex, game]);

  /*
   * Навигация по истории партии.
   */
  /*
   * Оценка просматриваемой позиции из разбора. Разбор приходит
   * с оценкой после каждого полухода, поэтому при листании партии
   * шкала показывает ту же цифру, что и подпись хода в истории,
   * и не ходит в движок заново.
   */
  const applyReviewEval = useCallback(
    (index) => {
      if (!review || !Array.isArray(review.plies)) return;

      if (index < 0) {
        setEvalScore({ type: "cp", value: 0 });
        return;
      }

      const ply = review.plies[index];

      if (!ply) return;

      if (ply.mate != null) {
        setEvalScore({ type: "mate", value: ply.mate });
        return;
      }

      if (ply.eval_cp != null) {
        setEvalScore({ type: "cp", value: Math.round(ply.eval_cp) });
      }
    },
    [review],
  );

  /*
   * Показ позиции после хода с индексом index (-1 - начальная).
   * Единая точка входа и для клика по ходу в истории, и для стрелок:
   * доска, подсветка последнего хода, оценка из разбора и режим
   * просмотра меняются вместе.
   */
  const showPosition = useCallback(
    (index) => {
      const g = gameRef.current;
      const history = moveHistory;

      const replay = new Chess();
      let from = null;
      let to = null;

      for (let i = 0; i <= index; i++) {
        if (!history[i]) continue;

        try {
          const played = replay.move(history[i]);

          if (played) {
            from = played.from;
            to = played.to;
          }
        } catch {
          break;
        }
      }

      const nextFen = replay.fen();

      g.load(nextFen);
      setFen(nextFen);
      setLastMove(from && to ? { from, to } : null);
      setSelectedSquare(null);
      setLegalMovesForSelected([]);
      applyReviewEval(index);

      setStockfishArrow(null);
      setMaiaArrow(null);
      setCompareData(null);

      if (index >= history.length - 1) {
        setIsViewMode(false);
        setViewIndex(-1);
      } else {
        setIsViewMode(true);
        setViewIndex(index);
      }
    },
    [moveHistory, applyReviewEval],
  );

  const onNavigate = useCallback(
    (direction) => {
      /*
       * Под карточкой выбора стороны навигация запрещена. Это вторая
       * линия защиты: панель истории тоже гасит кнопки, но onNavigate
       * можно вызвать и через boardRef.
       */
      if (sidePickerOpen) return;

      /*
       * Переход по истории меняет позицию на доске. Предход,
       * поставленный к текущей партии, к просматриваемому кадру
       * отношения не имеет.
       */
      clearPremove();
      setSelectedSquare(null);
      setLegalMovesForSelected([]);

      const history = moveHistory;

      if (history.length === 0) return;

      if (typeof direction === "number") {
        showPosition(direction);
        return;
      }

      let currentIndex = isViewMode
        ? viewIndex
        : history.length - 1;

      if (direction === "first") {
        currentIndex = -1;
      } else if (direction === "prev") {
        currentIndex = Math.max(-1, currentIndex - 1);
      } else if (direction === "next") {
        currentIndex = Math.min(
          history.length - 1,
          currentIndex + 1,
        );
      } else if (direction === "last") {
        currentIndex = history.length - 1;
      }

      showPosition(currentIndex);
    },
    [
      moveHistory,
      isViewMode,
      viewIndex,
      sidePickerOpen,
      clearPremove,
      showPosition,
    ],
  );

  /*
   * Листание партии стрелками, как на Lichess. Включается только после
   * завершения партии (иначе перехватывало бы прокрутку во время игры)
   * и не трогает поля ввода - прежде всего поле чата.
   */
  useEffect(() => {
    if (!gameResult) return undefined;

    const onKeyDown = (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;

      const target = event.target;

      if (
        target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
      ) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        onNavigate("prev");
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        onNavigate("next");
      } else if (event.key === "Home") {
        event.preventDefault();
        onNavigate("first");
      } else if (event.key === "End") {
        event.preventDefault();
        onNavigate("last");
      }
    };

    window.addEventListener("keydown", onKeyDown);

    return () => window.removeEventListener("keydown", onKeyDown);
  }, [gameResult, onNavigate]);

  /*
   * Новая игра.
   */
  const handleNewGame = useCallback(() => {
    clearPremove();
    game.reset();

    const startFen = game.fen();

    /*
     * Партия начата заново: отложенный ответ движка больше не применять,
     * а доску пересоздать - иначе react-chessboard может оставить
     * на ней прежнюю позицию.
     */
    gameGenRef.current += 1;
    setBoardKey((key) => key + 1);

    setFen(startFen);
    setMoveHistory([]);
    setPlayerMovesPlayed(0);
    setGameResult(null);
    setResultCardDismissed(false);
    setPositionSnapshots([startFen]);

    setViewIndex(-1);
    setIsViewMode(false);

    setLastMove(null);
    setSelectedSquare(null);
    setLegalMovesForSelected([]);

    setIsAiThinking(false);
    setMaiaError("");
    setIsComparing(false);

    setEvalScore(null);

    setReview(null);
    setReviewError("");
    setReviewLoading(false);

    setStockfishArrow(null);
    setMaiaArrow(null);
    setCompareData(null);

    fetchEval(startFen)
      .then((data) => {
        if (data.evaluation) {
          setEvalScore(data.evaluation);
        }
      })
      .catch(() => {});

    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        fen: startFen,
        moveHistory: [],
        positionSnapshots: [startFen],
        viewIndex: -1,
        turn: "w",
        evalScore: null,
        isComparing: false,
      }));
    }
  }, [game, onStateChange, clearPremove]);

  /*
   * Выбор стороны в меню при входе на страницу: доска разворачивается
   * так, чтобы игрок был внизу, и партия начинается заново. Если выбраны
   * чёрные, первый ход сделает Maia3 (эффект ниже - по начальной позиции
   * с пустой историей).
   */
  const handleChooseSide = useCallback(
    (color) => {
      setPlayerColor(color);
      setBoardFlipped(color === "b");
      setSidePickerOpen(false);
      handleNewGame();
    },
    [handleNewGame],
  );

  /*
   * «Новая игра» сначала спрашивает сторону: меню открывается и при
   * входе на страницу, и по кнопке. Карточку итога прячем, чтобы два
   * оверлея не накладывались друг на друга. Если игрок успел отмотать
   * партию назад, возвращаем доску к последнему ходу: под меню должна
   * быть актуальная позиция, а не случайный просмотренный кадр.
   */
  const openSidePicker = useCallback(() => {
    onNavigate("last");
    setGameResult(null);
    setResultCardDismissed(false);
    setSidePickerOpen(true);
  }, [onNavigate]);

  /*
   * Загрузка FEN.
   */
  const handleLoadFen = useCallback(
    (newFen) => {
      clearPremove();

      try {
        game.load(newFen);

        const validFen = game.fen();

        /*
         * Позиция загружена извне: отложенный ответ движка отменяем,
         * доску пересоздаём (см. boardKey).
         */
        gameGenRef.current += 1;
        setBoardKey((key) => key + 1);

        setFen(validFen);
        setMoveHistory([]);
        setPlayerMovesPlayed(0);
        setGameResult(null);
        setPositionSnapshots([validFen]);

        setViewIndex(-1);
        setIsViewMode(false);

        setLastMove(null);
        setSelectedSquare(null);
        setLegalMovesForSelected([]);

        setIsAiThinking(false);
        setMaiaError("");
        setIsComparing(false);

        setEvalScore(null);

        setReview(null);
        setReviewError("");
        setReviewLoading(false);

        setStockfishArrow(null);
        setMaiaArrow(null);
        setCompareData(null);

        fetchEval(validFen)
          .then((data) => {
            if (data.evaluation) {
              setEvalScore(data.evaluation);
            }
          })
          .catch(() => {});

        if (typeof onStateChange === "function") {
          onStateChange((prev) => ({
            ...prev,
            fen: validFen,
            moveHistory: [],
            positionSnapshots: [validFen],
            viewIndex: -1,
            turn: game.turn(),
          }));
        }
        return true;
      } catch (error) {
        // Сообщение покажет меню «Позиция», откуда загружают FEN.
        console.error("Неверный FEN:", error);
        return false;
      }
    },
    [game, onStateChange, clearPremove],
  );

  /*
   * ============================================================
   * СРАВНЕНИЕ STOCKFISH И MAIA3
   * ============================================================
   *
   * Именно эта функция используется кнопкой "Лучший ход".
   *
   * Backend:
   *
   * POST /api/compare-moves
   *
   * Возвращает:
   *
   * stockfish:
   *   move
   *   from
   *   to
   *   san
   *   evaluation
   *
   * maia3:
   *   move
   *   from
   *   to
   *   san
   *   elo
   *
   * После получения ответа рисуем ДВЕ стрелки.
   */
const fetchBestMove = useCallback(
  async (positionFen = game.fen()) => {
    if (isComparing) return;

    if (game.isGameOver()) {
      return;
    }

    setIsComparing(true);
    onStateChange((prev) => ({ ...prev, isComparing: true }));

    try {
      /*
       * ============================================================
       * ИСТОРИЯ ПАРТИИ В UCI
       * ============================================================
       *
       * moveHistory хранит SAN, например:
       *
       * e4
       * e5
       * Nf3
       *
       * Backend compare-moves ожидает историю в UCI:
       *
       * e2e4
       * e7e5
       * g1f3
       *
       * Историю берём из живого объекта партии: moveHistory в состоянии
       * React может отставать на ход, и тогда позиция на сервере не
       * сойдётся с присланным FEN.
       */
      const uciMoves = toUciMoves(game.history());

      /*
       * ============================================================
       * STOCKFISH + MAIA3
       * ============================================================
       *
       * Получаем:
       *
       * Stockfish -> объективно лучший ход.
       * Maia3     -> человекоподобный ход для выбранного Elo.
       */
      const data = await fetchCompareMoves(
        positionFen,
        elo,
        uciMoves,
      );

      if (!data || data.error) {
        console.warn(
          "[Compare] Ошибка:",
          data?.error,
        );

        setStockfishArrow(null);
        setMaiaArrow(null);
        setCompareData(null);

        return;
      }

      /*
       * ============================================================
       * СТРЕЛКА STOCKFISH
       * ============================================================
       *
       * Зелёная стрелка показывает объективно лучший ход.
       */
      if (
        data.stockfish &&
        data.stockfish.from &&
        data.stockfish.to
      ) {
        setStockfishArrow([
          data.stockfish.from,
          data.stockfish.to,
          "rgba(0, 150, 50, 0.85)",
        ]);
      } else {
        setStockfishArrow(null);
      }

      /*
       * ============================================================
       * СТРЕЛКА MAIA3
       * ============================================================
       *
       * Оранжевая стрелка показывает ход, который скорее
       * выбрал бы человек заданного Elo.
       */
      if (
        data.maia3 &&
        data.maia3.from &&
        data.maia3.to
      ) {
        setMaiaArrow([
          data.maia3.from,
          data.maia3.to,
          "rgba(220, 120, 20, 0.9)",
        ]);
      } else {
        setMaiaArrow(null);
      }

      /*
       * Сохраняем результат сравнения.
       *
       * Он используется в блоке под доской:
       *
       * Stockfish: ...
       * Maia3: ...
       */
      setCompareData(data);

      /*
       * ============================================================
       * ОБЪЯСНЕНИЕ ЛУЧШЕГО ХОДА STOCKFISH
       * ============================================================
       *
       * ВАЖНО:
       *
       * этот код находится ВНУТРИ fetchBestMove().
       *
       * Поэтому /api/explain-move вызывается ТОЛЬКО тогда,
       * когда пользователь нажимает кнопку "Лучший ход".
       *
       * При обычных ходах игрока GigaChess не вызывается.
       * При ходе Maia3 GigaChess тоже не вызывается.
       */

      if (
        data.stockfish &&
        data.stockfish.from &&
        data.stockfish.to
      ) {
        /*
         * Определяем UCI лучшего хода.
         *
         * Например:
         *
         * e2e4
         * g1f3
         * e7e8q
         */
        let stockfishUci = null;

        /*
         * Если compare-moves уже вернул готовый UCI,
         * используем его.
         */
        if (
          typeof data.stockfish.uci === "string" &&
          data.stockfish.uci.length >= 4
        ) {
          stockfishUci = data.stockfish.uci;
        }

        /*
         * В некоторых версиях backend UCI может находиться
         * в поле move.
         */
        if (
          !stockfishUci &&
          typeof data.stockfish.move === "string" &&
          data.stockfish.move.length >= 4
        ) {
          stockfishUci = data.stockfish.move;
        }

        /*
         * Если готового UCI нет - собираем его из from/to.
         */
        if (!stockfishUci) {
          stockfishUci =
            data.stockfish.from +
            data.stockfish.to +
            (data.stockfish.promotion || "");
        }

        /*
         * Ошибка объяснения НЕ должна ломать основную подсказку.
         *
         * Поэтому здесь отдельный try/catch.
         *
         * Даже если GigaChess недоступен:
         *
         * - стрелка Stockfish останется;
         * - стрелка Maia3 останется;
         * - compareData останется.
         */
        try {
          console.log(
            "[ExplainMove] Запрашиваем объяснение:",
            {
              fen: positionFen,
              move: stockfishUci,
              elo,
            },
          );

          const explanationData =
            await fetchExplainMove(
              positionFen,
              stockfishUci,
              elo,
            );

          console.log(
            "[ExplainMove] Ответ:",
            explanationData,
          );

          if (
            explanationData?.ok &&
            explanationData?.explanation
          ) {
            /*
             * SAN лучше брать из нового explain-move,
             * поскольку именно этот ход фактически объяснялся.
             */
            const explainedSan =
              explanationData.explained_move?.san ||
              data.stockfish.san ||
              stockfishUci;

            /*
             * Формируем сообщение для ChatPanel.
             *
             * Твой ChatPanel уже умеет отображать Markdown:
             *
             * ### заголовки
             * **жирный текст**
             */
            const message = [
              "### Подсказка Stockfish",
              "",
              `**Лучший ход:** ${explainedSan}`,
              "",
              explanationData.explanation,
            ].join("\n");

            /*
             * Передаём сообщение родительскому компоненту.
             *
             * Уникальный id нужен для того, чтобы пользователь
             * мог нажать "Лучший ход" повторно даже в той же
             * позиции - ChatPanel воспримет это как новое
             * сообщение.
             */
            if (
              typeof onStateChange === "function"
            ) {
              onStateChange((prev) => ({
                ...prev,

                hintMessage: {
                  id:
                    Date.now().toString() +
                    "_" +
                    Math.random()
                      .toString(36)
                      .slice(2),

                  text: message,

                  /*
                   * Позицию, в которой был запрошен лучший ход,
                   * ChatPanel рисует мини-доской прямо в сообщении.
                   */
                  fen: positionFen,

                  /*
                   * Можно сохранить исходные данные.
                   *
                   * Сейчас ChatPanel они не использует,
                   * но в будущем пригодятся для красивой
                   * карточки анализа.
                   */
                  explanation:
                    explanationData.explanation,

                  san: explainedSan,

                  uci:
                    explanationData.explained_move
                      ?.uci || stockfishUci,

                  from:
                    explanationData.explained_move
                      ?.from ||
                    data.stockfish.from,

                  to:
                    explanationData.explained_move
                      ?.to ||
                    data.stockfish.to,

                  /*
                   * Обе стрелки (Stockfish + Maia3), чтобы мини-доска
                   * в чате показывала их так же, как главная доска,
                   * когда ходы движков различаются.
                   */
                  stockfishArrow:
                    data.stockfish &&
                    data.stockfish.from &&
                    data.stockfish.to
                      ? {
                          from: data.stockfish.from,
                          to: data.stockfish.to,
                        }
                      : null,

                  maiaArrow:
                    data.maia3 &&
                    data.maia3.from &&
                    data.maia3.to
                      ? {
                          from: data.maia3.from,
                          to: data.maia3.to,
                        }
                      : null,

                  source:
                    explanationData.explanation_source ||
                    null,
                },
              }));
            }
          } else {
            console.warn(
              "[ExplainMove] Backend не вернул объяснение:",
              explanationData?.error ||
                explanationData,
            );
          }
        } catch (explanationError) {
          console.error(
            "[ExplainMove] Не удалось получить объяснение:",
            explanationError,
          );
        }
      }
    } catch (error) {
      /*
       * Ошибка самого сравнения Stockfish / Maia3.
       */
      console.error(
        "[Compare] Ошибка сравнения Stockfish/Maia3:",
        error,
      );

      setStockfishArrow(null);
      setMaiaArrow(null);
      setCompareData(null);
    } finally {
      setIsComparing(false);
      onStateChange((prev) => ({ ...prev, isComparing: false }));
    }
  },
  [
    game,
    elo,
    isComparing,
    onStateChange,
  ],
);

  /*
   * Ход Maia3.
   *
   * Здесь Maia3 выступает именно как соперник.
   */
  const makeAiMove = useCallback(async () => {
    if (game.isGameOver() || game.isDraw()) return;

    /*
     * Номер партии на момент вызова: после ответа движка сверяем -
     * если партия успела смениться, ход применять нельзя.
     */
    const gen = gameGenRef.current;

    setIsAiThinking(true);
    setMaiaError("");

    /*
     * Пока Maia делает ход, старые стрелки сравнения убираем.
     */
    setStockfishArrow(null);
    setMaiaArrow(null);
    setCompareData(null);

    try {
      /*
       * Историю берём из живого объекта партии, а не из состояния React.
       *
       * moveHistory обновляется через setState, и в момент вызова (через
       * 500 мс после хода игрока) замыкание держит значение с прошлого
       * рендера - без только что сыгранного хода. Сервер получал историю,
       * которая не сходилась с присланным FEN, писал "History FEN
       * mismatch" и работал только с текущей позицией. Maia3 запущена
       * с --use-uci-history и без истории играет вслепую: не видит, что
       * было в партии, и её уровень резко падает.
       */
      const uciMoves = toUciMoves(game.history());

      const data = await fetchMaiaMove(
        game.fen(),
        elo,
        uciMoves,
      );

      /*
       * Пока движок думал, могли начать новую партию или загрузить
       * другой FEN - тогда этот ход уже не для текущей позиции.
       */
      if (gen !== gameGenRef.current) return;

      if (data.error) {
        // Хода соперника не будет - ждать предходу нечего.
        clearPremove();
        setMaiaError(data.error);
        return;
      }

      const move = game.move({
        from: data.from,
        to: data.to,
        promotion: "q",
      });

      if (!move) {
        clearPremove();
        setMaiaError("Maia3 вернула нелегальный ход. Позиция не изменена.");
        return;
      }

      if (move) {
        const currentFen = game.fen();

        setFen(currentFen);

        setLastMove({
          from: data.from,
          to: data.to,
        });

        buildMoveHistory(move.san);
        takeSnapshot();
        updateStatus();
        showGameOverCard();

        /*
         * Позиция сменилась - подсветка выбора и псевдолегальные
         * клетки от старой позиции больше не актуальны.
         */
        setSelectedSquare(null);
        setLegalMovesForSelected([]);

        playMoveSound();

        /*
         * Ход AI меняет позицию, поэтому оценку движка запрашиваем
         * заново: иначе шкала отставала бы на полхода (показывала
         * позицию сразу после хода игрока).
         */
        fetchEvalForPosition(currentFen);

        if (typeof onStateChange === "function") {
          onStateChange((prev) => ({
            ...prev,
            fen: currentFen,
            moveHistory: [
              ...(prev.moveHistory || []),
              move.san,
            ],
            positionSnapshots: [
              ...(prev.positionSnapshots || []),
              currentFen,
            ],
            viewIndex: -1,
            turn: game.turn(),
          }));
        }
      }
    } catch (err) {
      console.error("Maia3 ошибка:", err);
      const detail = err?.response?.data?.detail;
      if (gen === gameGenRef.current) {
        clearPremove();
        if (typeof detail === "string") {
          setMaiaError(detail);
        } else if (err?.code === "ECONNABORTED") {
          setMaiaError(
            "Maia3 не ответила вовремя. Движок перезапущен - " +
              "нажмите «Повторить», ход будет сделан.",
          );
        } else {
          setMaiaError("Maia3 сейчас недоступна. Ход не был выполнен.");
        }
      }
    } finally {
      if (gen === gameGenRef.current) {
        setIsAiThinking(false);
      }
    }
  }, [
    game,
    elo,
    buildMoveHistory,
    takeSnapshot,
    updateStatus,
    onStateChange,
    clearPremove,
    fetchEvalForPosition,
    showGameOverCard,
  ]);

  /*
   * Первичная оценка позиции.
   */
  useEffect(() => {
    fetchEval(game.fen())
      .then((data) => {
        if (data.evaluation) {
          setEvalScore(data.evaluation);
        }
      })
      .catch(() => {});
  }, []);

  /*
   * Если шкалу переключили на движок, а оценки ещё нет (первый запрос
   * не удался или прошёл до переключения), запрашиваем её для текущей
   * позиции. Без этого капсула осталась бы материальной.
   */
  useEffect(() => {
    if (evalMode !== "engine" || evalScore) return undefined;

    let cancelled = false;

    fetchEval(game.fen())
      .then((data) => {
        if (!cancelled && data.evaluation) {
          setEvalScore(data.evaluation);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [evalMode, evalScore, game]);

  /*
   * Передаём evaluation наверх.
   */
  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        evalScore,
      }));
    }
  }, [evalScore, onStateChange]);

  /*
   * Передаём navigation state наверх.
   */
  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        viewIndex,
        isViewMode,
      }));
    }
  }, [viewIndex, isViewMode, onStateChange]);

  /*
   * Разбор нужен не только итоговой карточке: по нему история ходов
   * подписывает каждый ход оценкой, поэтому отдаём его наверх.
   */
  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        gameReview: review,
      }));
    }
  }, [review, onStateChange]);

  /*
   * ============================================================
   * ХОД ИГРОКА
   * ============================================================
   */
  const processPlayerMove = useCallback(
    (sourceSquare, targetSquare) => {
      let move;
      try {
        move = game.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: "q",
        });
      } catch {
        // Illegal drops are normal input; chess.js throws instead of returning null.
        return false;
      }

      if (move === null) {
        return false;
      }

      /*
       * Любой реальный ход отменяет ожидающий предход: либо он уже
       * выполняется (и эффект снял его заранее), либо игрок решил
       * сходить сам.
       */
      clearPremove();

      /*
       * Игрок сделал ход: с этого момента уровень Maia3 зафиксирован
       * до конца партии (за чёрных - начиная со второго хода партии).
       */
      setPlayerMovesPlayed((count) => count + 1);

      const currentFen = game.fen();

      /*
       * После реального хода игрока старые стрелки
       * больше не актуальны.
       */
      setStockfishArrow(null);
      setMaiaArrow(null);
      setCompareData(null);

      setFen(currentFen);

      setLastMove({
        from: sourceSquare,
        to: targetSquare,
      });

      setSelectedSquare(null);
      setLegalMovesForSelected([]);

      buildMoveHistory(move.san);
      takeSnapshot();

      updateStatus();
      showGameOverCard();
      playMoveSound();

      fetchEvalForPosition(currentFen);

      if (typeof onStateChange === "function") {
        onStateChange((prev) => {
          const safeHistory =
            prev.moveHistory || [];

          const safeSnapshots =
            prev.positionSnapshots || [
              prev.fen || currentFen,
            ];

          const currentMoveHistory =
            prev.viewIndex === -1
              ? safeHistory
              : safeHistory.slice(
                  0,
                  prev.viewIndex + 1,
                );

          const currentSnapshots =
            prev.viewIndex === -1
              ? safeSnapshots
              : safeSnapshots.slice(
                  0,
                  prev.viewIndex + 2,
                );

          return {
            ...prev,
            fen: currentFen,
            moveHistory: [
              ...currentMoveHistory,
              move.san,
            ],
            positionSnapshots: [
              ...currentSnapshots,
              currentFen,
            ],
            viewIndex: -1,
            turn: game.turn(),
          };
        });
      }

      saveMoveToDataset(
        currentFen,
        sourceSquare + targetSquare,
        getCurrentUserId(),
        gameId,
      ).catch(() => {});

      /*
       * После хода игрока Maia3 отвечает. Если за это время партию
       * начали заново - ответ не запрашиваем.
       */
      if (
        !game.isGameOver() &&
        !game.isDraw()
      ) {
        const gen = gameGenRef.current;
        setTimeout(() => {
          if (gen === gameGenRef.current) makeAiMove();
        }, 500);
      }

      return true;
    },
    [
      game,
      buildMoveHistory,
      takeSnapshot,
      updateStatus,
      fetchEvalForPosition,
      makeAiMove,
      onStateChange,
      clearPremove,
      showGameOverCard,
    ],
  );

  /*
   * Автоматическое выполнение предхода.
   *
   * Эффект срабатывает, когда ответ Maia3 уже применён к партии
   * (chess.js переключил turn на игрока). Небольшая задержка нужна,
   * чтобы доска успела проиграть анимацию хода соперника.
   *
   * Если ход окажется нелегальным (шах, связка, фигура съедена),
   * processPlayerMove вернёт false - предход просто исчезнет.
   */
  useEffect(() => {
    const pending = premoveRef.current;

    if (!pending) return undefined;
    if (isViewMode || sidePickerOpen || isAiThinking) return undefined;
    if (game.turn() !== playerColor) return undefined;

    if (game.isGameOver() || game.isDraw()) {
      clearPremove();
      return undefined;
    }

    const timer = setTimeout(() => {
      /*
       * За время задержки предход могли отменить или заменить,
       * а игрок - успеть сходить вручную.
       */
      if (premoveRef.current !== pending) return;
      if (game.turn() !== playerColor || isAiThinking) return;
      if (game.isGameOver() || game.isDraw()) {
        clearPremove();
        return;
      }

      clearPremove();
      processPlayerMove(pending.from, pending.to);
    }, PREMOVE_EXECUTION_DELAY);

    return () => clearTimeout(timer);
  }, [
    fen,
    turn,
    premove,
    isAiThinking,
    isViewMode,
    sidePickerOpen,
    playerColor,
    game,
    processPlayerMove,
    clearPremove,
  ]);

  /*
   * Играем чёрными: партия начинается с хода Maia3 за белых.
   *
   * Условия - начальная позиция, пустая история и закрытое меню выбора
   * стороны. Запускаем один раз на партию (aiFirstMoveGenRef), чтобы
   * неудачный запрос не превратился в бесконечный цикл повторов.
   */
  useEffect(() => {
    if (sidePickerOpen || playerColor !== "b") return;
    if (fen !== START_POSITION || moveHistory.length > 0) return;
    if (isAiThinking || gameResult) return;
    if (aiFirstMoveGenRef.current === gameGenRef.current) return;

    aiFirstMoveGenRef.current = gameGenRef.current;

    const gen = gameGenRef.current;
    const timer = setTimeout(() => {
      if (gen === gameGenRef.current) makeAiMove();
    }, 600);

    return () => clearTimeout(timer);
  }, [
    sidePickerOpen,
    playerColor,
    fen,
    moveHistory.length,
    isAiThinking,
    gameResult,
    makeAiMove,
  ]);

  /*
   * Клик по клеткам во время хода Maia3 ставит предход.
   * Цели подсвечиваются по геометрии фигуры, а не по chess.js:
   * так работают типовые предходы-перебои и ходы через временно
   * перекрытые клетки.
   */
  const handlePremoveSquareClick = useCallback(
    (square) => {
      if (game.isGameOver() || game.isDraw()) return;

      /*
       * Клик по уже подсвеченной цели снимает предход - удобная
       * отмена и на мобильных, где нет правого клика.
       */
      if (premove && square === premove.to) {
        clearPremove();
        return;
      }

      const piece = game.get(square);

      if (selectedSquare) {
        if (square === selectedSquare) {
          setSelectedSquare(null);
          setLegalMovesForSelected([]);
          return;
        }

        if (piece && piece.color === playerColor) {
          setSelectedSquare(square);
          setLegalMovesForSelected(getPremoveTargets(square));
          return;
        }

        if (isPremoveGeometryLegal(selectedSquare, square)) {
          const movingPiece = game.get(selectedSquare);

          setPremoveMove({
            from: selectedSquare,
            to: square,
            color: movingPiece.color,
            type: movingPiece.type,
          });
          setSelectedSquare(null);
          setLegalMovesForSelected([]);
          return;
        }

        setSelectedSquare(null);
        setLegalMovesForSelected([]);
        return;
      }

      if (piece && piece.color === playerColor) {
        clearPremove();
        setSelectedSquare(square);
        setLegalMovesForSelected(getPremoveTargets(square));
        return;
      }

      /*
       * Клик мимо своих фигур отменяет незавершённый выбор
       * и уже поставленный предход.
       */
      clearPremove();
      setSelectedSquare(null);
      setLegalMovesForSelected([]);
    },
    [
      game,
      premove,
      selectedSquare,
      playerColor,
      isPremoveGeometryLegal,
      getPremoveTargets,
      setPremoveMove,
      clearPremove,
    ],
  );

  /*
   * Drag & Drop.
   */
  const onDrop = useCallback(
    (sourceSquare, targetSquare) => {
      if (isViewMode || sidePickerOpen) return false;
      if (game.isGameOver() || game.isDraw()) return false;

      /*
       * Свой ход - обычная обработка. Предход на этот момент
       * уже не нужен.
       */
      if (game.turn() === playerColor) {
        clearPremove();
        return processPlayerMove(sourceSquare, targetSquare);
      }

      /*
       * Ход Maia3: запоминаем предход. Возвращаем false, чтобы
       * react-chessboard не двигал фигуру - позицию изменит
       * processPlayerMove после ответа соперника.
       */
      if (!isPremoveGeometryLegal(sourceSquare, targetSquare)) {
        return false;
      }

      const movingPiece = game.get(sourceSquare);

      setPremoveMove({
        from: sourceSquare,
        to: targetSquare,
        color: movingPiece.color,
        type: movingPiece.type,
      });
      setSelectedSquare(null);
      setLegalMovesForSelected([]);

      return false;
    },
    [
      game,
      isViewMode,
      sidePickerOpen,
      playerColor,
      isPremoveGeometryLegal,
      processPlayerMove,
      setPremoveMove,
      clearPremove,
    ],
  );

  /*
   * Клик по клеткам.
   */
  const onSquareClick = useCallback(
    (square) => {
      if (isViewMode || sidePickerOpen) return;
      if (game.isGameOver() || game.isDraw()) return;

      /*
       * Пока ходит Maia3, клики ставят предход.
       */
      if (game.turn() !== playerColor) {
        handlePremoveSquareClick(square);
        return;
      }

      /*
       * Начало ручного хода отменяет ожидающий предход: он мог
       * ждать короткую задержку перед автоматическим выполнением.
       */
      clearPremove();

      const piece = game.get(square);

      if (selectedSquare) {
        /*
         * Выбрана своя фигура и пользователь
         * выбрал другую свою фигуру.
         */
        if (
          piece &&
          piece.color === game.turn()
        ) {
          if (square === selectedSquare) {
            setSelectedSquare(null);
            setLegalMovesForSelected([]);
            return;
          }

          setSelectedSquare(square);

          const moves = game.moves({
            square,
            verbose: true,
          });

          setLegalMovesForSelected(
            moves.map((m) => m.to),
          );

          return;
        }

        /*
         * Попытка сделать ход.
         */
        const success = processPlayerMove(
          selectedSquare,
          square,
        );

        if (success) {
          return;
        }

        setSelectedSquare(null);
        setLegalMovesForSelected([]);

        return;
      }

      /*
       * Выбор своей фигуры.
       */
      if (
        piece &&
        piece.color === game.turn()
      ) {
        setSelectedSquare(square);

        const moves = game.moves({
          square,
          verbose: true,
        });

        setLegalMovesForSelected(
          moves.map((m) => m.to),
        );
      }
    },
    [
      game,
      selectedSquare,
      isViewMode,
      playerColor,
      sidePickerOpen,
      handlePremoveSquareClick,
      processPlayerMove,
      clearPremove,
    ],
  );

  /*
   * Если игрок начинает тянуть фигуру на своём ходу, ожидающий
   * предход уже не должен выполниться сам.
   */
  const handlePieceDragBegin = useCallback(
    (piece, sourceSquare) => {
      if (game.turn() === playerColor) {
        clearPremove();
      }

      setDragFrom({
        square: sourceSquare,
        piece,
      });
    },
    [game, playerColor, clearPremove],
  );

  const handlePieceDragEnd = useCallback(() => {
    setDragFrom(null);
  }, []);

  const handleSquareRightClick = useCallback(() => {
    clearPremove();
  }, [clearPremove]);

  const isDraggablePiece = useCallback(
    ({ piece }) =>
      !isViewMode &&
      !sidePickerOpen &&
      piece?.[0] === playerColor,
    [isViewMode, sidePickerOpen, playerColor],
  );

  /*
   * Стандартная проверка превращения из react-chessboard, но во время
   * хода Maia3 диалог не нужен: это предход, а он всё равно
   * превращается в ферзя в processPlayerMove.
   */
  const handlePromotionCheck = useCallback(
    (sourceSquare, targetSquare, piece) => {
      if (game.turn() !== playerColor) return false;

      return (
        ((piece === "wP" &&
          sourceSquare[1] === "7" &&
          targetSquare[1] === "8") ||
          (piece === "bP" &&
            sourceSquare[1] === "2" &&
            targetSquare[1] === "1")) &&
        Math.abs(
          sourceSquare.charCodeAt(0) -
            targetSquare.charCodeAt(0),
        ) <= 1
      );
    },
    [game, playerColor],
  );

  const handlePromotionPieceSelect =
    useCallback(() => {
      return true;
    }, []);

  const setMaiaLevel = useCallback(
    (level) => {
      /*
       * Пока партия идёт, уровень зафиксирован. Проверка стоит здесь,
       * а не только на элементах управления: так её нельзя обойти
       * вызовом setMaiaLevel через boardRef.
       */
      if (levelLocked) return;

      const nextElo = Math.max(
        800,
        Math.min(2600, Math.round(Number(level) / 100) * 100),
      );
      setElo(nextElo);
      setStockfishArrow(null);
      setMaiaArrow(null);
      setCompareData(null);
      if (typeof onStateChange === "function") {
        onStateChange((prev) => ({ ...prev, maiaRating: nextElo }));
      }
    },
    [onStateChange, levelLocked],
  );

  /*
   * Публичные методы компонента.
   */
  useImperativeHandle(ref, () => ({
    onNavigate,
    handleNewGame,
    setMaiaLevel,
    getFen: () => gameRef.current.fen(),
    // Меню «Позиция»: загрузка FEN и экспорт партии в PGN.
    loadFen: handleLoadFen,
    getPgn: () => {
      try {
        return gameRef.current.pgn();
      } catch {
        return "";
      }
    },
    // Кнопка «Подсказка» живёт в ChatPanel - дёргает анализ через ref.
    requestBestMove: () => fetchBestMove(),
  }));

  /*
   * ============================================================
   * MATERIAL ADVANTAGE
   * ============================================================
   */
  const getMaterialAdvantage = useCallback(() => {
    const board = game.board();

    const pieceValues = {
      p: 1,
      n: 3,
      b: 3,
      r: 5,
      q: 9,
    };

    /*
     * Полный комплект на начало партии: разница с тем, что стоит на
     * доске, - это съеденные фигуры.
     */
    const fullSet = {
      p: 8,
      n: 2,
      b: 2,
      r: 2,
      q: 1,
    };

    let white = 0;
    let black = 0;

    const whitePieces = {};
    const blackPieces = {};

    board.forEach((row) => {
      row.forEach((square) => {
        if (!square) return;

        const value =
          pieceValues[square.type] || 0;

        if (square.color === "w") {
          white += value;

          whitePieces[square.type] =
            (whitePieces[square.type] || 0) + 1;
        } else {
          black += value;

          blackPieces[square.type] =
            (blackPieces[square.type] || 0) + 1;
        }
      });
    });

    const diff = white - black;

    /*
     * Перевес показываем по реально съеденным фигурам, а не раскладкой
     * численной разницы на «удобные» фигуры: если с доски ушёл ферзь,
     * в перевесе будет ферзь, а не слон с ладьёй.
     *
     * Считаем перевес по каждому типу отдельно для обеих сторон:
     * даже отстающий по общему материалу игрок может иметь лишнюю
     * фигуру конкретного типа (например, ладью). Взаимные взятия
     * одного типа сокращаем.
     */
    const surplus = {
      w: {},
      b: {},
    };

    MATERIAL_PIECE_ORDER.forEach((type) => {
      /*
       * Ограничение полным комплектом - страховка от превращения
       * пешки: «лишняя» фигура превращения не должна считаться
       * съеденной у соперника.
       */
      const whiteCount = Math.min(
        whitePieces[type] || 0,
        fullSet[type],
      );

      const blackCount = Math.min(
        blackPieces[type] || 0,
        fullSet[type],
      );

      if (whiteCount > blackCount) {
        surplus.w[type] = whiteCount - blackCount;
      } else if (blackCount > whiteCount) {
        surplus.b[type] = blackCount - whiteCount;
      }
    });

    return {
      diff,
      surplus,
    };
  }, [game]);

  const materialAdv =
    getMaterialAdvantage();

  /*
   * Данные для строк игроков: строка показывает лишние фигуры той
   * стороны, которая в ней сидит, а цвет иконок - цвет соперника,
   * ведь это его фигуры были взяты.
   */
  const materialByColor = {
    w: {
      diff: materialAdv.diff,
      pieces: materialAdv.surplus.w,
      pieceColor: "black",
    },
    b: {
      diff: -materialAdv.diff,
      pieces: materialAdv.surplus.b,
      pieceColor: "white",
    },
  };

  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        materialDiff: materialAdv.diff,
      }));
    }
  }, [
    materialAdv.diff,
    onStateChange,
  ]);

  /*
   * Материальный перевес: иконки конкретных фигур, которые сторона
   * взяла сверх соперника, и число - только у лидера по общему
   * материалу. Цвет иконок - цвет соперника: это его фигуры съедены.
   */
  const MaterialDisplay = ({
    diff = 0,
    pieces = {},
    pieceColor = "black",
  }) => {
    const captured = [];

    MATERIAL_PIECE_ORDER.forEach((type) => {
      const count = pieces[type] || 0;

      for (let i = 0; i < count; i += 1) {
        captured.push(type);
      }
    });

    if (captured.length === 0 && diff <= 0) return null;

    return (
      <div
        className="material-display"
        title={
          diff > 0
            ? `Материальный перевес: +${diff}`
            : "Взятые фигуры"
        }
      >
        <span className="material-pieces">
          {captured.map((type, index) => (
            <img
              key={`${type}-${index}`}
              className="material-piece"
              src={`/pieces/${pieceColor}_${PIECE_FILE_NAMES[type]}.svg`}
              alt=""
              aria-hidden="true"
            />
          ))}
        </span>

        {diff > 0 && (
          <span className="material-diff">
            +{diff}
          </span>
        )}
      </div>
    );
  };

  /*
   * ============================================================
   * СТИЛИ ДОСКИ
   * ============================================================
   */
  const cornerStyle = (radius) => ({
    borderRadius: radius,
  });

  const customSquareStyles = {};

  if (!boardFlipped) {
    customSquareStyles.a1 =
      cornerStyle("15px 0 0 0");

    customSquareStyles.h1 =
      cornerStyle("0 15px 0 0");

    customSquareStyles.a8 =
      cornerStyle("0 0 0 15px");

    customSquareStyles.h8 =
      cornerStyle("0 0 15px 0");
  } else {
    customSquareStyles.a8 =
      cornerStyle("15px 0 0 0");

    customSquareStyles.h8 =
      cornerStyle("0 15px 0 0");

    customSquareStyles.a1 =
      cornerStyle("0 0 0 15px");

    customSquareStyles.h1 =
      cornerStyle("0 0 15px 0");
  }

  /*
   * Последний ход.
   */
  if (lastMove) {
    customSquareStyles[lastMove.from] = {
      backgroundColor:
        "rgba(201, 169, 110, 0.45)",
    };

    customSquareStyles[lastMove.to] = {
      backgroundColor:
        "rgba(201, 169, 110, 0.6)",
    };
  }

  /*
   * Выбранная клетка.
   */
  if (selectedSquare) {
    customSquareStyles[selectedSquare] = {
      backgroundColor:
        "rgba(201, 169, 110, 0.55)",
    };
  }

  /*
   * Легальные ходы.
   */
  for (const sq of legalMovesForSelected) {
    const targetPiece = game.get(sq);

    if (!customSquareStyles[sq]) {
      if (targetPiece) {
        customSquareStyles[sq] = {
          backgroundImage:
            "radial-gradient(circle, transparent 48%, rgba(100,80,40,0.25) 50%, rgba(100,80,40,0.25) 80%, transparent 82%)",
          backgroundPosition: "center",
          backgroundSize: "100% 100%",
          backgroundRepeat: "no-repeat",
        };
      } else {
        customSquareStyles[sq] = {
          backgroundImage:
            "radial-gradient(circle, rgba(100,80,40,0.2) 15%, transparent 16%)",
          backgroundPosition: "center",
          backgroundSize: "100% 100%",
          backgroundRepeat: "no-repeat",
        };
      }
    }
  }

  /*
   * Предход рисуем последним, чтобы он перекрывал подсветку
   * последнего хода и выбранной клетки.
   */
  if (premove) {
    customSquareStyles[premove.from] = {
      ...(customSquareStyles[premove.from] || {}),
      backgroundColor: "rgba(179, 84, 58, 0.5)",
    };

    customSquareStyles[premove.to] = {
      ...(customSquareStyles[premove.to] || {}),
      backgroundColor: "rgba(179, 84, 58, 0.72)",
    };
  }

  /*
   * Две стрелки одновременно.
   *
   * Если оба движка выбрали один и тот же ход,
   * показываем одну зелёную стрелку.
   *
   * Если ходы разные - показываем обе.
   */
  const boardArrows = [];

  if (premove) {
    boardArrows.push([
      premove.from,
      premove.to,
      "rgba(179, 84, 58, 0.9)",
    ]);
  }

  if (stockfishArrow) {
    boardArrows.push(stockfishArrow);
  }

  if (
    maiaArrow &&
    (!stockfishArrow ||
      maiaArrow[0] !== stockfishArrow[0] ||
      maiaArrow[1] !== stockfishArrow[1])
  ) {
    boardArrows.push(maiaArrow);
  }

  /*
   * «Призрак» фигуры на целевой клетке предхода. Абсолютный слой
   * внутри boardBox, поэтому на расчёт размера доски не влияет.
   */
  const premoveGhost = (() => {
    if (!premove) return null;

    const file = premove.to.charCodeAt(0) - 97;
    const rank = Number(premove.to[1]);
    const squareSize = boardWidth / 8;
    const left =
      (boardFlipped ? 7 - file : file) * squareSize;
    const top =
      (boardFlipped ? rank - 1 : 8 - rank) * squareSize;
    const colorName =
      premove.color === "w" ? "white" : "black";

    return (
      <img
        src={`/pieces/${colorName}_${PIECE_FILE_NAMES[premove.type]}.svg`}
        alt=""
        aria-hidden="true"
        draggable={false}
        style={{
          position: "absolute",
          left,
          top,
          width: squareSize,
          height: squareSize,
          opacity: 0.55,
          pointerEvents: "none",
          zIndex: 8,
        }}
      />
    );
  })();

  /*
   * «Призрак» фигуры на клетке, откуда её тянут. react-chessboard
   * скрывает оригинал, и без такого слоя теряется точка старта.
   */
  const dragGhost = (() => {
    if (!dragFrom) return null;

    const { square, piece } = dragFrom;
    const file = square.charCodeAt(0) - 97;
    const rank = Number(square[1]);
    const squareSize = boardWidth / 8;
    const left =
      (boardFlipped ? 7 - file : file) * squareSize;
    const top =
      (boardFlipped ? rank - 1 : 8 - rank) * squareSize;
    const colorName =
      piece[0] === "w" ? "white" : "black";
    const pieceName =
      PIECE_FILE_NAMES[piece[1]?.toLowerCase()];

    if (!pieceName) return null;

    return (
      <img
        src={`/pieces/${colorName}_${pieceName}.svg`}
        alt=""
        aria-hidden="true"
        draggable={false}
        style={{
          position: "absolute",
          left,
          top,
          width: squareSize,
          height: squareSize,
          opacity: 0.4,
          pointerEvents: "none",
          zIndex: 7,
        }}
      />
    );
  })();

  return (
    <div className="board-section">
      <div className="board-wrapper" data-logo-anchor>

        {/* =====================================================
            Верхняя панель
        ====================================================== */}
        <div className="player-bar">
          {/*
           * Перевес верхней стороны живёт в её же строке, а не плашкой
           * поверх доски: так он ничего не перекрывает.
           */}
          <MaterialDisplay
            {...materialByColor[boardFlipped ? "w" : "b"]}
          />
        </div>

        {maiaError && (
          <div className="maia-engine-error" role="alert">
            <span>{maiaError}</span>
            <button
              type="button"
              onClick={() => makeAiMove()}
              disabled={isAiThinking}
            >
              Повторить
            </button>
          </div>
        )}

        {/* =====================================================
            Доска
        ====================================================== */}
        <div
          ref={boardBoxRef}
          className="eval-board-surface"
          style={{
            position: "relative",
            display: "inline-block",
          }}
        >
          <ReactChessboard
            key={boardKey}
            id={boardId}
            position={fen}
            onPieceDrop={onDrop}
            onSquareClick={onSquareClick}
            onPieceDragBegin={handlePieceDragBegin}
            onPieceDragEnd={handlePieceDragEnd}
            onSquareRightClick={handleSquareRightClick}
            onPromotionCheck={handlePromotionCheck}
            isDraggablePiece={isDraggablePiece}
            boardWidth={boardWidth}
            animationDuration={200}
            boardOrientation={
              boardFlipped
                ? "black"
                : "white"
            }
            showBoardNotation={true}
            customNotationStyle={{
              fontSize: "12px",
              fontFamily:
                "Inter, sans-serif",
              fontWeight: "500",
              color: "#4a3220",
            }}
            areArrowsAllowed={true}

            /*
             * Здесь теперь может быть:
             *
             * [
             *   ["g8", "f6", "зелёный"],
             *   ["f8", "c5", "оранжевый"]
             * ]
             */
            customArrows={boardArrows}

            customArrowColor="rgba(0, 150, 50, 0.75)"

            onPromotionPieceSelect={
              handlePromotionPieceSelect
            }

            customPieces={{
              wK: () => (
                <img
                  src="/pieces/white_king.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              wQ: () => (
                <img
                  src="/pieces/white_queen.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              wR: () => (
                <img
                  src="/pieces/white_rook.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              wB: () => (
                <img
                  src="/pieces/white_bishop.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              wN: () => (
                <img
                  src="/pieces/white_knight.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              wP: () => (
                <img
                  src="/pieces/white_pawn.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bK: () => (
                <img
                  src="/pieces/black_king.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bQ: () => (
                <img
                  src="/pieces/black_queen.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bR: () => (
                <img
                  src="/pieces/black_rook.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bB: () => (
                <img
                  src="/pieces/black_bishop.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bN: () => (
                <img
                  src="/pieces/black_knight.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),

              bP: () => (
                <img
                  src="/pieces/black_pawn.svg"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                />
              ),
            }}

            customBoardStyle={{
              borderRadius: "15px",
              backgroundImage:
                "linear-gradient(0deg, rgba(74, 178, 45, 0.43) 0%, rgba(74, 178, 45, 0.43) 100%), url(/textures/green-marble.png)",
              backgroundPosition:
                "-0.111px 0px",
              backgroundSize:
                "100.027% 100%",
              backgroundRepeat:
                "no-repeat",
              backgroundColor:
                "lightgray",
            }}

            customDarkSquareStyle={{
              boxShadow:
                "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
              backgroundColor:
                "rgba(223, 239, 252, 0.20)",
            }}

            customLightSquareStyle={{
              boxShadow:
                "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
              backgroundImage:
                "url(/textures/white-marble.png)",
              backgroundPosition: "50%",
              backgroundSize: "cover",
              backgroundRepeat:
                "no-repeat",
              backgroundColor:
                "rgba(255, 255, 255, 0.75)",
            }}

            customSquareStyles={
              customSquareStyles
            }
          />

          {dragGhost}

          {premoveGhost}

          {/*
           * Выбор стороны - та же карточка, что и итог партии.
           * Открывается при входе на главную и по кнопке «Новая игра»;
           * клик по фону оставляет текущую сторону и партию.
           */}
          {sidePickerOpen && (
            <div
              className="gameover-overlay"
              onClick={() => setSidePickerOpen(false)}
            >
              <div
                className="gameover-card gameover-card--side"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="gameover-title">
                  За кого играем?
                </div>
                <div className="gameover-reason">
                  Партия начнётся с начальной позиции
                </div>
                <div className="side-picker">
                  <button
                    type="button"
                    className="side-pick"
                    onClick={() => handleChooseSide("w")}
                  >
                    <img
                      className="side-pick-piece"
                      src="/pieces/white_king.svg"
                      alt=""
                    />
                    <span className="side-pick-name">
                      Белые
                    </span>
                  </button>
                  <button
                    type="button"
                    className="side-pick"
                    onClick={() => handleChooseSide("b")}
                  >
                    <img
                      className="side-pick-piece"
                      src="/pieces/black_king.svg"
                      alt=""
                    />
                    <span className="side-pick-name">
                      Чёрные
                    </span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/*
           * Итог партии в духе chess.com: затемнение доски + карточка
           * со счётом, причиной и кнопкой новой игры. Клик по фону
           * прячет карточку, чтобы рассмотреть финальную позицию.
           */}
          {gameResult && !resultCardDismissed && (
            <div
              className="gameover-overlay"
              onClick={() => setResultCardDismissed(true)}
            >
              <div
                className={`gameover-card gameover-card--${gameResult.outcome}`}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="gameover-title">
                  {gameResult.outcome === "win"
                    ? "Победа!"
                    : gameResult.outcome === "loss"
                      ? "Поражение"
                      : "Ничья"}
                </div>
                <div className="gameover-reason">
                  {gameResult.reason}
                </div>

                {/*
                 * Разбор партии: до нажатия - кнопка, после ответа
                 * сервера - звёзды, точность и счётчик ходов.
                 */}
                {review ? (
                  <GameReviewSummary review={review} />
                ) : (
                  <>
                    <button
                      type="button"
                      className="gameover-btn gameover-btn--review"
                      onClick={handleReviewGame}
                      disabled={reviewLoading}
                    >
                      {reviewLoading
                        ? "Разбираем партию..."
                        : "Разобрать партию"}
                    </button>
                    {reviewLoading && (
                      <div className="gameover-review-note">
                        Движок проверяет каждый ваш ход, это может занять
                        до 40 секунд
                      </div>
                    )}
                  </>
                )}

                {reviewError && (
                  <div className="gameover-review-note gameover-review-note--error">
                    {reviewError}
                  </div>
                )}

                <div className="gameover-actions">
                  {/*
                   * Разбор уже готов: главное действие - посмотреть ходы
                   * с оценками. Кнопка прячет карточку, чтобы доска
                   * освободилась для листания.
                   */}
                  {review && (
                    <button
                      type="button"
                      className="gameover-btn gameover-btn--review"
                      onClick={() => setResultCardDismissed(true)}
                    >
                      Разбор по ходам
                    </button>
                  )}
                  <button
                    type="button"
                    className="gameover-btn"
                    onClick={openSidePicker}
                  >
                    Новая игра
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* =====================================================
            Кнопки
        ====================================================== */}
        <div className="controls-row">

          <button
            className="ctrl-btn"
            style={{
              background: "#C9A96E",
            }}
            onClick={() =>
              setBoardFlipped(
                (f) => !f,
              )
            }
          >
            ↺ Доска
          </button>

          {/*
           * «Новая игра» открывает меню выбора стороны: партия начнётся
           * сразу после выбора. Клик по фону - отмена, текущая партия
           * остаётся.
           */}
          <button
            className="ctrl-btn"
            style={{
              background: "#C9A96E",
            }}
            onClick={
              openSidePicker
            }
          >
            Новая игра
          </button>

          {/*
           * Разбор доступен и после того, как итоговую карточку закрыли
           * кликом по фону: кнопка открывает её снова и запускает разбор.
           */}
          {gameResult && (
            <button
              type="button"
              className="ctrl-btn ctrl-btn--review"
              onClick={openReviewCard}
              disabled={reviewLoading}
            >
              {reviewLoading
                ? "Разбираем..."
                : review
                  ? "Итог партии"
                  : "Разобрать партию"}
            </button>
          )}

          {/*
           * На узком экране шкала становится тонкой полосой, и
           * переключатель внутри неё не помещается: там режим
           * меняется этой кнопкой. На широком экране она скрыта -
           * переключатель есть в самой шкале.
           */}
          <button
            type="button"
            className="ctrl-btn ctrl-btn--eval-mode"
            onClick={onToggleEvalMode}
            disabled={!onToggleEvalMode}
          >
            {evalMode === "engine" ? "Оценка: движок" : "Оценка: материал"}
          </button>

          {/*
           * Перевес нижней стороны - в её строке под доской,
           * а не плашкой поверх поля.
           */}
          <MaterialDisplay
            {...materialByColor[boardFlipped ? "b" : "w"]}
          />
        </div>

        {/* =====================================================
            Легенда сравнения
        ====================================================== */}
        {compareData && (
          <div className="compare-card" role="status">
            <div className="compare-card__row">
              <span
                className="compare-card__dot compare-card__dot--stockfish"
                aria-hidden="true"
              />
              <strong className="compare-card__label">Stockfish:</strong>{" "}
              {compareData.stockfish?.san || "0"}
            </div>

            <div className="compare-card__row">
              <span
                className="compare-card__dot compare-card__dot--maia"
                aria-hidden="true"
              />
              <strong className="compare-card__label">Maia3:</strong>{" "}
              {compareData.maia3?.san || "0"} ({elo} ELO)
            </div>

            {typeof compareData.same_move === "boolean" && (
              <div className="compare-card__note">
                {compareData.same_move
                  ? "Maia3 выбрала оптимальный ход Stockfish."
                  : "Maia3 выбрала другой ход - это и есть различие между оптимальной и человекоподобной игрой."}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default ChessboardComponent;

