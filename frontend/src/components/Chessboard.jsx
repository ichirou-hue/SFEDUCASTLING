import {
  useState,
  useRef,
  useCallback,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import FenBar from "./FenBar.jsx";

import {
  fetchMaiaMove,
  fetchEval,
  fetchStockfishAnalysis,
  fetchCompareMoves,
  saveMoveToDataset,
} from "../api.js";

const userId =
  localStorage.getItem("sfedu_user_id") ||
  "user_" + Math.random().toString(36).substr(2, 9);

localStorage.setItem("sfedu_user_id", userId);

let gameId = "game_" + Date.now();

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
  { onStateChange },
  ref,
) {
  const gameRef = useRef(new Chess());

  const [fen, setFen] = useState(gameRef.current.fen());
  const [moveHistory, setMoveHistory] = useState([]);
  const [turn, setTurn] = useState("w");

  const [boardFlipped, setBoardFlipped] = useState(false);
  const [elo, setElo] = useState(1500);

  const [isAiThinking, setIsAiThinking] = useState(false);

  const [lastMove, setLastMove] = useState(null);

  const [positionSnapshots, setPositionSnapshots] = useState([]);
  const [viewIndex, setViewIndex] = useState(-1);
  const [isViewMode, setIsViewMode] = useState(false);

  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalMovesForSelected, setLegalMovesForSelected] = useState([]);

  const [boardWidth, setBoardWidth] = useState(560);
  const boardBoxRef = useRef(null);

  const [evalScore, setEvalScore] = useState(null);

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

  const game = gameRef.current;

  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        boardFlipped,
      }));
    }
  }, [boardFlipped, onStateChange]);

  useEffect(() => {
    if (!boardBoxRef.current) return;

    const el = boardBoxRef.current;

    const report = () => {
      if (typeof onStateChange === "function") {
        onStateChange((prev) => ({
          ...prev,
          boardHeight: el.offsetHeight,
        }));
      }
    };

    report();

    const ro = new ResizeObserver(report);
    ro.observe(el);

    return () => ro.disconnect();
  }, [onStateChange]);

  /*
   * Размер доски.
   */
  useEffect(() => {
    const updateSize = () => {
      const sideW = 300;
      const gapW = 40;
      const uiHeight = 110;

      const availW = window.innerWidth - sideW - gapW - 40;
      const availH = window.innerHeight * 0.85 - uiHeight;

      const size = Math.floor(Math.min(availW, availH));

      setBoardWidth(Math.max(320, Math.min(size, 560)));
    };

    updateSize();

    window.addEventListener("resize", updateSize);

    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const updateStatus = useCallback(() => {
    const g = gameRef.current;
    setTurn(g.turn());
  }, []);

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
  const onNavigate = useCallback(
    (direction) => {
      const g = gameRef.current;
      const history = moveHistory;

      if (history.length === 0) return;

      const START_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

      const getFen = (index) => {
        if (index === -1) return START_FEN;

        const tempGame = new Chess();

        for (let i = 0; i <= index; i++) {
          if (history[i]) {
            tempGame.move(history[i]);
          }
        }

        return tempGame.fen();
      };

      if (typeof direction === "number") {
        const targetIndex = direction;

        if (targetIndex >= history.length - 1) {
          setIsViewMode(false);
          setViewIndex(-1);

          const lastFen = getFen(history.length - 1);

          g.load(lastFen);
          setFen(lastFen);

          setStockfishArrow(null);
          setMaiaArrow(null);
          setCompareData(null);

          return;
        }

        setIsViewMode(true);
        setViewIndex(targetIndex);

        const fenToLoad = getFen(targetIndex);

        g.load(fenToLoad);
        setFen(fenToLoad);

        setStockfishArrow(null);
        setMaiaArrow(null);
        setCompareData(null);

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

      const fenToLoad = getFen(currentIndex);

      g.load(fenToLoad);
      setFen(fenToLoad);

      setStockfishArrow(null);
      setMaiaArrow(null);
      setCompareData(null);

      if (currentIndex >= history.length - 1) {
        setIsViewMode(false);
        setViewIndex(-1);
      } else {
        setIsViewMode(true);
        setViewIndex(currentIndex);
      }
    },
    [moveHistory, isViewMode, viewIndex],
  );

  /*
   * Новая игра.
   */
  const handleNewGame = useCallback(() => {
    game.reset();

    const startFen = game.fen();

    setFen(startFen);
    setMoveHistory([]);
    setPositionSnapshots([startFen]);

    setViewIndex(-1);
    setIsViewMode(false);

    setLastMove(null);
    setSelectedSquare(null);
    setLegalMovesForSelected([]);

    setIsAiThinking(false);

    setEvalScore(null);

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
      }));
    }
  }, [game, onStateChange]);

  /*
   * Загрузка FEN.
   */
  const handleLoadFen = useCallback(
    (newFen) => {
      try {
        game.load(newFen);

        const validFen = game.fen();

        setFen(validFen);
        setMoveHistory([]);
        setPositionSnapshots([validFen]);

        setViewIndex(-1);
        setIsViewMode(false);

        setLastMove(null);
        setSelectedSquare(null);
        setLegalMovesForSelected([]);

        setIsAiThinking(false);

        setEvalScore(null);

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
      } catch (error) {
        console.error("Неверный FEN:", error);
        alert("Ошибка: Неверный формат FEN-строки!");
      }
    },
    [game, onStateChange],
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

      try {
        /*
         * История передаётся в backend в UCI.
         *
         * moveHistory содержит SAN, поэтому восстанавливаем
         * партию и получаем UCI-последовательность.
         */
        const historyGame = new Chess();
        const uciMoves = [];

        for (const san of moveHistory) {
          try {
            const move = historyGame.move(san);

            if (move) {
              uciMoves.push(move.from + move.to + (move.promotion || ""));
            }
          } catch (error) {
            console.warn(
              "[Compare] Не удалось восстановить ход:",
              san,
            );
            break;
          }
        }

        /*
         * ВАЖНО:
         * если пользователь загрузил произвольный FEN без истории,
         * moves будет [].
         */
        const data = await fetchCompareMoves(
          positionFen,
          elo,
          uciMoves,
        );

        if (!data || data.error) {
          console.warn("[Compare] Ошибка:", data?.error);
          setStockfishArrow(null);
          setMaiaArrow(null);
          setCompareData(null);
          return;
        }

        /*
         * Stockfish:
         * зелёная стрелка.
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
         * Maia3:
         * оранжевая стрелка.
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
         * Сохраняем ответ для информационного блока.
         */
        setCompareData(data);
      } catch (error) {
        console.error(
          "[Compare] Ошибка сравнения Stockfish/Maia3:",
          error,
        );

        setStockfishArrow(null);
        setMaiaArrow(null);
        setCompareData(null);
      } finally {
        setIsComparing(false);
      }
    },
    [
      game,
      elo,
      moveHistory,
      isComparing,
    ],
  );

  /*
   * Ход Maia3.
   *
   * Здесь Maia3 выступает именно как соперник.
   */
  const makeAiMove = useCallback(async () => {
    if (game.isGameOver() || game.isDraw()) return;

    setIsAiThinking(true);

    /*
     * Пока Maia делает ход, старые стрелки сравнения убираем.
     */
    setStockfishArrow(null);
    setMaiaArrow(null);
    setCompareData(null);

    try {
      /*
       * Формируем историю партии в UCI.
       */
      const historyGame = new Chess();
      const uciMoves = [];

      for (const san of moveHistory) {
        try {
          const move = historyGame.move(san);

          if (move) {
            uciMoves.push(
              move.from +
                move.to +
                (move.promotion || ""),
            );
          }
        } catch (error) {
          console.warn(
            "[Maia3] Не удалось восстановить ход:",
            san,
          );
          break;
        }
      }

      const data = await fetchMaiaMove(
        game.fen(),
        elo,
        uciMoves,
      );

      if (data.error) {
        setIsAiThinking(false);
        return;
      }

      const move = game.move({
        from: data.from,
        to: data.to,
        promotion: "q",
      });

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

        playMoveSound();

        if (data.evaluation) {
          setEvalScore(data.evaluation);
        }

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

      /*
       * Fallback — случайный легальный ход.
       */
      const legalMoves = game.moves({
        verbose: true,
      });

      if (legalMoves.length > 0) {
        const pick =
          legalMoves[
            Math.floor(
              Math.random() * legalMoves.length,
            )
          ];

        game.move(pick);

        const currentFen = game.fen();

        setFen(currentFen);

        setLastMove({
          from: pick.from,
          to: pick.to,
        });

        buildMoveHistory(pick.san);
        takeSnapshot();
        updateStatus();

        playMoveSound();

        if (typeof onStateChange === "function") {
          onStateChange((prev) => ({
            ...prev,
            fen: currentFen,
            moveHistory: [
              ...(prev.moveHistory || []),
              pick.san,
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
    }

    setIsAiThinking(false);
  }, [
    game,
    elo,
    moveHistory,
    buildMoveHistory,
    takeSnapshot,
    updateStatus,
    onStateChange,
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
   * Оценка конкретной позиции.
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
   * ============================================================
   * ХОД ИГРОКА
   * ============================================================
   */
  const processPlayerMove = useCallback(
    (sourceSquare, targetSquare) => {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });

      if (move === null) {
        return false;
      }

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
        userId,
        gameId,
      ).catch(() => {});

      /*
       * После хода игрока Maia3 отвечает.
       */
      if (
        !game.isGameOver() &&
        !game.isDraw()
      ) {
        setTimeout(makeAiMove, 500);
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
    ],
  );

  /*
   * Drag & Drop.
   */
  const onDrop = useCallback(
    (sourceSquare, targetSquare) => {
      if (
        isViewMode ||
        isAiThinking ||
        game.turn() !== "w"
      ) {
        return false;
      }

      return processPlayerMove(
        sourceSquare,
        targetSquare,
      );
    },
    [
      game,
      isViewMode,
      isAiThinking,
      processPlayerMove,
    ],
  );

  /*
   * Клик по клеткам.
   */
  const onSquareClick = useCallback(
    (square) => {
      if (isViewMode) return;
      if (isAiThinking) return;
      if (game.turn() !== "w") return;

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
      isAiThinking,
      processPlayerMove,
    ],
  );

  const handlePromotionPieceSelect =
    useCallback(() => {
      return true;
    }, []);

  /*
   * Публичные методы компонента.
   */
  useImperativeHandle(ref, () => ({
    onNavigate,
    handleNewGame,
    getFen: () => gameRef.current.fen(),
    handleLoadFen,
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
    const excess = {};

    if (diff > 0) {
      let remaining = diff;

      ["q", "r", "b", "n", "p"].forEach(
        (type) => {
          const count =
            whitePieces[type] || 0;

          const needed = Math.min(
            Math.floor(
              remaining /
                pieceValues[type],
            ),
            count,
          );

          if (needed > 0) {
            excess[type] = needed;

            remaining -=
              needed * pieceValues[type];
          }
        },
      );
    } else if (diff < 0) {
      let remaining = Math.abs(diff);

      ["q", "r", "b", "n", "p"].forEach(
        (type) => {
          const count =
            blackPieces[type] || 0;

          const needed = Math.min(
            Math.floor(
              remaining /
                pieceValues[type],
            ),
            count,
          );

          if (needed > 0) {
            excess[type] = needed;

            remaining -=
              needed * pieceValues[type];
          }
        },
      );
    }

    return {
      diff,
      excess,
    };
  }, [game]);

  const materialAdv =
    getMaterialAdvantage();

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

  const MaterialDisplay = ({ diff }) => {
    if (diff <= 0) return null;

    return (
      <div className="material-display">
        <span className="material-diff">
          +{diff}
        </span>
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
   * Две стрелки одновременно.
   *
   * Если оба движка выбрали один и тот же ход,
   * показываем одну зелёную стрелку.
   *
   * Если ходы разные — показываем обе.
   */
  const boardArrows = [];

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

  return (
    <div className="board-section">
      <div className="board-wrapper">

        {/* =====================================================
            Верхняя панель
        ====================================================== */}
        <div className="player-bar">
          <div className="rating-label">
            Противник (Maia3):{" "}
            <span>{elo}</span> ELO
          </div>

          <input
            type="range"
            min="1000"
            max="2600"
            step="100"
            value={elo}
            onChange={(e) => {
              const newElo = parseInt(
                e.target.value,
                10,
              );

              setElo(newElo);

              /*
               * При изменении Elo старое сравнение
               * Maia3 больше не актуально.
               */
              setStockfishArrow(null);
              setMaiaArrow(null);
              setCompareData(null);

              if (
                typeof onStateChange ===
                "function"
              ) {
                onStateChange((prev) => ({
                  ...prev,
                  maiaRating: newElo,
                }));
              }
            }}
            style={{
              width: 100,
              accentColor: "#C9A96E",
              cursor: "pointer",
            }}
          />
        </div>

        {/* =====================================================
            Доска
        ====================================================== */}
        <div
          ref={boardBoxRef}
          style={{
            position: "relative",
            display: "inline-block",
          }}
        >
          <MaterialDisplay
            diff={
              boardFlipped
                ? materialAdv.diff
                : -materialAdv.diff
            }
          />

          <ReactChessboard
            position={fen}
            onPieceDrop={onDrop}
            onSquareClick={onSquareClick}
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
                "Cormorant Garamond, Georgia, serif",
              fontWeight: "500",
              color: "#225a73",
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

          <MaterialDisplay
            diff={
              boardFlipped
                ? -materialAdv.diff
                : materialAdv.diff
            }
          />
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

          <button
            className="ctrl-btn"
            style={{
              background: "#C9A96E",
            }}
            onClick={
              handleNewGame
            }
          >
            Новая игра
          </button>

          <button
            className="ctrl-btn"
            style={{
              background: "#83b787",
              opacity:
                isComparing
                  ? 0.7
                  : 1,
            }}
            disabled={isComparing}
            onClick={() =>
              fetchBestMove(
                game.fen(),
              )
            }
          >
            {isComparing
              ? "Анализ..."
              : "Лучший ход"}
          </button>
        </div>

        {/* =====================================================
            Легенда сравнения
        ====================================================== */}
        {compareData && (
          <div
            style={{
              marginTop: "10px",
              padding: "10px 14px",
              borderRadius: "10px",
              background:
                "rgba(255,255,255,0.75)",
              fontFamily:
                "Cormorant Garamond, Georgia, serif",
              fontSize: "15px",
              lineHeight: 1.5,
            }}
          >
            <div>
              <span
                style={{
                  display:
                    "inline-block",
                  width: "12px",
                  height: "12px",
                  borderRadius:
                    "50%",
                  background:
                    "rgba(0, 150, 50, 0.85)",
                  marginRight: "7px",
                }}
              />

              <strong>
                Stockfish:
              </strong>{" "}
              {compareData.stockfish?.san ||
                "—"}
            </div>

            <div>
              <span
                style={{
                  display:
                    "inline-block",
                  width: "12px",
                  height: "12px",
                  borderRadius:
                    "50%",
                  background:
                    "rgba(220, 120, 20, 0.9)",
                  marginRight: "7px",
                }}
              />

              <strong>
                Maia3:
              </strong>{" "}
              {compareData.maia3?.san ||
                "—"}{" "}
              ({elo} ELO)
            </div>

            {typeof compareData.same_move ===
              "boolean" && (
              <div
                style={{
                  marginTop: "4px",
                  opacity: 0.75,
                }}
              >
                {compareData.same_move
                  ? "Maia3 выбрала оптимальный ход Stockfish."
                  : "Maia3 выбрала другой ход — это и есть различие между оптимальной и человекоподобной игрой."}
              </div>
            )}
          </div>
        )}

        <FenBar
          fen={fen}
          onLoadFen={handleLoadFen}
        />
      </div>
    </div>
  );
});

export default ChessboardComponent;

