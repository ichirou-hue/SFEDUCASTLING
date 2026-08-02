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
import { fetchMaiaMove, fetchEval, saveMoveToDataset } from "../api.js";

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
  const [evalScore, setEvalScore] = useState(null);

  const game = gameRef.current;

  useEffect(() => {
    const updateSize = () => {
      const topH = 48;
      const sideW = 340;
      const gapW = 60;
      const padW = 120;
      const availW = window.innerWidth - sideW * 2 - gapW - padW;
      const availH = window.innerHeight - topH - 100;
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

  const buildMoveHistory = useCallback(
    (newMoveSan) => {
      setMoveHistory((prev) => {
        const safePrev = prev || [];
        const validHistory =
          viewIndex === -1 ? safePrev : safePrev.slice(0, viewIndex + 1);
        return [...validHistory, newMoveSan];
      });
    },
    [viewIndex],
  );

  const takeSnapshot = useCallback(() => {
    setPositionSnapshots((prev) => {
      const safePrev = prev || [];
      const validSnapshots =
        viewIndex === -1 ? safePrev : safePrev.slice(0, viewIndex + 2);

      return [...validSnapshots, game.fen()];
    });
    setViewIndex(-1);
    setIsViewMode(false);
  }, [viewIndex, game]);

  const onNavigate = useCallback(
    (direction) => {
      const g = gameRef.current;
      const history = moveHistory; // Используем текстовую историю ходов, она всегда точна!
      const START_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

      if (history.length === 0) return;

      // Вспомогательная функция: 100% точно восстанавливает FEN, проигрывая ходы заново
      const getFenForIndex = (index) => {
        if (index === -1) return START_FEN;
        const tempGame = new Chess();
        for (let i = 0; i <= index; i++) {
          if (history[i]) tempGame.move(history[i]);
        }
        return tempGame.fen();
      };

      // 1. Если кликнули мышкой по конкретному ходу в списке
      if (typeof direction === "number") {
        const targetIndex = direction;
        if (targetIndex >= history.length - 1) {
          setIsViewMode(false);
          setViewIndex(-1);
          const lastFen = getFenForIndex(history.length - 1);
          g.load(lastFen);
          setFen(lastFen);
        } else {
          setIsViewMode(true);
          setViewIndex(targetIndex);
          const fenToLoad = getFenForIndex(targetIndex);
          g.load(fenToLoad);
          setFen(fenToLoad);
        }
        return;
      }

      // 2. Логика для стрелок
      let currentIndex = isViewMode ? viewIndex : history.length - 1;

      if (direction === "first") {
        currentIndex = -1;
      } else if (direction === "prev") {
        currentIndex = Math.max(-1, currentIndex - 1);
      } else if (direction === "next") {
        currentIndex = Math.min(history.length - 1, currentIndex + 1);
      } else if (direction === "last") {
        currentIndex = history.length - 1;
      }

      // 3. Загружаем нужную позицию на доску
      if (currentIndex >= history.length - 1) {
        setIsViewMode(false);
        setViewIndex(-1);
        const lastFen = getFenForIndex(history.length - 1);
        g.load(lastFen);
        setFen(lastFen);
      } else {
        setIsViewMode(true);
        setViewIndex(currentIndex);
        const fenToLoad = getFenForIndex(currentIndex);
        g.load(fenToLoad);
        setFen(fenToLoad);
      }
    },
    [moveHistory, isViewMode, viewIndex],
  );

  const handleNewGame = useCallback(() => {
    // 1. Сбрасываем внутреннюю логику шахмат
    game.reset();
    const startFen = game.fen();

    // 2. Локальный сброс состояний доски
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
    fetchEval(startFen).then((data) => {
      if (data.evaluation) setEvalScore(data.evaluation);
    }).catch(() => {});

    // 3. Глобальный сброс для левой панели, чтобы стереть список
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        fen: startFen,
        moveHistory: [], // <-- Жестко очищаем список ходов
        positionSnapshots: [startFen], // <-- Оставляем только стартовую позицию
        viewIndex: -1,
        turn: "w",
        evalScore: null,
      }));
    }
  }, [game, onStateChange]);
  const handleLoadFen = useCallback(
    (newFen) => {
      try {
        // Пытаемся загрузить FEN в шахматный движок
        game.load(newFen);
        const validFen = game.fen(); // Получаем проверенный FEN от движка

        // 1. Локальный сброс состояний доски под новую позицию
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
        fetchEval(validFen).then((data) => {
          if (data.evaluation) setEvalScore(data.evaluation);
        }).catch(() => {});

        // 2. Глобальный сброс для левой панели
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

  const makeAiMove = useCallback(async () => {
    if (game.isGameOver() || game.isDraw()) return;
    setIsAiThinking(true);

    try {
      const data = await fetchMaiaMove(game.fen(), elo);
      if (data.error) {
        setIsAiThinking(false);
        return;
      }
      const move = game.move({ from: data.from, to: data.to, promotion: "q" });

      if (move) {
        const currentFen = game.fen();
        setFen(currentFen);
        setLastMove({ from: data.from, to: data.to });
        buildMoveHistory(move.san);
        takeSnapshot();
        updateStatus();
        playMoveSound();
        if (data.evaluation) setEvalScore(data.evaluation);

        // Отправка хода БОТА в глобальный список
        if (typeof onStateChange === "function") {
          onStateChange((prev) => ({
            ...prev,
            fen: currentFen,
            moveHistory: [...(prev.moveHistory || []), move.san],
            positionSnapshots: [...(prev.positionSnapshots || []), currentFen],
            viewIndex: -1,
            turn: game.turn(),
          }));
        }
      }
    } catch (err) {
      console.error("Maia2 ошибка:", err);
      const legalMoves = game.moves({ verbose: true });
      if (legalMoves.length > 0) {
        const pick = legalMoves[Math.floor(Math.random() * legalMoves.length)];
        game.move(pick);
        const currentFen = game.fen();
        setFen(currentFen);
        setLastMove({ from: pick.from, to: pick.to });
        buildMoveHistory(pick.san);
        takeSnapshot();
        updateStatus();
        playMoveSound();

        // То же самое, если бот сделал случайный ход из-за ошибки сети
        if (typeof onStateChange === "function") {
          onStateChange((prev) => ({
            ...prev,
            fen: currentFen,
            moveHistory: [...(prev.moveHistory || []), pick.san],
            positionSnapshots: [...(prev.positionSnapshots || []), currentFen],
            viewIndex: -1,
            turn: game.turn(),
          }));
        }
      }
    }
    setIsAiThinking(false);
  }, [game, elo, buildMoveHistory, takeSnapshot, updateStatus, onStateChange]);
  useEffect(() => {
    const updateSize = () => {
      const sideW = 300; // Ширина левой панели
      const gapW = 40; // Расстояние между панелью и доской
      const uiHeight = 110; // НОВОЕ: Резервируем место под FEN-строку и верхнюю плашку с ELO

      const availW = window.innerWidth - sideW - gapW - 40;
      // Вычитаем из 85vh высоту дополнительных элементов внутри рамки
      const availH = window.innerHeight * 0.85 - uiHeight;

      const size = Math.floor(Math.min(availW, availH));

      setBoardWidth(Math.max(320, size));
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    fetchEval(game.fen()).then((data) => {
      if (data.evaluation) setEvalScore(data.evaluation);
    }).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({ ...prev, evalScore }));
    }
  }, [evalScore, onStateChange]);

  const fetchEvalForPosition = useCallback((fen) => {
    fetchEval(fen).then((data) => {
      if (data.evaluation) setEvalScore(data.evaluation);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (typeof onStateChange === "function") {
      onStateChange((prev) => ({
        ...prev,
        viewIndex: viewIndex,
        isViewMode: isViewMode,
      }));
    }
  }, [viewIndex, isViewMode, onStateChange]);
  const onDrop = useCallback(
    (sourceSquare, targetSquare) => {
      if (isViewMode || isAiThinking || game.turn() !== "w") return false;

      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });
      if (move === null) return false;

      const currentFen = game.fen();

      // 1. Локальные вызовы
      setFen(currentFen);
      setLastMove({ from: sourceSquare, to: targetSquare });
      setSelectedSquare(null);
      setLegalMovesForSelected([]);
      buildMoveHistory(move.san);
      takeSnapshot();
      updateStatus();
      playMoveSound();
      fetchEvalForPosition(currentFen);

      // 2. Глобальная отправка ТВОЕГО хода
      if (typeof onStateChange === "function") {
        onStateChange((prev) => {
          const safeHistory = prev.moveHistory || [];
          const safeSnapshots = prev.positionSnapshots || [
            prev.fen || currentFen,
          ];

          const currentMoveHistory =
            prev.viewIndex === -1
              ? safeHistory
              : safeHistory.slice(0, prev.viewIndex + 1);
          const currentSnapshots =
            prev.viewIndex === -1
              ? safeSnapshots
              : safeSnapshots.slice(0, prev.viewIndex + 2);

          return {
            ...prev,
            fen: currentFen,
            moveHistory: [...currentMoveHistory, move.san], // <-- Сохраняем твой ход
            positionSnapshots: [...currentSnapshots, currentFen],
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

      if (!game.isGameOver() && !game.isDraw()) {
        setTimeout(makeAiMove, 500);
      }
      return true;
    },
    [
      game,
      isViewMode,
      isAiThinking,
      buildMoveHistory,
      takeSnapshot,
      updateStatus,
      makeAiMove,
      onStateChange,
      userId,
      gameId,
    ],
  );
  const onSquareClick = useCallback(
    (square) => {
      if (isViewMode) return;
      if (isAiThinking) return;
      if (game.turn() !== "w") return;

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

        const move = game.move({
          from: selectedSquare,
          to: square,
          promotion: "q",
        });

        // ЕСЛИ ХОД УСПЕШНЫЙ:
        if (move) {
          const currentFen = game.fen(); // Сохраняем текущий FEN

          // 1. Локальные обновления
          setFen(currentFen);
          setLastMove({ from: selectedSquare, to: square });
          setSelectedSquare(null);
          setLegalMovesForSelected([]);

          buildMoveHistory(move.san);
          takeSnapshot();
          updateStatus();
          playMoveSound();
          fetchEvalForPosition(currentFen);

          // 2. НОВОЕ: Глобальная отправка ТВОЕГО хода в левую панель
          if (typeof onStateChange === "function") {
            onStateChange((prev) => {
              const safeHistory = prev.moveHistory || [];
              const safeSnapshots = prev.positionSnapshots || [
                prev.fen || currentFen,
              ];

              const currentMoveHistory =
                prev.viewIndex === -1
                  ? safeHistory
                  : safeHistory.slice(0, prev.viewIndex + 1);
              const currentSnapshots =
                prev.viewIndex === -1
                  ? safeSnapshots
                  : safeSnapshots.slice(0, prev.viewIndex + 2);

              return {
                ...prev,
                fen: currentFen,
                moveHistory: [...currentMoveHistory, move.san],
                positionSnapshots: [...currentSnapshots, currentFen],
                viewIndex: -1,
                turn: game.turn(),
              };
            });
          }

          saveMoveToDataset(
            currentFen, // используем сохраненную переменную
            selectedSquare + square,
            userId,
            gameId,
          ).catch(() => {});

          if (!game.isGameOver() && !game.isDraw()) {
            setTimeout(makeAiMove, 500);
          }
          return;
        }

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
    [
      game,
      selectedSquare,
      isViewMode,
      isAiThinking,
      buildMoveHistory,
      takeSnapshot,
      updateStatus,
      makeAiMove,
      onStateChange,
      userId,
      gameId,
    ],
  );
  const handlePromotionPieceSelect = useCallback(() => {
    return true;
  }, []);

  useImperativeHandle(ref, () => ({
    onNavigate,
    handleNewGame,
    getFen: () => gameRef.current.fen(),
    handleLoadFen,
  }));

  const getMaterialAdvantage = useCallback(() => {
    const board = game.board();
    const pieceValues = { p: 1, n: 3, b: 3, r: 5, q: 9 };
    let white = 0;
    let black = 0;
    const whitePieces = {};
    const blackPieces = {};

    board.forEach((row) => {
      row.forEach((square) => {
        if (square) {
          const value = pieceValues[square.type] || 0;
          if (square.color === "w") {
            white += value;
            whitePieces[square.type] = (whitePieces[square.type] || 0) + 1;
          } else {
            black += value;
            blackPieces[square.type] = (blackPieces[square.type] || 0) + 1;
          }
        }
      });
    });

    const diff = white - black;
    const excess = {};

    if (diff > 0) {
      let remaining = diff;
      ["q", "r", "b", "n", "p"].forEach((type) => {
        const count = whitePieces[type] || 0;
        const needed = Math.min(Math.floor(remaining / pieceValues[type]), count);
        if (needed > 0) {
          excess[type] = needed;
          remaining -= needed * pieceValues[type];
        }
      });
    } else if (diff < 0) {
      let remaining = Math.abs(diff);
      ["q", "r", "b", "n", "p"].forEach((type) => {
        const count = blackPieces[type] || 0;
        const needed = Math.min(Math.floor(remaining / pieceValues[type]), count);
        if (needed > 0) {
          excess[type] = needed;
          remaining -= needed * pieceValues[type];
        }
      });
    }

    return { diff, excess };
  }, [game]);

  const materialAdv = getMaterialAdvantage();

  const MaterialDisplay = ({ diff }) => {
    if (diff <= 0) return null;

    return (
      <div className="material-display">
        <span className="material-diff">+{diff}</span>
      </div>
    );
  };

  const cornerStyle = (radius) => ({ borderRadius: radius });
  const customSquareStyles = {};
  if (!boardFlipped) {
    customSquareStyles.a1 = cornerStyle("15px 0 0 0");
    customSquareStyles.h1 = cornerStyle("0 15px 0 0");
    customSquareStyles.a8 = cornerStyle("0 0 0 15px");
    customSquareStyles.h8 = cornerStyle("0 0 15px 0");
  } else {
    customSquareStyles.a8 = cornerStyle("15px 0 0 0");
    customSquareStyles.h8 = cornerStyle("0 15px 0 0");
    customSquareStyles.a1 = cornerStyle("0 0 0 15px");
    customSquareStyles.h1 = cornerStyle("0 0 15px 0");
  }
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

  return (
    <div className="board-section">
      <div className="board-wrapper">
        <div className="player-bar">
          <div className="rating-label">
            Противник (Maia2): <span>{elo}</span> ELO
          </div>
          <input
            type="range"
            min="1100"
            max="1900"
            step="100"
            value={elo}
            onChange={(e) => {
              const newElo = parseInt(e.target.value, 10);

              // 1. Оставляем локальное обновление для самой доски
              setElo(newElo);

              // 2. Отправляем новое значение наверх в App.js
              if (typeof onStateChange === "function") {
                onStateChange((prev) => ({ ...prev, maiaRating: newElo }));
              }
            }}
            style={{ width: 100, accentColor: "#C9A96E", cursor: "pointer" }}
          />
        </div>

        <div style={{ position: "relative", display: "inline-block" }}>
        <MaterialDisplay
          diff={boardFlipped ? materialAdv.diff : -materialAdv.diff}
        />

        <ReactChessboard
          position={fen}
          onPieceDrop={onDrop}
          onSquareClick={onSquareClick}
          boardWidth={boardWidth}
          animationDuration={200}
          boardOrientation={boardFlipped ? "black" : "white"}
          showBoardNotation={true}
          customNotationStyle={{
            fontSize: "12px",
            fontFamily: "Cormorant Garamond, Georgia, serif",
            fontWeight: "500",
            color: "#225a73",
          }}
          areArrowsAllowed={true}
          onPromotionPieceSelect={handlePromotionPieceSelect}
          customPieces={{
            wK: () => (
              <img
                src="/pieces/white_king.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            wQ: () => (
              <img
                src="/pieces/white_queen.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            wR: () => (
              <img
                src="/pieces/white_rook.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            wB: () => (
              <img
                src="/pieces/white_bishop.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            wN: () => (
              <img
                src="/pieces/white_knight.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            wP: () => (
              <img
                src="/pieces/white_pawn.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bK: () => (
              <img
                src="/pieces/black_king.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bQ: () => (
              <img
                src="/pieces/black_queen.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bR: () => (
              <img
                src="/pieces/black_rook.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bB: () => (
              <img
                src="/pieces/black_bishop.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bN: () => (
              <img
                src="/pieces/black_knight.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
            bP: () => (
              <img
                src="/pieces/black_pawn.svg"
                style={{ width: "100%", height: "100%" }}
              />
            ),
          }}
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

        <MaterialDisplay
          diff={boardFlipped ? -materialAdv.diff : materialAdv.diff}
        />
        </div>

        <div className="controls-row">
          <button
            className="ctrl-btn"
            style={{ background: "#C9A96E" }}
            onClick={() => setBoardFlipped((f) => !f)}
          >
            ↺ Доска
          </button>
          <button
            className="ctrl-btn"
            style={{ background: "#C9A96E" }}
            onClick={handleNewGame}
          >
            Новая игра
          </button>
        </div>

        <FenBar onLoadFen={handleLoadFen} />
      </div>
    </div>
  );
});

export default ChessboardComponent;
