import { useState, useEffect, useRef, useCallback } from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import { fetchPuzzles, askLLM } from "../api.js";
import MiniBoard from "./MiniBoard.jsx";

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

const THEME_LABELS = {
  mateIn1: "Мат в 1 ход",
  mateIn2: "Мат в 2 хода",
  mateIn3: "Мат в 3 хода",
  mateIn4: "Мат в 4 хода",
  fork: "Вилка",
  pin: "Связка",
  skewer: "Кол",
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
  endgame: "♟", middlegame: "♝", opening: "♞",
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
      label: THEME_LABELS[pick] || pick,
      icon: THEME_ICONS[pick] || "♟",
    };
  }
  return { label: "Тактика", icon: "♟" };
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

export default function PuzzlesPage() {
  const [puzzles, setPuzzles] = useState([]);
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
  const [hintArrow, setHintArrow] = useState(null);
  const [boardArrows, setBoardArrows] = useState([]);
  const [boardOrientation, setBoardOrientation] = useState("white");
  const [puzzleSolved, setPuzzleSolved] = useState(false);

  const [chatMessages, setChatMessages] = useState([
    {
      role: "ai",
      text: "Задайте вопрос по текущей позиции, и я помогу разобраться!",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  const gameRef = useRef(new Chess());
  const solutionMovesRef = useRef([]);
  const solutionIndexRef = useRef(0);
  const puzzlesRef = useRef([]);
  const loadSeqRef = useRef(0);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  useEffect(() => {
    const updateSize = () => {
      const availW = window.innerWidth - 260 - 360 - 48 - 24;
      const availH = window.innerHeight * 0.85 - 100;
      const size = Math.floor(Math.min(Math.max(availW, 280), availH));
      setBoardWidth(Math.max(320, Math.min(size, 620)));
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

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
    fetchPuzzles(20)
      .then((data) => {
        const qs = data.puzzles || [];
        if (qs.length === 0) {
          setError("Нет доступных задач");
          return;
        }
        puzzlesRef.current = qs;
        setPuzzles(qs);
        setCurrentIndex(0);
        setLoading(false);
      })
      .catch((e) => {
        setError("Ошибка загрузки задач: " + e.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (puzzles.length > 0 && puzzles[currentIndex]) {
      loadPuzzle(currentIndex);
    }
  }, [currentIndex, puzzles, loadPuzzle]);

  const currentPuzzle = puzzles[currentIndex];
  const themeInfo = currentPuzzle
    ? getThemeInfo(currentPuzzle)
    : { label: "Тактика", icon: "♟" };

  const handleMove = useCallback(
    (sourceSquare, targetSquare) => {
      if (waitingForOpponent || showSolution) return false;

      const expectedMove = solutionMovesRef.current[solutionIndexRef.current];
      if (!expectedMove) return false;

      const expectedCore = expectedMove.substring(0, 4);
      const moveCore = sourceSquare + targetSquare;

      if (moveCore === expectedCore) {
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
          setSolvedCount((c) => c + 1);
          setPuzzleSolved(true);
          return true;
        }

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

      setResultMessage({
        type: "error",
        text: "Неверный ход. Попробуйте ещё раз.",
      });
      return false;
    },
    [waitingForOpponent, showSolution],
  );

  const onSquareClick = useCallback(
    (square) => {
      if (waitingForOpponent || showSolution) return;

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
    [selectedSquare, waitingForOpponent, showSolution, handleMove],
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

  if (selectedSquare) {
    customSquareStyles[selectedSquare] = {
      backgroundColor: "rgba(201, 169, 110, 0.55)",
    };
  }

  for (const sq of legalMovesForSelected) {
    const targetPiece = gameRef.current.get(sq);
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

  if (lastMove) {
    customSquareStyles[lastMove.from] = {
      backgroundColor: "rgba(201, 169, 110, 0.45)",
    };
    customSquareStyles[lastMove.to] = {
      backgroundColor: "rgba(201, 169, 110, 0.6)",
    };
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

  return (
    <div className="puzzles-layout">
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
        <div className={`puzzle-board-wrapper ${puzzleSolved ? "puzzle-board-wrapper--solved" : ""}`}>
          <ReactChessboard
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
              color: "#225a73",
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

      <div className="puzzle-chat-section">
        <div className="chat-header">
          <div className="chat-avatar">
            <img src="/bot-icon.svg" alt="Ассистент" />
          </div>
          <div className="chat-title-block">
            <h2>ЧАТ ПО ЗАДАЧЕ</h2>
            <span className="chat-subtitle">Шахматный ассистент</span>
          </div>
        </div>

        <div className="chat-messages">
          {chatMessages.map((msg, i) => (
            <div
              className={`chat-msg ${msg.role === "user" ? "user" : ""}`}
              key={i}
            >
              <div className="msg-avatar">
                {msg.role === "user" ? "👤" : "🤖"}
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
              <div className="msg-avatar">🤖</div>
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
    </div>
  );
}
