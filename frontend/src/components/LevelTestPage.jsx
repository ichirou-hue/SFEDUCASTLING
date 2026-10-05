import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import {
  checkLevelTestAnswer,
  finishLevelTest,
  skipLevelTestQuestion,
  startLevelTest,
} from "../api.js";

const START_FEN = "8/8/8/8/8/8/8/8 w - - 0 1";

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
    () => <img src={src} alt="" style={{ width: "100%", height: "100%" }} />,
  ]),
);

const LEVEL_DESCRIPTIONS = {
  1: "Осваиваете базовые тактические идеи и распознавание простых угроз.",
  2: "Уверенно решаете базовые задачи и можете переходить к более сложным комбинациям.",
  3: "Хорошо распознаёте тактические мотивы и находите сильные продолжения.",
  4: "Стабильно справляетесь со сложными тактическими позициями.",
};

function orientationFromFen(fen) {
  return fen?.split(" ")?.[1] === "b" ? "black" : "white";
}

function makeGame(fen) {
  try {
    return new Chess(fen);
  } catch {
    return new Chess();
  }
}

function isPromotionMove(game, sourceSquare, targetSquare) {
  const piece = game.get(sourceSquare);
  if (!piece || piece.type !== "p") return false;
  const rank = targetSquare?.[1];
  return (piece.color === "w" && rank === "8") || (piece.color === "b" && rank === "1");
}

function uciFromMove(move) {
  if (!move) return "";
  return `${move.from}${move.to}${move.promotion || ""}`;
}

export default function LevelTestPage({ onCompleted = null }) {
  const navigate = useNavigate();
  const [phase, setPhase] = useState("intro");
  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [fen, setFen] = useState(START_FEN);
  const [orientation, setOrientation] = useState("white");
  const [boardWidth, setBoardWidth] = useState(560);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalTargets, setLegalTargets] = useState([]);
  const [lastMove, setLastMove] = useState(null);
  const [questionState, setQuestionState] = useState("ready");
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [pendingPromotion, setPendingPromotion] = useState(null);
  const [testId, setTestId] = useState(null);
  const [personalization, setPersonalization] = useState(null);

  const gameRef = useRef(makeGame(START_FEN));
  const questionStartedAtRef = useRef(Date.now());

  const currentQuestion = questions[currentIndex] || null;
  const progress = questions.length
    ? Math.round(((currentIndex + (questionState === "answered" ? 1 : 0)) / questions.length) * 100)
    : 0;

  useEffect(() => {
    const updateSize = () => {
      const availableWidth = window.innerWidth - 390;
      const availableHeight = window.innerHeight - 235;
      const size = Math.floor(Math.min(availableWidth, availableHeight, 610));
      setBoardWidth(Math.max(300, size));
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const loadQuestion = useCallback((question) => {
    if (!question) return;
    const game = makeGame(question.fen);
    gameRef.current = game;
    setFen(game.fen());
    setOrientation(orientationFromFen(question.fen));
    setSelectedSquare(null);
    setLegalTargets([]);
    setLastMove(null);
    setQuestionState("ready");
    setFeedback(null);
    setPendingPromotion(null);
    questionStartedAtRef.current = Date.now();
  }, []);

  useEffect(() => {
    if (phase === "testing" && currentQuestion) {
      loadQuestion(currentQuestion);
    }
  }, [phase, currentIndex, currentQuestion, loadQuestion]);

  const beginTest = async () => {
    setError(null);
    setResult(null);
    setPhase("loading");
    try {
      const data = await startLevelTest();
      const loaded = data.questions || [];
      if (!loaded.length) {
        throw new Error(data.error || "Сервер не вернул задачи для теста");
      }
      const restoredAnswers = data.answers || [];
      const answeredIds = new Set(restoredAnswers.map((item) => String(item.puzzle_id)));
      let resumeIndex = loaded.findIndex((item) => !answeredIds.has(String(item.id)));
      if (resumeIndex < 0) resumeIndex = loaded.length - 1;

      setTestId(data.test_id);
      setPersonalization(data.personalization || null);
      setQuestions(loaded);
      setAnswers(restoredAnswers.map((item) => ({
        puzzle_id: item.puzzle_id,
        correct: Boolean(item.correct),
      })));
      setCurrentIndex(Math.max(0, resumeIndex));

      if (restoredAnswers.length >= loaded.length) {
        await finishTest(restoredAnswers, data.test_id);
        return;
      }
      setPhase("testing");
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось начать тест");
      setPhase("intro");
    }
  };

  const finishTest = async (finalAnswers, explicitTestId = null) => {
    const id = explicitTestId || testId;
    if (!id) {
      setError("Не найден идентификатор теста. Запустите тест заново.");
      return;
    }
    setPhase("finishing");
    setError(null);
    try {
      const data = await finishLevelTest(id, finalAnswers || []);
      if (data.error) throw new Error(data.error);
      window.dispatchEvent(new Event("sfedu-assessment-updated"));

      // В составе обязательного assessment результат пока не показываем:
      // сначала пользователь видит отдельный экран фидбэка, и только затем итог.
      if (onCompleted) {
        onCompleted(data);
        return;
      }

      // Standalone-режим LevelTestPage сохраняет старое поведение.
      setResult(data);
      setPhase("result");
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось рассчитать результат");
      setPhase("testing");
    }
  };

  const submitUci = async (uci, nextFen, from, to) => {
    if (!currentQuestion || questionState !== "ready") return false;
    setQuestionState("checking");
    setFeedback(null);
    setSelectedSquare(null);
    setLegalTargets([]);
    setLastMove({ from, to });
    setFen(nextFen);

    try {
      const checked = await checkLevelTestAnswer(
        testId,
        currentQuestion.id,
        uci,
        Math.max(0, Date.now() - questionStartedAtRef.current),
      );
      const strictCorrect = Boolean(checked.correct);
      const strongAlternative = !strictCorrect && Boolean(checked.strong_but_different);
      const answer = {
        puzzle_id: currentQuestion.id,
        correct: strictCorrect,
      };
      const nextAnswers = [...answers, answer];
      setAnswers(nextAnswers);
      setQuestionState("answered");

      if (strictCorrect) {
        setFeedback({ type: "success", text: "Верно. Ход совпал с решением задачи." });
      } else if (strongAlternative) {
        setFeedback({
          type: "strong",
          text: "Ход сильный, но для определения уровня засчитан только эталонный ответ.",
        });
      } else {
        setFeedback({ type: "error", text: "Ответ не засчитан. Переходим дальше без показа решения." });
      }
      return true;
    } catch (e) {
      setQuestionState("ready");
      loadQuestion(currentQuestion);
      setFeedback({
        type: "error",
        text: e?.response?.data?.detail || e?.message || "Не удалось проверить ход",
      });
      return false;
    }
  };

  const attemptMove = async (sourceSquare, targetSquare, promotion = null) => {
    if (questionState !== "ready" || !currentQuestion) return false;

    const baseGame = makeGame(currentQuestion.fen);
    if (isPromotionMove(baseGame, sourceSquare, targetSquare) && !promotion) {
      setPendingPromotion({ sourceSquare, targetSquare });
      return false;
    }

    let move = null;
    try {
      move = baseGame.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: promotion || "q",
      });
    } catch {
      move = null;
    }

    if (!move) {
      setFeedback({ type: "error", text: "Такой ход невозможен в этой позиции." });
      return false;
    }

    gameRef.current = baseGame;
    return submitUci(uciFromMove(move), baseGame.fen(), sourceSquare, targetSquare);
  };

  const onSquareClick = (square) => {
    if (questionState !== "ready") return;
    const game = gameRef.current;
    const piece = game.get(square);

    if (!selectedSquare) {
      if (piece && piece.color === game.turn()) {
        setSelectedSquare(square);
        const moves = game.moves({ square, verbose: true });
        setLegalTargets([...new Set(moves.map((move) => move.to))]);
      }
      return;
    }

    if (square === selectedSquare) {
      setSelectedSquare(null);
      setLegalTargets([]);
      return;
    }

    if (piece && piece.color === game.turn()) {
      setSelectedSquare(square);
      const moves = game.moves({ square, verbose: true });
      setLegalTargets([...new Set(moves.map((move) => move.to))]);
      return;
    }

    attemptMove(selectedSquare, square);
  };

  const handleNext = async () => {
    if (questionState !== "answered") return;
    if (currentIndex >= questions.length - 1) {
      await finishTest(answers);
      return;
    }
    setCurrentIndex((index) => index + 1);
  };

  const handleSkip = async () => {
    if (!currentQuestion || questionState !== "ready" || !testId) return;
    setQuestionState("checking");
    try {
      await skipLevelTestQuestion(
        testId,
        currentQuestion.id,
        Math.max(0, Date.now() - questionStartedAtRef.current),
      );
      const nextAnswers = [
        ...answers.filter((item) => item.puzzle_id !== currentQuestion.id),
        { puzzle_id: currentQuestion.id, correct: false },
      ];
      setAnswers(nextAnswers);

      if (currentIndex >= questions.length - 1) {
        await finishTest(nextAnswers);
        return;
      }
      setCurrentIndex((index) => index + 1);
    } catch (e) {
      setQuestionState("ready");
      setFeedback({
        type: "error",
        text: e?.response?.data?.detail || e?.message || "Не удалось сохранить пропуск",
      });
    }
  };

  const squareStyles = {};
  if (selectedSquare) {
    squareStyles[selectedSquare] = { backgroundColor: "rgba(201, 169, 110, 0.58)" };
  }
  legalTargets.forEach((square) => {
    const targetPiece = gameRef.current.get(square);
    squareStyles[square] = targetPiece
      ? {
          backgroundImage:
            "radial-gradient(circle, transparent 48%, rgba(64,97,53,0.35) 50%, rgba(64,97,53,0.35) 80%, transparent 82%)",
        }
      : {
          backgroundImage:
            "radial-gradient(circle, rgba(64,97,53,0.34) 15%, transparent 16%)",
        };
  });
  if (lastMove) {
    squareStyles[lastMove.from] = { backgroundColor: "rgba(201, 169, 110, 0.42)" };
    squareStyles[lastMove.to] = { backgroundColor: "rgba(201, 169, 110, 0.64)" };
  }

  if (phase === "intro" || phase === "loading") {
    return (
      <main className="level-test-page level-test-page--centered">
        <section className="level-test-intro-card">
          <div className="level-test-kicker">{onCompleted ? "Шаг 2 из 3 · персональный тест" : "SFEDUCASTLING"}</div>
          <h1>Проверка шахматного уровня</h1>
          <p>
            Тест состоит из 20 персонально подобранных тактических позиций.
            Большая часть соответствует вашей предварительной оценке, а соседние уровни проверяют границы.
          </p>
          <div className="level-test-rules">
            <div><strong>20</strong><span>задач</span></div>
            <div><strong>1</strong><span>ход на позицию</span></div>
            <div><strong>4</strong><span>рейтинговых диапазона</span></div>
          </div>
          <p className="level-test-note">
            Во время теста рейтинг позиции, подсказки и решение скрыты, чтобы результат отражал ваш текущий уровень.
          </p>
          {personalization?.rating_group && (
            <div className="level-test-personalization">
              Стартовый диапазон: <strong>{personalization.rating_group}</strong>
              {personalization.weak_topic_titles?.length > 0 && (
                <span> · учтены слабые темы: {personalization.weak_topic_titles.join(", ")}</span>
              )}
            </div>
          )}
          {error && <div className="level-test-error">{error}</div>}
          <button className="level-test-primary-btn" onClick={beginTest} disabled={phase === "loading"}>
            {phase === "loading" ? "Формируем новый набор..." : "Начать проверку уровня"}
          </button>
        </section>
      </main>
    );
  }

  if (phase === "finishing") {
    return (
      <main className="level-test-page level-test-page--centered">
        <div className="level-test-loading-card">Рассчитываем ваш уровень...</div>
      </main>
    );
  }

  if (phase === "result" && result) {
    const level = result.result?.level ?? result.level ?? 1;
    const name = result.result?.name || "Новичок";
    const breakdown = Object.entries(result.score?.by_group || {})
      .map(([key, value]) => ({ level: Number(key), ...value }))
      .sort((a, b) => a.level - b.level);
    const totalCorrect = result.score?.correct ?? breakdown.reduce((sum, item) => sum + (item.correct || 0), 0);
    const totalQuestions = result.score?.total ?? breakdown.reduce((sum, item) => sum + (item.total || 0), 0);
    const finalRating = result.rating ?? result.score?.final_rating ?? result.result?.rating;

    return (
      <main className="level-test-page level-test-page--centered">
        <section className="level-test-result-card">
          <div className="level-test-result-icon">♕</div>
          <div className="level-test-kicker">Результат проверки</div>
          <h1>{name}</h1>
          <div className="level-test-level-badge">Уровень {level} из 4</div>
          {finalRating != null && (
            <div className="level-test-rating-value">Итоговый рейтинг: <strong>{finalRating}</strong></div>
          )}
          <p>{LEVEL_DESCRIPTIONS[level]}</p>
          <div className="level-test-total-score">
            <strong>{totalCorrect}</strong>
            <span>из {totalQuestions} задач решено точно</span>
          </div>

          <div className="level-test-breakdown">
            {breakdown.map((item) => (
              <div className="level-test-breakdown-row" key={item.level}>
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.range || `Диапазон ${item.level}`}</span>
                </div>
                <div className="level-test-breakdown-score">
                  {item.correct} / {item.total}
                </div>
              </div>
            ))}
          </div>

          {result.feedback?.length > 0 && (
            <div className="level-test-result-feedback">
              {result.feedback.map((item) => <div key={item}>• {item}</div>)}
            </div>
          )}
          <div className="level-test-result-actions">
            <button className="level-test-primary-btn" onClick={() => navigate("/training")}>
              Перейти к обучению
            </button>
            <button className="level-test-secondary-btn" onClick={() => navigate("/")}>
              На главную
            </button>
          </div>
        </section>
      </main>
    );
  }

  if (!currentQuestion) {
    return (
      <main className="level-test-page level-test-page--centered">
        <div className="level-test-error">Не удалось загрузить текущую задачу.</div>
      </main>
    );
  }

  return (
    <main className="level-test-page">
      <section className="level-test-shell">
        <header className="level-test-header">
          <div>
            <div className="level-test-kicker">Проверка уровня</div>
            <h1>Найдите лучший ход</h1>
          </div>
          <div className="level-test-counter">{currentIndex + 1} / {questions.length}</div>
        </header>

        <div className="level-test-progress-track">
          <div className="level-test-progress-fill" style={{ width: `${progress}%` }} />
        </div>

        <div className="level-test-content">
          <div className="level-test-board-column">
            <div className="level-test-board-wrap">
              <ReactChessboard
                position={fen}
                boardOrientation={orientation}
                boardWidth={boardWidth}
                onPieceDrop={(source, target) => {
                  attemptMove(source, target);
                  return false;
                }}
                onSquareClick={onSquareClick}
                arePiecesDraggable={questionState === "ready"}
                animationDuration={180}
                customPieces={customPieces}
                customSquareStyles={squareStyles}
                customBoardStyle={{ borderRadius: "14px" }}
                customDarkSquareStyle={{ backgroundColor: "#6f9163" }}
                customLightSquareStyle={{ backgroundColor: "#e8e0c8" }}
              />
            </div>
          </div>

          <aside className="level-test-side-card">
            <div className="level-test-side-number">Задача {currentIndex + 1}</div>
            <h2>Ваш ход</h2>
            <p>
              Сделайте один лучший ход за сторону, которой принадлежит очередь хода. Можно перетащить фигуру или выбрать клетки кликом.
            </p>

            <div className="level-test-turn">
              Ходят: <strong>{orientation === "white" ? "белые" : "чёрные"}</strong>
            </div>

            {questionState === "checking" && (
              <div className="level-test-feedback level-test-feedback--checking">Проверяем ход...</div>
            )}

            {feedback && (
              <div className={`level-test-feedback level-test-feedback--${feedback.type}`}>
                {feedback.text}
              </div>
            )}

            {questionState === "answered" ? (
              <button className="level-test-primary-btn" onClick={handleNext}>
                {currentIndex >= questions.length - 1 ? "Узнать результат" : "Следующая задача"}
              </button>
            ) : (
              <button
                className="level-test-skip-btn"
                onClick={handleSkip}
                disabled={questionState !== "ready"}
              >
                Пропустить задачу
              </button>
            )}

            <div className="level-test-privacy-note">
              Рейтинг и тема задачи намеренно скрыты до завершения теста.
            </div>
          </aside>
        </div>
      </section>

      {pendingPromotion && (
        <div className="level-test-modal-backdrop">
          <div className="level-test-promotion-modal">
            <h3>Выберите фигуру для превращения</h3>
            <div className="level-test-promotion-options">
              {["q", "r", "b", "n"].map((piece) => (
                <button
                  key={piece}
                  onClick={() => {
                    const pending = pendingPromotion;
                    setPendingPromotion(null);
                    attemptMove(pending.sourceSquare, pending.targetSquare, piece);
                  }}
                >
                  {{ q: "♕", r: "♖", b: "♗", n: "♘" }[piece]}
                </button>
              ))}
            </div>
            <button className="level-test-modal-cancel" onClick={() => setPendingPromotion(null)}>
              Отмена
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
