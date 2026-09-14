import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  checkTrainingTask,
  fetchTrainingLesson,
  fetchTrainingModule,
  fetchTrainingModules,
} from "../api.js";
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
    () => <img src={src} alt="" style={{ width: "100%", height: "100%" }} />,
  ]),
);

const MODULE_ICONS = {
  pawn: "♙",
  knight: "♘",
  bishop: "♗",
  rook: "♖",
  queen: "♕",
  king: "♔",
  "special-rules": "♙",
  "check-mate-stalemate": "♚",
};

const TASK_TYPE_LABELS = {
  select_squares: "Выбор клеток",
  make_move: "Ход на доске",
  choose_option: "Выбор ответа",
};

function getApiError(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback;
}

function getTaskHint(task) {
  if (!task) return "";

  const explicitHint = task.payload?.hint;
  if (typeof explicitHint === "string" && explicitHint.trim()) {
    return explicitHint.trim();
  }

  if (task.task_type === "select_squares") {
    const source = task.source_square ? ` с клетки ${task.source_square}` : "";
    return `Начните${source}: вспомните форму хода фигуры и учитывайте препятствия и занятые клетки.`;
  }

  if (task.task_type === "make_move") {
    const source = task.source_square
      ? ` Обратите внимание на фигуру на ${task.source_square}.`
      : "";
    return `Сначала проверьте самые естественные легальные ходы: шахи, взятия и прямые угрозы.${source}`;
  }

  if (task.task_type === "choose_option") {
    return "Сверьтесь с теорией слева и исключите варианты, которые нарушают разобранное в уроке правило.";
  }

  return "Вернитесь к теории урока и разбейте задачу на один простой шаг.";
}

export default function TrainingPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedModuleSlug = searchParams.get("module");

  const [modules, setModules] = useState([]);
  const [selectedModule, setSelectedModule] = useState(null);
  const [lessons, setLessons] = useState([]);
  const [lesson, setLesson] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [taskIndex, setTaskIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);
  const [selectedSquares, setSelectedSquares] = useState([]);
  const [selectedOption, setSelectedOption] = useState(null);
  const [moveSource, setMoveSource] = useState(null);
  const [taskResult, setTaskResult] = useState(null);
  const [boardFen, setBoardFen] = useState(null);
  const [boardWidth, setBoardWidth] = useState(500);
  const [hintsUsed, setHintsUsed] = useState(0);
  const [hintVisible, setHintVisible] = useState(false);
  const taskStartedAtRef = useRef(Date.now());
  const openingModuleRef = useRef(null);

  const currentTask = tasks[taskIndex] || null;

  useEffect(() => {
    let cancelled = false;

    async function loadModules() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchTrainingModules();
        if (!cancelled) setModules(data.modules || []);
      } catch (err) {
        if (!cancelled) {
          setError(getApiError(err, "Не удалось загрузить учебные модули."));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadModules();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const updateSize = () => {
      const availableWidth = Math.max(320, window.innerWidth - 720);
      const availableHeight = Math.max(320, window.innerHeight - 180);
      setBoardWidth(Math.max(320, Math.min(540, availableWidth, availableHeight)));
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    if (!currentTask) return;
    setSelectedSquares([]);
    setSelectedOption(null);
    setMoveSource(null);
    setTaskResult(null);
    setBoardFen(currentTask.fen);
    setHintsUsed(0);
    setHintVisible(false);
    taskStartedAtRef.current = Date.now();
  }, [currentTask?.id]);

  const openModule = useCallback(async (module) => {
    if (!module.enabled) return;

    try {
      setLoading(true);
      setError(null);
      const data = await fetchTrainingModule(module.slug);
      setSelectedModule(data.module);
      setLessons(data.lessons || []);
      setLesson(null);
      setTasks([]);
      setTaskIndex(0);
    } catch (err) {
      setError(getApiError(err, "Не удалось открыть учебный модуль."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!requestedModuleSlug || modules.length === 0) return;
    if (selectedModule?.slug === requestedModuleSlug) return;
    if (openingModuleRef.current === requestedModuleSlug) return;

    const module = modules.find(
      (item) => item.slug === requestedModuleSlug && item.enabled,
    );
    if (!module) return;

    openingModuleRef.current = requestedModuleSlug;
    openModule(module).finally(() => {
      openingModuleRef.current = null;
    });
  }, [modules, openModule, requestedModuleSlug, selectedModule?.slug]);

  const openLesson = useCallback(async (lessonId) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchTrainingLesson(lessonId);
      setLesson(data.lesson);
      setTasks(data.tasks || []);
      setTaskIndex(0);
    } catch (err) {
      setError(getApiError(err, "Не удалось открыть урок."));
    } finally {
      setLoading(false);
    }
  }, []);

  const backToModules = () => {
    navigate("/training");
    setSelectedModule(null);
    setLessons([]);
    setLesson(null);
    setTasks([]);
    setTaskIndex(0);
    setError(null);
  };

  const backToLessons = () => {
    setLesson(null);
    setTasks([]);
    setTaskIndex(0);
    setError(null);
  };

  const submitAnswer = useCallback(
    async (answer, moveToApply = null) => {
      if (!currentTask || checking) return null;

      try {
        setChecking(true);
        setError(null);
        const responseTimeMs = Math.max(0, Date.now() - taskStartedAtRef.current);
        const result = await checkTrainingTask(
          currentTask.id,
          answer,
          hintsUsed,
          responseTimeMs,
        );
        setTaskResult(result);
        window.dispatchEvent(new Event("sfedu-progress-updated"));

        if (result.correct && moveToApply && currentTask.task_type === "make_move") {
          try {
            const game = new Chess(currentTask.fen);
            const move = game.move({
              from: moveToApply.from,
              to: moveToApply.to,
              promotion: "q",
            });
            if (move) setBoardFen(game.fen());
          } catch {
            // Визуальное применение хода не влияет на серверную проверку.
          }
        }

        return result;
      } catch (err) {
        setError(getApiError(err, "Не удалось проверить ответ."));
        return null;
      } finally {
        setChecking(false);
      }
    },
    [checking, currentTask, hintsUsed],
  );

  const handleCheckSquares = () => {
    submitAnswer({ selected_squares: selectedSquares });
  };

  const handleCheckOption = () => {
    if (!selectedOption) return;
    submitAnswer({ option: selectedOption });
  };

  const handleShowTaskHint = () => {
    if (!currentTask || taskResult?.correct) return;
    if (!hintVisible) {
      setHintsUsed((count) => count + 1);
      setHintVisible(true);
    }
  };

  const openPuzzlesForCurrentTopic = () => {
    if (!selectedModule?.slug) return;
    navigate(`/puzzles?topic=${encodeURIComponent(selectedModule.slug)}`);
  };

  const submitMove = useCallback(
    (from, to) => {
      if (!currentTask || taskResult?.correct || checking) return;
      submitAnswer({ move: `${from}${to}` }, { from, to });
      setMoveSource(null);
    },
    [checking, currentTask, submitAnswer, taskResult?.correct],
  );

  const handlePieceDrop = useCallback(
    (sourceSquare, targetSquare) => {
      if (currentTask?.task_type !== "make_move") return false;
      submitMove(sourceSquare, targetSquare);
      // Возвращаем false: доску меняем только после подтверждения backend.
      return false;
    },
    [currentTask?.task_type, submitMove],
  );

  const handleSquareClick = useCallback(
    (square) => {
      if (!currentTask || taskResult?.correct || checking) return;

      if (currentTask.task_type === "select_squares") {
        if (square === currentTask.source_square) return;
        setSelectedSquares((previous) =>
          previous.includes(square)
            ? previous.filter((item) => item !== square)
            : [...previous, square],
        );
        return;
      }

      if (currentTask.task_type === "make_move") {
        if (!moveSource) {
          setMoveSource(square);
          return;
        }
        if (moveSource === square) {
          setMoveSource(null);
          return;
        }
        submitMove(moveSource, square);
      }
    },
    [checking, currentTask, moveSource, submitMove, taskResult?.correct],
  );

  const customSquareStyles = useMemo(() => {
    const styles = {};

    if (currentTask?.source_square) {
      styles[currentTask.source_square] = {
        boxShadow: "inset 0 0 0 4px rgba(201, 169, 110, 0.95)",
      };
    }

    for (const square of selectedSquares) {
      styles[square] = {
        backgroundColor: "rgba(79, 147, 92, 0.55)",
        boxShadow: "inset 0 0 0 3px rgba(47, 92, 54, 0.85)",
      };
    }

    if (moveSource) {
      styles[moveSource] = {
        backgroundColor: "rgba(201, 169, 110, 0.55)",
        boxShadow: "inset 0 0 0 4px rgba(180, 135, 44, 0.95)",
      };
    }

    return styles;
  }, [currentTask?.source_square, moveSource, selectedSquares]);

  const nextTask = () => {
    if (taskIndex < tasks.length - 1) {
      setTaskIndex((index) => index + 1);
    }
  };

  if (loading && modules.length === 0) {
    return (
      <div className="training-page training-page--centered">
        <div className="training-loading">Загрузка учебного курса...</div>
      </div>
    );
  }

  if (!selectedModule) {
    return (
      <div className="training-page">
        <section className="training-hero">
          <div className="training-kicker">Учебный режим</div>
          <h1>Обучение шахматам</h1>
          <p>
            Изучайте особенности каждой фигуры последовательно: сначала теория,
            затем интерактивные задания на шахматной доске.
          </p>
        </section>

        {error && <div className="training-global-error">{error}</div>}

        <div className="training-modules-grid">
          {modules.map((module) => (
            <button
              type="button"
              key={module.id}
              className={`training-module-card ${
                module.enabled ? "" : "training-module-card--disabled"
              }`}
              onClick={() =>
                navigate(`/training?module=${encodeURIComponent(module.slug)}`)
              }
              disabled={!module.enabled}
            >
              <span className="training-module-icon">
                {MODULE_ICONS[module.slug] || "♟"}
              </span>
              <span className="training-module-title">{module.title}</span>
              <span className="training-module-description">
                {module.description}
              </span>
              <span className="training-module-meta">
                {module.enabled
                  ? `${module.lesson_count} уроков · ${module.task_count} заданий`
                  : "Скоро"}
              </span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="training-page">
        <div className="training-toolbar">
          <button type="button" className="training-back-btn" onClick={backToModules}>
            ← Все модули
          </button>
          <button
            type="button"
            className="learning-crosslink"
            onClick={openPuzzlesForCurrentTopic}
          >
            Проверить на пазлах этой темы →
          </button>
        </div>

        <section className="training-module-header">
          <div className="training-module-header-icon">
            {MODULE_ICONS[selectedModule.slug] || "♟"}
          </div>
          <div>
            <div className="training-kicker">Учебный модуль</div>
            <h1>{selectedModule.title}</h1>
            <p>{selectedModule.description}</p>
          </div>
        </section>

        {error && <div className="training-global-error">{error}</div>}

        <div className="training-lessons-list">
          {lessons.map((item, index) => (
            <button
              type="button"
              className="training-lesson-card"
              key={item.id}
              onClick={() => openLesson(item.id)}
            >
              <span className="training-lesson-number">{index + 1}</span>
              <span className="training-lesson-content">
                <span className="training-lesson-title">{item.title}</span>
                <span className="training-lesson-meta">
                  {item.task_count} {item.task_count === 1 ? "задание" : "задания"}
                </span>
              </span>
              <span className="training-lesson-arrow">→</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="training-page training-page--lesson">
      <div className="training-toolbar">
        <button type="button" className="training-back-btn" onClick={backToLessons}>
          ← Уроки модуля
        </button>
        <button
          type="button"
          className="learning-crosslink"
          onClick={openPuzzlesForCurrentTopic}
        >
          Проверить на пазлах этой темы →
        </button>
        <div className="training-task-counter">
          Задание {Math.min(taskIndex + 1, tasks.length)} из {tasks.length}
        </div>
      </div>

      {error && <div className="training-global-error">{error}</div>}

      <div className="training-lesson-layout">
        <aside className="training-theory-panel">
          <div className="training-kicker">{selectedModule.title}</div>
          <h2>{lesson.title}</h2>
          <p>{lesson.theory}</p>

          <div className="training-lesson-progress">
            {tasks.map((task, index) => (
              <button
                type="button"
                key={task.id}
                className={`training-progress-dot ${
                  index === taskIndex ? "training-progress-dot--active" : ""
                }`}
                onClick={() => setTaskIndex(index)}
                title={`Задание ${index + 1}: ${task.title}`}
              >
                {index + 1}
              </button>
            ))}
          </div>
        </aside>

        <main className="training-board-column">
          {currentTask ? (
            <div className="training-board-wrapper">
              <ReactChessboard
                position={boardFen || currentTask.fen}
                onPieceDrop={handlePieceDrop}
                onSquareClick={handleSquareClick}
                boardWidth={boardWidth}
                animationDuration={180}
                boardOrientation="white"
                showBoardNotation={true}
                customPieces={customPieces}
                customSquareStyles={customSquareStyles}
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
              />
            </div>
          ) : (
            <div className="training-empty">В этом уроке пока нет заданий.</div>
          )}
        </main>

        <aside className="training-task-panel">
          {currentTask && (
            <>
              <div className="training-task-type">
                {TASK_TYPE_LABELS[currentTask.task_type] || "Учебное задание"}
              </div>
              <h2>{currentTask.title}</h2>
              <p className="training-task-instruction">{currentTask.instruction}</p>

              {currentTask.task_type === "select_squares" && (
                <div className="training-selected-summary">
                  <span>Выбрано клеток: {selectedSquares.length}</span>
                  {selectedSquares.length > 0 && (
                    <span className="training-selected-list">
                      {selectedSquares.slice().sort().join(", ")}
                    </span>
                  )}
                </div>
              )}

              {currentTask.task_type === "make_move" && (
                <div className="training-help-text">
                  Перетащите фигуру с выделенного поля или выберите начальную и
                  конечную клетки двумя кликами.
                </div>
              )}

              {currentTask.task_type === "choose_option" && (
                <div className="training-options">
                  {(currentTask.payload?.options || []).map((option) => (
                    <button
                      type="button"
                      key={option.id}
                      className={`training-option-btn ${
                        selectedOption === option.id
                          ? "training-option-btn--selected"
                          : ""
                      }`}
                      onClick={() => {
                        setSelectedOption(option.id);
                        if (!taskResult?.correct) setTaskResult(null);
                      }}
                      disabled={checking || taskResult?.correct}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              )}

              {currentTask.task_type === "select_squares" && !taskResult?.correct && (
                <div className="training-actions">
                  <button
                    type="button"
                    className="training-primary-btn"
                    onClick={handleCheckSquares}
                    disabled={checking || selectedSquares.length === 0}
                  >
                    {checking ? "Проверка..." : "Проверить"}
                  </button>
                  <button
                    type="button"
                    className="training-secondary-btn"
                    onClick={() => {
                      setSelectedSquares([]);
                      setTaskResult(null);
                    }}
                    disabled={checking || selectedSquares.length === 0}
                  >
                    Сбросить
                  </button>
                </div>
              )}

              {currentTask.task_type === "choose_option" && !taskResult?.correct && (
                <div className="training-actions">
                  <button
                    type="button"
                    className="training-primary-btn"
                    onClick={handleCheckOption}
                    disabled={checking || !selectedOption}
                  >
                    {checking ? "Проверка..." : "Проверить ответ"}
                  </button>
                </div>
              )}

              {!taskResult?.correct && (
                <>
                  <button
                    type="button"
                    className="training-secondary-btn"
                    onClick={handleShowTaskHint}
                    disabled={checking || hintVisible}
                  >
                    {hintVisible ? "Подсказка открыта" : "Подсказка"}
                  </button>
                  {hintVisible && (
                    <div className="training-hint-box">
                      {getTaskHint(currentTask)}
                      <span className="training-hint-meta">
                        Использовано подсказок: {hintsUsed}
                      </span>
                    </div>
                  )}
                </>
              )}

              {checking && currentTask.task_type === "make_move" && (
                <div className="training-checking">Проверяем ход...</div>
              )}

              {taskResult && (
                <div
                  className={`training-result ${
                    taskResult.correct
                      ? "training-result--success"
                      : "training-result--error"
                  }`}
                >
                  <strong>{taskResult.correct ? "Верно!" : "Попробуйте ещё раз"}</strong>
                  <span>{taskResult.feedback}</span>
                  <span className="training-result-score">
                    Результат: {Math.round((taskResult.score || 0) * 100)}%
                  </span>
                  {taskResult.correct && taskResult.explanation && (
                    <p>{taskResult.explanation}</p>
                  )}
                </div>
              )}

              {taskResult?.correct && taskIndex < tasks.length - 1 && (
                <button type="button" className="training-next-btn" onClick={nextTask}>
                  Следующее задание →
                </button>
              )}

              {taskResult?.correct && taskIndex === tasks.length - 1 && (
                <div className="training-lesson-complete">
                  <div className="training-complete-icon">✓</div>
                  <strong>Урок завершён</strong>
                  <span>Результат сохранён. Можно перейти к следующему уроку.</span>
                  <button type="button" className="training-next-btn" onClick={backToLessons}>
                    К списку уроков
                  </button>
                </div>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
