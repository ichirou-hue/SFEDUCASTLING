import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Chessboard as ReactChessboard } from "react-chessboard";
import { Chess } from "chess.js";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  checkTrainingTask,
  fetchTrainingDue,
  fetchTrainingLesson,
  fetchTrainingModule,
  fetchTrainingModules,
  fetchTrainingProgress,
} from "../api.js";
import SiteFooter from "./SiteFooter.jsx";
import RatingBandProgress from "./RatingBandProgress.jsx";
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

// Пауза перед автопереключением на следующее задание после верного
// ответа (мс): достаточно, чтобы прочитать разбор, но не скучно ждать.
const AUTO_ADVANCE_DELAY_MS = 2500;

const MODULE_ICONS = {
  pawn: "♙",
  knight: "♘",
  bishop: "♗",
  rook: "♖",
  queen: "♕",
  king: "♔",
  "special-rules": "⚙",
  "check-mate-stalemate": "♚",
  pieces: "♟",
  "king-safety": "♚",
  "endgames": "♔",
  "tactics": "♞",
  "opening-principles": "♜",
  openings: "♘",
};

const TASK_TYPE_LABELS = {
  select_squares: "Выбор клеток",
  make_move: "Ход на доске",
  choose_option: "Выбор ответа",
};

function getApiError(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback;
}

function formatTaskCount(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod100 >= 11 && mod100 <= 14) return `${count} заданий`;
  if (mod10 === 1) return `${count} задание`;
  if (mod10 >= 2 && mod10 <= 4) return `${count} задания`;
  return `${count} заданий`;
}

/*
 * Тонкий прогресс-бар. Повторяет визуал .progress-meter из ProgressWidget,
 * чтобы процент прохождения выглядел одинаково в профиле и в обучении.
 */
function ProgressBar({ value, label }) {
  const safe = Math.min(100, Math.max(0, Number(value) || 0));

  return (
    <div
      className="training-progress-bar"
      role="progressbar"
      aria-valuenow={Math.round(safe)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      title={label}
    >
      <span style={{ width: `${safe}%` }} />
    </div>
  );
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

export default function TrainingPage({ user, onRegister }) {
  // Свой id доски - иначе доски делят один dnd-менеджер и мини-доски
  // рисуют у курсора чужую перетаскиваемую фигуру (см. Chessboard.jsx).
  const boardId = useId();

  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedModuleSlug = searchParams.get("module");
  const showBasicCourse = searchParams.get("basics") === "1";

  const [modules, setModules] = useState([]);
  const [dueItems, setDueItems] = useState([]);
  const [duePanelOpen, setDuePanelOpen] = useState(false);
  // Прогресс курса: modules.items[] несёт счётчики заданий по каждому
  // модулю и его урокам. Грузится отдельно от списка модулей, потому что
  // требует JWT и считается по training_attempts.
  const [progress, setProgress] = useState(null);
  const [selectedModule, setSelectedModule] = useState(null);
  const [lessons, setLessons] = useState([]);
  const [lesson, setLesson] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [taskIndex, setTaskIndex] = useState(0);
  // ID заданий текущего урока, решённые верно. Серверные данные из
  // lessonProgressById.solved_task_ids плюс то, что решено в этой сессии —
  // так дотсы не теряют состояние при переходах внутри урока.
  const [solvedTaskIds, setSolvedTaskIds] = useState(() => new Set());
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
  // Слаг модуля, который реально открыт сейчас. Нужен против гонки в
  // backToModules: navigate("/training") + setSelectedModule(null) могут
  // отрендериться раньше, чем обновится searchParams, и эффект автооткрытия
  // видит stale-slug при пустом selectedModule — и тут же открывает модуль
  // заново (первый клик «съедается», срабатывает только второй).
  // С этим ref stale-рендер пропускается: слаг равен отображаемому.
  const displayedSlugRef = useRef(null);

  // Память скролла: три ветки рендера (grid/lessons/lesson) — три разных
  // DOM-узла, при переходе контейнер перемонтируется и scrollTop слетает
  // в 0. Запоминаем позицию перед уходом, восстанавливаем при монтировании.
  const pageRef = useRef(null);
  const branchScrollRef = useRef({ grid: 0, lessons: 0, lesson: 0 });

  const saveBranchScroll = (branch) => {
    const el = pageRef.current;
    if (el) branchScrollRef.current[branch] = el.scrollTop;
  };

  const currentTask = tasks[taskIndex] || null;
  // Следим за конкретным пользователем, а не за объектом целиком:
  // обновление профиля (идентичный id) не должно перезагружать курс.
  const userId = user?.id ?? null;

  // Быстрый доступ к прогрессу: по слагу модуля и по id урока.
  const moduleProgressBySlug = useMemo(() => {
    const map = new Map();
    for (const item of progress?.modules?.items || []) {
      map.set(item.slug, item);
    }
    return map;
  }, [progress]);

  // Конь-маршрутчик на тропинке модулей: сидит над первым непройденным
  // модулем (изначально — над первым). Когда модуль закрывается, конь
  // перепрыгивает на следующий. Всё пройдено — отдыхает над последним.
  const currentModuleIndex = useMemo(() => {
    if (modules.length === 0) return -1;
    const index = modules.findIndex((module) => {
      if (!module.enabled) return false;
      const moduleProgress = moduleProgressBySlug.get(module.slug);
      if (!moduleProgress || !moduleProgress.tasks_total) return true;
      return moduleProgress.tasks_solved < moduleProgress.tasks_total;
    });
    return index === -1 ? modules.length - 1 : index;
  }, [modules, moduleProgressBySlug]);

  const modulesPathRef = useRef(null);
  const moduleNodeRefs = useRef(new Map());
  const [knightPos, setKnightPos] = useState(null);
  const [knightHops, setKnightHops] = useState(0);
  const [knightInstant, setKnightInstant] = useState(false);
  const prevKnightIndexRef = useRef(null);
  // Над каким модулем конь остался, когда ушли с сетки (открыли урок).
  // Нужно, чтобы по возвращении проиграть пропущенный прыжок.
  const gridLeaveIndexRef = useRef(null);
  const currentModuleIndexRef = useRef(currentModuleIndex);

  useEffect(() => {
    currentModuleIndexRef.current = currentModuleIndex;
  });

  const measureKnightNode = useCallback(
    (index) => {
      const module = modules[index];
      const node = module ? moduleNodeRefs.current.get(module.id) : null;
      if (!node) return null;
      const size = 88;
      return {
        top: node.offsetTop - size - 10,
        left: node.offsetLeft + node.offsetWidth / 2 - size / 2,
      };
    },
    [modules],
  );

  const placeKnight = useCallback(() => {
    setKnightPos(measureKnightNode(currentModuleIndex));
  }, [measureKnightNode, currentModuleIndex]);

  useEffect(() => {
    if (
      prevKnightIndexRef.current !== null &&
      prevKnightIndexRef.current !== currentModuleIndex
    ) {
      // Индекс сменился — конь прыгает (перезапуск анимации через key).
      setKnightHops((hops) => hops + 1);
    }
    prevKnightIndexRef.current = currentModuleIndex;
    placeKnight();
    // Шрифты могут чуть сдвинуть раскладку после замера — перемерить.
    const timer = setTimeout(placeKnight, 500);
    return () => clearTimeout(timer);
  }, [currentModuleIndex, placeKnight]);

  useEffect(() => {
    window.addEventListener("resize", placeKnight);
    return () => window.removeEventListener("resize", placeKnight);
  }, [placeKnight]);

  const lessonProgressById = useMemo(() => {
    const map = new Map();
    for (const item of progress?.modules?.items || []) {
      for (const lessonItem of item.lessons || []) {
        map.set(lessonItem.id, lessonItem);
      }
    }
    return map;
  }, [progress]);

  // Дуолинго-раскладка: уроки группируются в разделы (section_title),
  // внутри — узлы тропинки со сквозной нумерацией. Состояние узла:
  // done — все задания решены, current — первый нерешённый, todo — остальные.
  // Запирания нет: все узлы открыты, цвет лишь подсказывает порядок.
  const lessonSections = useMemo(() => {
    let foundCurrent = false;
    let globalIndex = 0;
    const sections = [];
    for (const item of lessons) {
      const title = item.section_title || null;
      let section = sections[sections.length - 1];
      if (!section || section.title !== title) {
        section = {
          key: `${title ?? "plain"}-${sections.length}`,
          title,
          lessons: [],
          done: 0,
        };
        sections.push(section);
      }
      const lessonProgress = lessonProgressById.get(item.id);
      const done =
        !!lessonProgress &&
        lessonProgress.tasks_total > 0 &&
        lessonProgress.tasks_solved >= lessonProgress.tasks_total;
      let state = "todo";
      if (done) {
        state = "done";
        section.done += 1;
      } else if (!foundCurrent) {
        state = "current";
        foundCurrent = true;
      }
      section.lessons.push({
        item,
        state,
        progress: lessonProgress,
        globalIndex: globalIndex++,
      });
    }
    return sections;
  }, [lessons, lessonProgressById]);

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
          setModules([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    async function loadDue() {
      try {
        const data = await fetchTrainingDue(20);
        if (!cancelled) setDueItems(data.items || []);
      } catch {
        // Просроченные повторения — вспомогательная секция:
        // её сбой не должен ломать загрузку курса.
        if (!cancelled) setDueItems([]);
      }
    }

    async function loadProgress() {
      try {
        const data = await fetchTrainingProgress();
        if (!cancelled) setProgress(data || null);
      } catch {
        // Без прогресса курс всё равно читается — просто не показываем бары.
        if (!cancelled) setProgress(null);
      }
    }

    if (!userId) {
      // Гость: вместо списка показывается заглушка «после регистрации».
      // Сбрасываем состояние при выходе/переключении аккаунта, иначе после
      // входа останутся старые данные (пустой список или гостевой 401).
      setModules([]);
      setDueItems([]);
      setProgress(null);
      setSelectedModule(null);
      setLessons([]);
      setLesson(null);
      setTasks([]);
      setTaskIndex(0);
      setError(null);
      setLoading(false);
      return;
    }

    loadModules();
    loadDue();
    loadProgress();

    // После каждого верного ответа страница сама шлёт это событие —
    // перечитываем прогресс, чтобы бары обновились без перезахода.
    const refreshProgress = () => loadProgress();
    window.addEventListener("sfedu-progress-updated", refreshProgress);

    return () => {
      cancelled = true;
      window.removeEventListener("sfedu-progress-updated", refreshProgress);
    };
  }, [userId]);

  useEffect(() => {
    const updateSize = () => {
      const isMobile = window.matchMedia("(max-width: 760px)").matches;
      const boardColumn = document.querySelector(".training-board-column");
      const availableWidth = isMobile
        ? (boardColumn?.clientWidth || window.innerWidth - 24) - 30
        : window.innerWidth - 720;
      const availableHeight = Math.max(320, window.innerHeight - 180);
      setBoardWidth(Math.max(isMobile ? 240 : 320, Math.min(540, availableWidth, availableHeight)));
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [selectedModule?.slug, lesson?.id]);

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

  // Автопереключение после верного ответа: даём время прочитать разбор
  // и переходим к следующему заданию. На последнем задании таймер не
  // ставим — показывается блок «Урок завершён». Ручной переход (кнопка
  // «Следующее задание», дотсы) меняет taskIndex → таймер отменяется,
  // а новый урок сбрасывает taskResult в эффекте выше.
  useEffect(() => {
    if (!taskResult?.correct) return undefined;
    if (taskIndex >= tasks.length - 1) return undefined;

    const timer = setTimeout(() => {
      setTaskIndex((index) => index + 1);
    }, AUTO_ADVANCE_DELAY_MS);

    return () => clearTimeout(timer);
  }, [taskResult?.correct, taskIndex, tasks.length]);

  const openModule = useCallback(async (module) => {
    if (!module.enabled) return;

    // Единая воронка открытия (эффект по URL, openDueItem): фиксируем,
    // что показываем именно этот модуль, до асинхронной загрузки.
    displayedSlugRef.current = module.slug;
    // Уходим с сетки — запоминаем её скролл для возврата.
    saveBranchScroll("grid");

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
    if (requestedModuleSlug === displayedSlugRef.current) return;
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

  // Стрелка браузера назад: URL вернулся на /training без ?module=,
  // а состояние ещё держит открытый модуль — сбрасываем как backToModules.
  // Заодно сбрасываем displayedSlugRef: раз слага нет в адресе, повторный
  // заход на тот же модуль (вперёд/карточка) должен открывать его заново.
  useEffect(() => {
    if (requestedModuleSlug) return;
    displayedSlugRef.current = null;
    if (!selectedModule) return;
    setSelectedModule(null);
    setLessons([]);
    setLesson(null);
    setTasks([]);
    setTaskIndex(0);
    setError(null);
  }, [requestedModuleSlug, selectedModule]);

  const openLesson = useCallback(async (lessonId, opts = {}) => {
    // По умолчанию уходим со списка уроков: запоминаем его скролл.
    // openDueItem идёт с сетки (контейнер ещё старый) — ему сохранять
    // "lessons" нельзя, иначе туда запишется чужая позиция.
    if (opts.save !== false) saveBranchScroll("lessons");
    // Новый урок всегда открывается сверху.
    branchScrollRef.current.lesson = 0;
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

  // Заполняем решённые задания урока из серверного прогресса.
  // Зависим только от id урока: перезагрузка прогресса после ответа не
  // должна сбрасывать локально добавленные решения.
  useEffect(() => {
    if (!lesson) {
      setSolvedTaskIds(new Set());
      return;
    }
    const lessonProgress = lessonProgressById.get(lesson.id);
    setSolvedTaskIds(new Set(lessonProgress?.solved_task_ids || []));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lesson?.id]);

  // Повторение сразу открывает нужное задание. Сохраняем слаг в URL:
  // иначе эффект синхронизации состояния с query-параметром сбрасывает модуль.
  const openDueItem = useCallback(
    async (item) => {
      const moduleSlug = item?.module?.slug;
      const lessonId = item?.lesson?.id;
      const taskId = item?.task?.id;

      if (!moduleSlug || lessonId == null || taskId == null) {
        setError("Не удалось открыть повторение: данные задания неполные.");
        return;
      }

      displayedSlugRef.current = moduleSlug;
      setDuePanelOpen(false);
      navigate(`/training?module=${encodeURIComponent(moduleSlug)}`);
      saveBranchScroll("grid");
      branchScrollRef.current.lesson = 0;

      try {
        setLoading(true);
        setError(null);
        const [moduleData, lessonData] = await Promise.all([
          fetchTrainingModule(moduleSlug),
          fetchTrainingLesson(lessonId),
        ]);
        const module = moduleData.module;
        const lessonTasks = lessonData.tasks || [];
        const dueTaskIndex = lessonTasks.findIndex(
          (task) => String(task.id) === String(taskId),
        );

        if (!module?.enabled) {
          throw new Error("Модуль для этого повторения больше недоступен.");
        }
        if (dueTaskIndex < 0) {
          throw new Error("Задание для повторения больше недоступно в этом уроке.");
        }

        setSelectedModule(module);
        setLessons(moduleData.lessons || []);
        setLesson(lessonData.lesson);
        setTasks(lessonTasks);
        setTaskIndex(dueTaskIndex);
      } catch (err) {
        setError(getApiError(err, "Не удалось открыть задание для повторения."));
      } finally {
        setLoading(false);
      }
    },
    [navigate],
  );

  const backToModules = () => {
    // replace: это действие «назад», а не новый шаг — иначе в истории
    // останется дубль ?module=x, и стрелка браузера будет открывать
    // модуль заново вместо возврата к списку.
    navigate("/training", { replace: true });
    saveBranchScroll("lessons");
    setSelectedModule(null);
    setLessons([]);
    setLesson(null);
    setTasks([]);
    setTaskIndex(0);
    setError(null);
  };

  const backToLessons = () => {
    saveBranchScroll("lesson");
    setLesson(null);
    setTasks([]);
    setTaskIndex(0);
    setError(null);
  };

  const currentLessonIndex = lessons.findIndex((item) => item.id === lesson?.id);
  const navigateLesson = (delta) => {
    const target = lessons[currentLessonIndex + delta];
    if (target) openLesson(target.id);
  };

  // Какая ветка смонтирована: сетка модулей, тропинка уроков или урок.
  // Зависимости loading/lessons/tasks — контент подтягивается асинхронно:
  // восстанавливаем позицию, когда всё уже на месте. Обновления прогресса
  // эти зависимости не трогают, поэтому скролл посреди урока не дёргается.
  const branch = !selectedModule ? "grid" : !lesson ? "lessons" : "lesson";

  useEffect(() => {
    if (loading) return;
    const el = pageRef.current;
    if (el) el.scrollTop = branchScrollRef.current[branch] || 0;
  }, [branch, loading, lessons.length, tasks.length]);

  // Возврат на сетку модулей: если конь «ушёл» с одного модуля, а вернулся
  // уже при другом текущем (модуль закрыли в уроке, сетка была размонтирована
  // и прыжок увидеть было негде) — проигрываем пропущенное: мгновенно
  // встаём на старый узел, затем планирование + прыг на новый.
  useEffect(() => {
    if (branch !== "grid") return;
    const from = gridLeaveIndexRef.current;
    const to = currentModuleIndexRef.current;
    if (from === null || from === to || from < 0 || to < 0) {
      setKnightInstant(false);
      placeKnight();
      return;
    }
    gridLeaveIndexRef.current = to;
    const startPos = measureKnightNode(from);
    if (!startPos) {
      setKnightInstant(false);
      placeKnight();
      return;
    }
    setKnightInstant(true);
    setKnightPos(startPos);
    const frame = requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        setKnightInstant(false);
        const endPos = measureKnightNode(to);
        if (endPos) setKnightPos(endPos);
        setKnightHops((hops) => hops + 1);
      }),
    );
    return () => cancelAnimationFrame(frame);
    // measure/place намеренно вне зависимостей: повторный прогон только
    // при смене ветки/состава модулей, иначе задвоится с живым эффектом.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [branch, modules.length]);

  // Уход с сетки: запоминаем индекс для будущего replay (см. выше).
  // Читаем живой ref в cleanup — там всегда актуальное значение.
  useEffect(() => {
    if (branch !== "grid") return;
    return () => {
      gridLeaveIndexRef.current = currentModuleIndexRef.current;
    };
  }, [branch]);

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
        if (result.correct) {
          // Дотсы урока обновляются сразу; после ответа страница шлёт
          // sfedu-progress-updated, так что бары модулей перечитаются сами.
          setSolvedTaskIds((prev) => {
            const next = new Set(prev);
            next.add(currentTask.id);
            return next;
          });
        }
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

  if (loading && modules.length === 0 && user) {
    return (
      <div className="training-page training-page--centered" ref={pageRef}>
        <div className="training-loading">Загрузка учебного курса...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="training-page training-page--centered" ref={pageRef}>
        <div className="training-center-wrap">
          <div className="training-register-wall">
            <div className="training-kicker">Учебный режим</div>
            <h1>Обучение доступно после регистрации</h1>
            <p>
              Зарегистрируйтесь, чтобы проходить учебные модули по шахматам
              и сохранять свой прогресс.
            </p>
            <button
              type="button"
              className="training-primary-btn"
              onClick={onRegister}
            >
              Зарегистрироваться
            </button>
          </div>
        </div>
        <SiteFooter />
      </div>
    );
  }

  if (
    !selectedModule &&
    !requestedModuleSlug &&
    !showBasicCourse &&
    Number(user.elo) >= 2000
  ) {
    return (
      <div className="training-page training-page--centered">
        <section className="training-register-wall">
          <div className="training-kicker">Продвинутый уровень</div>
          <RatingBandProgress userId={userId} />
          <h1>Практика для вашего уровня</h1>
          <p>
            Базовый курс объясняет правила и ходы фигур. Для рейтинга от 2000
            начните с тактических задач своего уровня.
          </p>
          <button
            type="button"
            className="training-primary-btn"
            onClick={() => navigate("/puzzles?rating_min=2000")}
          >
            Решать задачи от 2000 →
          </button>
          <button
            type="button"
            className="training-secondary-btn"
            onClick={() => navigate("/training?basics=1")}
          >
            Открыть базовый курс
          </button>
        </section>
      </div>
    );
  }

  if (!selectedModule) {
    const course = progress?.modules;

    return (
      <div
        className={`training-page training-page--grid training-page--grid-with-due${
          duePanelOpen ? " training-page--due-open" : ""
        }`}
        ref={pageRef}
      >
        <main className="training-course-main">
        <section className="training-hero">
          <div className="training-kicker">Учебный режим</div>
          <h1>Обучение шахматам</h1>
          <p>
            Изучайте особенности каждой фигуры последовательно: сначала теория,
            затем интерактивные задания на шахматной доске.
          </p>
          <RatingBandProgress userId={userId} />

          {course && course.total > 0 && (
            <div className="training-course-progress">
              <div className="training-course-progress__head">
                <span>
                  Пройдено модулей: {course.completed} из {course.total}
                </span>
                <strong>{Math.round(course.percent)}%</strong>
              </div>
              <ProgressBar
                value={course.percent}
                label={`Прогресс курса: ${Math.round(course.percent)}%`}
              />
            </div>
          )}
        </section>

        {error && <div className="training-global-error">{error}</div>}

        <div className="training-modules-path" ref={modulesPathRef}>
          {modules.map((module, index) => {
            const moduleProgress = moduleProgressBySlug.get(module.slug);

            return (
              <button
                type="button"
                key={module.id}
                ref={(el) => {
                  if (el) moduleNodeRefs.current.set(module.id, el);
                  else moduleNodeRefs.current.delete(module.id);
                }}
                className={`training-module-node training-module-node--${
                  index % 2 === 0 ? "left" : "right"
                } ${module.enabled ? "" : "training-module-node--disabled"}`}
                onClick={() =>
                  navigate(`/training?module=${encodeURIComponent(module.slug)}`)
                }
                disabled={!module.enabled}
              >
                <span className="training-module-node-icon" aria-hidden="true">
                  {MODULE_ICONS[module.slug] || "♟"}
                </span>
                <span className="training-module-node-main">
                  <span className="training-module-node-title">{module.title}</span>
                  <span className="training-module-node-description">
                    {module.description}
                  </span>

                  {module.enabled && moduleProgress && (
                    <span className="training-module-progress">
                      <span className="training-module-progress__head">
                        <span>
                          {moduleProgress.tasks_solved} из{" "}
                          {moduleProgress.tasks_total} заданий
                        </span>
                        <strong>{Math.round(moduleProgress.percent)}%</strong>
                      </span>
                      <ProgressBar
                        value={moduleProgress.percent}
                        label={`${module.title}: ${Math.round(moduleProgress.percent)}%`}
                      />
                    </span>
                  )}

                  <span className="training-module-meta">
                    {module.enabled
                      ? `${module.lesson_count} уроков · ${module.task_count} заданий`
                      : "Скоро"}
                  </span>
                </span>
              </button>
            );
          })}
          {knightPos && (
            <div
              className={`training-knight-rider${
                knightInstant ? " training-knight-rider--instant" : ""
              }`}
              style={{ top: knightPos.top, left: knightPos.left }}
              aria-hidden="true"
            >
              <span className="training-knight-hop" key={knightHops}>
                <span className="training-knight-glyph">♘</span>
              </span>
            </div>
          )}
        </div>
        </main>

        <aside
          className={`training-due-rail ${
            duePanelOpen ? "training-due-rail--open" : "training-due-rail--collapsed"
          }`}
          aria-label="Интервальные повторения"
        >
          <button
            type="button"
            className="training-due-toggle"
            aria-expanded={duePanelOpen}
            aria-controls="training-due-panel"
            aria-label={`${duePanelOpen ? "Скрыть" : "Показать"} задания для повторения: ${dueItems.length}`}
            title="Задания для повторения"
            onClick={() => setDuePanelOpen((open) => !open)}
          >
            <span className="training-due-toggle-icon" aria-hidden="true">🔁</span>
            <span className="training-due-toggle-label">Повторения</span>
            <span className="training-due-toggle-count">{dueItems.length}</span>
            <span className="training-due-toggle-chevron" aria-hidden="true">
              {duePanelOpen ? "−" : "+"}
            </span>
          </button>

          {duePanelOpen && (
            <section className="training-due-panel" id="training-due-panel">
              <div className="training-due-heading">
                <strong>Пора повторить: {formatTaskCount(dueItems.length)}</strong>
                <span>Повторите задания, чтобы закрепить материал.</span>
              </div>
              {dueItems.length > 0 ? (
                <div className="training-due-list">
                  {dueItems.map((item) => (
                    <button
                      type="button"
                      key={item.review.id}
                      className="training-due-card"
                      onClick={() => openDueItem(item)}
                    >
                      <span className="training-due-card-main">
                        <strong>{item.task.title}</strong>
                        <span>
                          {item.module.title} · {item.lesson.title}
                        </span>
                      </span>
                      <span className="training-due-card-meta">
                        повторений: {item.review.reps}
                      </span>
                      <span className="training-due-card-arrow" aria-hidden="true">→</span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="training-due-empty">Сейчас нет заданий для повтора.</p>
              )}
            </section>
          )}
        </aside>

        <SiteFooter />
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="training-page training-page--lessons" ref={pageRef}>
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

        <div className="training-path">
          {lessonSections.map((section) => (
            <div className="training-path-section" key={section.key}>
              {section.title && (
                <div className="training-path-banner">
                  <strong>{section.title}</strong>
                  <span>
                    {section.done} из {section.lessons.length} уроков
                  </span>
                </div>
              )}
              <div className="training-path-nodes">
                {section.lessons.map(({ item, state, progress, globalIndex }) => (
                  <button
                    type="button"
                    className={`training-path-node training-path-node--${state} ${
                      globalIndex % 2 === 0 ? "training-path-node--left" : "training-path-node--right"
                    }`}
                    key={item.id}
                    onClick={() => openLesson(item.id)}
                    title={`${item.title}${
                      state === "done" ? " - пройден" : state === "current" ? " - текущий" : ""
                    }`}
                  >
                    <span className="training-path-circle" aria-hidden="true">
                      {state === "done" ? "✓" : globalIndex + 1}
                    </span>
                    <span className="training-path-label">
                      <span className="training-path-title">{item.title}</span>
                      <span className="training-path-meta">
                        {item.task_count}{" "}
                        {item.task_count === 1 ? "задание" : "задания"}
                      </span>
                      {progress && (
                        <span className="training-path-progress-bar">
                          <ProgressBar
                            value={progress.percent}
                            label={`${item.title}: ${progress.tasks_solved} из ${progress.tasks_total}`}
                          />
                        </span>
                      )}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
        <SiteFooter />
      </div>
    );
  }

  return (
    <div className="training-page training-page--lesson" ref={pageRef}>
      <div className="training-toolbar training-toolbar--spread">
        <button type="button" className="training-back-btn" onClick={backToLessons}>
          ← Уроки модуля
        </button>
      </div>

      {error && <div className="training-global-error">{error}</div>}

      <div className="training-lesson-layout">
        <aside className="training-theory-panel">
          <div className="training-kicker">{selectedModule.title}</div>
          <h2>{lesson.title}</h2>
          <p>{lesson.theory}</p>

          <div className="training-theory-navigation">
            <button
              type="button"
              className="training-progress-nav"
              onClick={() => navigateLesson(-1)}
              disabled={currentLessonIndex <= 0}
              aria-label="Предыдущий урок"
              title="Предыдущий урок"
            >
              ←
            </button>
            <span>{currentLessonIndex + 1} / {lessons.length}</span>
            <button
              type="button"
              className="training-progress-nav"
              onClick={() => navigateLesson(1)}
              disabled={currentLessonIndex < 0 || currentLessonIndex >= lessons.length - 1}
              aria-label="Следующий урок"
              title="Следующий урок"
            >
              →
            </button>
          </div>

          <div className="training-lesson-progress">
            <button
              type="button"
              className="training-progress-nav"
              onClick={() => setTaskIndex((index) => Math.max(0, index - 1))}
              disabled={taskIndex <= 0}
              aria-label="Предыдущее задание"
              title="Предыдущее задание"
            >
              ←
            </button>
            {tasks.map((task, index) => {
              const solved = solvedTaskIds.has(task.id);
              // Последовательное прохождение: точка открыта, если задание
              // уже решено, оно первое, или решено предыдущее. Прыгнуть
              // с 1 сразу на 3, не решив 2, нельзя.
              const unlocked =
                index === 0 ||
                solved ||
                solvedTaskIds.has(tasks[index - 1].id);

              return (
                <button
                  type="button"
                  key={task.id}
                  className={[
                    "training-progress-dot",
                    index === taskIndex ? "training-progress-dot--active" : "",
                    solved ? "training-progress-dot--solved" : "",
                    unlocked ? "" : "training-progress-dot--locked",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={() => setTaskIndex(index)}
                  disabled={!unlocked}
                  title={`Задание ${index + 1}: ${task.title}${
                    solved
                      ? " - решено"
                      : unlocked
                        ? ""
                        : " - сначала решите предыдущее"
                  }`}
                >
                  {index + 1}
                </button>
              );
            })}
            <button
              type="button"
              className="training-progress-nav"
              onClick={() => setTaskIndex((index) => Math.min(tasks.length - 1, index + 1))}
              disabled={taskIndex >= tasks.length - 1}
              aria-label="Следующее задание"
              title="Следующее задание"
            >
              →
            </button>
          </div>
        </aside>

        <main className="training-board-column">
          <button
            type="button"
            className="learning-crosslink learning-crosslink--over-board"
            onClick={openPuzzlesForCurrentTopic}
          >
            Проверить на пазлах этой темы →
          </button>
          {currentTask ? (
            <div className="training-board-wrapper" data-logo-anchor>
              <ReactChessboard
                id={boardId}
                position={boardFen || currentTask.fen}
                onPieceDrop={handlePieceDrop}
                onSquareClick={handleSquareClick}
                boardWidth={boardWidth}
                animationDuration={180}
                boardOrientation="white"
                showBoardNotation={true}
                customNotationStyle={{
                  fontSize: "12px",
                  fontFamily: "Cormorant Garamond, Georgia, serif",
                  fontWeight: "500",
                  color: "#4a3220",
                }}
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
                  {taskResult.correct && taskIndex < tasks.length - 1 && (
                    <span className="training-auto-hint">
                      Следующее задание откроется автоматически…
                    </span>
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
      <SiteFooter />
    </div>
  );
}
