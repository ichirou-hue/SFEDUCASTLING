import { useEffect, useMemo, useState } from "react";
import { fetchPhaseAccuracy, getAccessToken } from "../api.js";
import "./PhaseAccuracyWidget.css";

const CHART = { width: 740, height: 270, left: 76, right: 700, top: 28, bottom: 184 };
const MIN_MOVES_FOR_INSIGHT = 5;
const PENDING_KEY = "sfedu-phase-accuracy-pending";
const GAME_COUNT_KEY = "sfedu-phase-accuracy-game-count";
const ANALYSIS_WAIT_MS = 180_000;

function pointFor(phase, index) {
  const x = CHART.left + (index * (CHART.right - CHART.left)) / 2;
  const y = CHART.top + ((100 - phase.accuracy_percent) / 100) * (CHART.bottom - CHART.top);
  return { x, y };
}

function curvePath(items) {
  if (items.length < 2) return "";
  const runs = [];
  items.forEach((item) => {
    const currentRun = runs[runs.length - 1];
    if (!currentRun || item.index !== currentRun[currentRun.length - 1].index + 1) runs.push([]);
    runs[runs.length - 1].push(item);
  });

  return runs.filter((run) => run.length > 1).map((run) => {
    const segments = run.slice(0, -1).map((item, index) => {
      const nextItem = run[index + 1];
      const point = item.point;
      const next = nextItem.point;
      const previous = run[index - 1]?.point || point;
      const afterNext = run[index + 2]?.point || next;
      const tension = 0.18;
      const cp1x = point.x + (next.x - previous.x) * tension;
      const cp1y = point.y + (next.y - previous.y) * tension;
      const cp2x = next.x - (afterNext.x - point.x) * tension;
      const cp2y = next.y - (afterNext.y - point.y) * tension;
      return `C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${next.x} ${next.y}`;
    });
    return `M ${run[0].point.x} ${run[0].point.y} ${segments.join(" ")}`;
  }).join(" ");
}

function insightFor(phases, gamesAnalyzed) {
  if (!gamesAnalyzed) return "Завершите несколько партий, чтобы увидеть точность по этапам.";
  const sampled = phases.filter((phase) => phase.moves_evaluated >= MIN_MOVES_FOR_INSIGHT && phase.accuracy_percent != null);
  if (!sampled.length) {
    const hasAny = phases.some((phase) => phase.moves_evaluated > 0);
    return hasAny
      ? "Пока мало оценённых ходов для надёжного вывода. Текущие значения уже показаны ниже."
      : "Оценки ходов пока не готовы или отсутствуют. Обновите профиль после анализа партии.";
  }
  const strongest = [...sampled].sort((a, b) => b.accuracy_percent - a.accuracy_percent)[0];
  const weakest = [...sampled].sort((a, b) => a.accuracy_percent - b.accuracy_percent)[0];
  if (strongest.key === weakest.key || strongest.accuracy_percent - weakest.accuracy_percent < 3) {
    return "Точность по этапам пока близкая — выраженного слабого места нет.";
  }
  return `Увереннее всего — ${strongest.label.toLowerCase()}. Больше внимания можно уделить этапу «${weakest.label.toLowerCase()}».`;
}

function pluralize(count, forms) {
  const category = new Intl.PluralRules("ru").select(count);
  return forms[category] || forms.other;
}

function ErrorCount({ count, forms, tone }) {
  return (
    <span className={`phase-accuracy__error phase-accuracy__error--${tone}`}>
      <strong>{count}</strong> {pluralize(count, forms)}
    </span>
  );
}

function AccuracyChart({ phases, selectedKey, onSelect }) {
  const plotted = phases
    .map((phase, index) => ({ phase, index, point: phase.accuracy_percent == null ? null : pointFor(phase, index) }))
    .filter((item) => item.point);
  const path = curvePath(plotted);
  const areaPath = plotted.length === phases.length && path
    ? `${path} L ${plotted.at(-1).point.x} ${CHART.bottom} L ${plotted[0].point.x} ${CHART.bottom} Z`
    : "";

  return (
    <div className="phase-accuracy__chart-wrap">
      <svg
        className="phase-accuracy__chart"
        viewBox={`0 0 ${CHART.width} ${CHART.height}`}
        role="group"
        aria-label="График точности ходов по этапам шахматной партии"
      >
        <defs>
          <linearGradient id="phaseAccuracyLine" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#4c9b75" />
            <stop offset="52%" stopColor="#62a9a0" />
            <stop offset="100%" stopColor="#6f88c5" />
          </linearGradient>
          <linearGradient id="phaseAccuracyFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#6aa88d" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#6aa88d" stopOpacity="0.015" />
          </linearGradient>
        </defs>

        {[100, 75, 50, 25, 0].map((value) => {
          const y = CHART.top + ((100 - value) / 100) * (CHART.bottom - CHART.top);
          return (
            <g key={value} className="phase-accuracy__grid-row">
              <text x="12" y={y + 4}>{value}%</text>
              <line x1={CHART.left - 12} x2={CHART.right + 12} y1={y} y2={y} />
            </g>
          );
        })}

        {areaPath && <path className="phase-accuracy__area" d={areaPath} />}
        {path && <path className="phase-accuracy__curve" d={path} />}

        {phases.map((phase, index) => {
          const point = phase.accuracy_percent == null ? null : pointFor(phase, index);
          const selected = selectedKey === phase.key;
          return (
            <g
              key={phase.key}
              className={`phase-accuracy__point${selected ? " is-selected" : ""}${point ? "" : " is-empty"}`}
              role="button"
              tabIndex="0"
              aria-label={`${phase.label}: ${point ? `${phase.accuracy_percent}%` : "пока нет данных"}, ${phase.moves_evaluated} оценённых ходов`}
              aria-pressed={selected}
              onClick={() => onSelect(phase.key)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(phase.key);
                }
              }}
            >
              <title>{`${phase.label}: ${point ? `${phase.accuracy_percent}%` : "нет данных"} · ${phase.moves_evaluated} оценённых ходов`}</title>
              {point ? (
                <>
                  <circle className="phase-accuracy__point-halo" cx={point.x} cy={point.y} r="15" />
                  <circle className="phase-accuracy__point-core" cx={point.x} cy={point.y} r="7" />
                  <text className="phase-accuracy__point-value" x={point.x} y={point.y - 19} textAnchor="middle">
                    {phase.accuracy_percent}%
                  </text>
                </>
              ) : (
                <circle className="phase-accuracy__point-empty" cx={pointFor({ accuracy_percent: 50 }, index).x} cy={CHART.bottom + 17} r="4" />
              )}
              <text className="phase-accuracy__axis-label" x={pointFor({ accuracy_percent: 50 }, index).x} y={CHART.height - 13} textAnchor="middle">
                {phase.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function PhaseAccuracyWidget() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [errorStatus, setErrorStatus] = useState(null);
  const [selectedKey, setSelectedKey] = useState(null);

  const load = async (silent = false) => {
    if (!getAccessToken()) {
      setLoading(false);
      return null;
    }
    if (!silent) setLoading(true);
    if (!silent) {
      setError(false);
      setErrorStatus(null);
    }
    try {
      const result = await fetchPhaseAccuracy();
      setData(result);
      localStorage.setItem(GAME_COUNT_KEY, String(result.games_analyzed || 0));
      return result;
    } catch (err) {
      if (!silent) {
        setError(true);
        setErrorStatus(err?.response?.status || null);
      }
      return null;
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    let pollTimer = null;
    let pollDeadline = null;
    let baseline = null;
    try {
      const pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
      if (pending && Date.now() - pending.requestedAt < ANALYSIS_WAIT_MS) {
        baseline = Number.isFinite(pending.baseline) ? pending.baseline : null;
      } else {
        localStorage.removeItem(PENDING_KEY);
      }
    } catch {
      localStorage.removeItem(PENDING_KEY);
    }

    const stopPolling = (clearPending = false) => {
      if (pollTimer) window.clearInterval(pollTimer);
      if (pollDeadline) window.clearTimeout(pollDeadline);
      pollTimer = null;
      pollDeadline = null;
      if (clearPending) localStorage.removeItem(PENDING_KEY);
    };

    const startPolling = (startingCount = null) => {
      if (pollTimer || cancelled) return;
      if (baseline == null) baseline = startingCount;
      const poll = async () => {
        const result = await load(true);
        if (result && baseline != null && result.games_analyzed > baseline) {
          stopPolling(true);
        }
      };
      pollTimer = window.setInterval(poll, 5000);
      pollDeadline = window.setTimeout(() => stopPolling(true), ANALYSIS_WAIT_MS);
    };

    if (!getAccessToken()) {
      setLoading(false);
      return undefined;
    }
    fetchPhaseAccuracy()
      .then((result) => {
        if (!cancelled) {
          setData(result);
          localStorage.setItem(GAME_COUNT_KEY, String(result.games_analyzed || 0));
          if (localStorage.getItem(PENDING_KEY)) startPolling(result.games_analyzed || 0);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(true);
          setErrorStatus(err?.response?.status || null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    const handleGameFinished = () => {
      try {
        const pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
        baseline = Number.isFinite(pending?.baseline) ? pending.baseline : null;
      } catch {
        baseline = null;
      }
      if (baseline == null) {
        load(true).then((result) => {
          if (!cancelled) startPolling(result?.games_analyzed || 0);
        });
      } else {
        startPolling();
      }
    };
    window.addEventListener("sfedu-game-finished", handleGameFinished);
    return () => {
      cancelled = true;
      stopPolling();
      window.removeEventListener("sfedu-game-finished", handleGameFinished);
    };
  }, []);

  const phases = data?.phases || [];
  const selected = useMemo(
    () => phases.find((phase) => phase.key === selectedKey)
      || [...phases].filter((phase) => phase.accuracy_percent != null)
        .sort((a, b) => a.accuracy_percent - b.accuracy_percent)[0]
      || phases[0],
    [phases, selectedKey],
  );

  const hasAccessToken = Boolean(getAccessToken());
  if (!hasAccessToken) {
    return (
      <section className="phase-accuracy phase-accuracy--message" aria-labelledby="phase-accuracy-title">
        <div>
          <div className="user-profile-kicker">Разбор партий</div>
          <h2 id="phase-accuracy-title">Точность по этапам</h2>
          <span>Войдите в аккаунт, чтобы загрузить анализ ваших партий.</span>
        </div>
      </section>
    );
  }
  if (loading) {
    return (
      <section className="phase-accuracy phase-accuracy--loading" aria-labelledby="phase-accuracy-title">
        <div>
          <div className="user-profile-kicker">Разбор партий</div>
          <h2 id="phase-accuracy-title">Точность по этапам</h2>
          <span>Загружаем оценки ходов…</span>
        </div>
      </section>
    );
  }
  if (error) {
    const errorMessage = errorStatus === 404
      ? "Адрес статистики не найден. Остановите run.bat и запустите его заново."
      : errorStatus === 401 || errorStatus === 403
        ? "Сессия не подтверждена. Войдите в аккаунт заново."
        : errorStatus >= 500
          ? `Ошибка сервера${errorStatus ? ` (${errorStatus})` : ""}. Посмотрите сообщение об ошибке в окне run.bat.`
          : "Сервер статистики недоступен. Проверьте, что run.bat запущен.";
    return (
      <section className="phase-accuracy phase-accuracy--message">
        <div>
          <div className="user-profile-kicker">Разбор партий</div>
          <h2 id="phase-accuracy-title">Точность по этапам</h2>
          <span>{errorMessage}</span>
        </div>
        <button type="button" onClick={load}>Повторить</button>
      </section>
    );
  }
  if (!data || !phases.length) {
    return (
      <section className="phase-accuracy phase-accuracy--message" aria-labelledby="phase-accuracy-title">
        <div>
          <div className="user-profile-kicker">Разбор партий</div>
          <h2 id="phase-accuracy-title">Точность по этапам</h2>
          <span>Пока нет данных для графика. Завершите партию — после её сохранения появятся оценки ходов.</span>
        </div>
        <button type="button" onClick={load}>Обновить</button>
      </section>
    );
  }

  const insight = insightFor(phases, data.games_analyzed);
  const selectedErrors = selected?.errors || {};

  return (
    <section className="phase-accuracy" aria-labelledby="phase-accuracy-title">
      <header className="phase-accuracy__header">
        <div>
          <div className="user-profile-kicker">Разбор партий</div>
          <h2 id="phase-accuracy-title">Точность по этапам</h2>
          <p>Как часто удаются хорошие ходы в разных частях партии</p>
        </div>
        <span className="phase-accuracy__games">
          {pluralize(data.games_analyzed, { one: "партия", few: "партии", many: "партий" })}
        </span>
      </header>

      <div className="phase-accuracy__insight">
        <span className="phase-accuracy__insight-icon" aria-hidden="true">✦</span>
        <span>{insight}</span>
      </div>

      <AccuracyChart phases={phases} selectedKey={selected?.key} onSelect={setSelectedKey} />

      <div className="phase-accuracy__summary" aria-label="Показатели по этапам">
        {phases.map((phase) => (
          <button
            type="button"
            key={phase.key}
            className={`phase-accuracy__summary-item${selected?.key === phase.key ? " is-selected" : ""}`}
            onClick={() => setSelectedKey(phase.key)}
            aria-pressed={selected?.key === phase.key}
          >
            <span>{phase.label}</span>
            <strong>{phase.accuracy_percent == null ? "—" : `${phase.accuracy_percent}%`}</strong>
            <small>{phase.moves_evaluated} оценённых ходов</small>
          </button>
        ))}
      </div>

      {selected && (
        <div className="phase-accuracy__details" aria-live="polite">
          <div className="phase-accuracy__details-heading">
            <strong>{selected.label}</strong>
            <span>{selected.moves_without_eval > 0
              ? `${selected.moves_without_eval} ходов без оценки`
              : `${selected.moves_total} ходов игрока`}</span>
          </div>
          <div className="phase-accuracy__errors">
            <ErrorCount count={selectedErrors.inaccuracy || 0} forms={{ one: "неточность", few: "неточности", many: "неточностей" }} tone="soft" />
            <ErrorCount count={selectedErrors.mistake || 0} forms={{ one: "ошибка", few: "ошибки", many: "ошибок" }} tone="medium" />
            <ErrorCount count={selectedErrors.blunder || 0} forms={{ one: "грубая ошибка", few: "грубые ошибки", many: "грубых ошибок" }} tone="strong" />
          </div>
        </div>
      )}

      <footer className="phase-accuracy__footnote">
        Точность — доля ходов хорошего качества среди оценённых Stockfish. Наведите на точку или нажмите на этап для подробностей.
      </footer>
    </section>
  );
}
