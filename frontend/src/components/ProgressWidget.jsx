import { useNavigate } from "react-router-dom";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchLearningProgress,
  fetchTrainingProgress,
  getAccessToken,
} from "../api.js";
import "./ProgressWidget.css";

const TOPIC_ICONS = {
  pawn: "♙",
  knight: "♘",
  bishop: "♗",
  rook: "♖",
  queen: "♕",
  king: "♔",
  "special-rules": "♙",
  "check-mate-stalemate": "♚",
};

function percent(value) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value)}%`;
}

export default function ProgressWidget() {
  const navigate = useNavigate();
  const [training, setTraining] = useState(null);
  const [learning, setLearning] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!getAccessToken()) {
      setTraining(null);
      setLearning(null);
      setLoading(false);
      setError(null);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const [trainingData, learningData] = await Promise.all([
        fetchTrainingProgress(),
        fetchLearningProgress(),
      ]);
      setTraining(trainingData);
      setLearning(learningData);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Не удалось загрузить прогресс.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const refresh = () => load();
    window.addEventListener("sfedu-progress-updated", refresh);
    return () => window.removeEventListener("sfedu-progress-updated", refresh);
  }, [load]);

  const activeTopics = useMemo(
    () => (training?.topics || []).filter((topic) => topic.attempts > 0),
    [training],
  );

  if (!getAccessToken()) return null;

  if (loading && !training && !learning) {
    return (
      <section className="progress-widget progress-widget--loading">
        Загружаем прогресс…
      </section>
    );
  }

  if (error && !training && !learning) {
    return (
      <section className="progress-widget progress-widget--error">
        {error}
      </section>
    );
  }

  const modules = training?.modules || { completed: 0, total: 0, percent: 0 };
  const trainingAttempts = training?.attempts || {
    total: 0,
    correct: 0,
    accuracy: 0,
  };
  const puzzles = learning || {
    attempted: 0,
    solved: 0,
    attempts: 0,
    correct_attempts: 0,
    accuracy: 0,
  };

  return (
    <section className="progress-widget" aria-label="Общий прогресс обучения">
      <div className="progress-widget__header">
        <div>
          <div className="progress-widget__kicker">Два трека · один прогресс</div>
          <h2>Мой прогресс</h2>
        </div>
        <div className="progress-widget__streak" title="Серия учебных дней">
          <span className="progress-widget__streak-icon">🔥</span>
          <span>
            <strong>{training?.streak || 0}</strong>
            <small>дн. подряд</small>
          </span>
        </div>
      </div>

      <div className="progress-widget__tracks">
        <article className="progress-track-card">
          <div className="progress-track-card__title">
            <span>♘</span>
            <strong>Курс</strong>
          </div>
          <div className="progress-track-card__value">
            {modules.completed} / {modules.total}
          </div>
          <div className="progress-track-card__caption">модулей пройдено</div>
          <div className="progress-meter" aria-hidden="true">
            <span style={{ width: `${Math.min(100, modules.percent || 0)}%` }} />
          </div>
          <div className="progress-track-card__footer">
            <span>Точность</span>
            <strong>{percent(trainingAttempts.accuracy)}</strong>
          </div>
        </article>

        <article className="progress-track-card">
          <div className="progress-track-card__title">
            <span>♜</span>
            <strong>Пазлы</strong>
          </div>
          <div className="progress-track-card__value">{puzzles.solved}</div>
          <div className="progress-track-card__caption">
            решено · {puzzles.attempted} попробовано
          </div>
          <div className="progress-meter" aria-hidden="true">
            <span style={{ width: `${Math.min(100, puzzles.accuracy || 0)}%` }} />
          </div>
          <div className="progress-track-card__footer">
            <span>Точность попыток</span>
            <strong>{percent(puzzles.accuracy)}</strong>
          </div>
        </article>
      </div>

      <div className="progress-widget__adaptive">
        <div>
          <strong>Персональная тренировка</strong>
          <span>70% задач по слабым темам, 30% — закрепление сильных.</span>
        </div>
        <button type="button" onClick={() => navigate("/puzzles?adaptive=1")}>
          Потренировать слабые места →
        </button>
      </div>

      <div className="progress-widget__topics">
        <div className="progress-widget__topics-title">Точность по темам курса</div>
        {activeTopics.length > 0 ? (
          <div className="progress-topic-list">
            {activeTopics.map((topic) => (
              <div className="progress-topic" key={topic.module_id}>
                <span className="progress-topic__icon">
                  {TOPIC_ICONS[topic.slug] || "♟"}
                </span>
                <span className="progress-topic__name">{topic.title}</span>
                <span className="progress-topic__bar" aria-hidden="true">
                  <span
                    style={{ width: `${Math.min(100, topic.accuracy || 0)}%` }}
                  />
                </span>
                <strong>{percent(topic.accuracy)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="progress-widget__empty">
            Выполните первое задание курса — здесь появится точность по темам.
          </div>
        )}
      </div>
    </section>
  );
}
