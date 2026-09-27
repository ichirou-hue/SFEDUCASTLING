import { useNavigate } from "react-router-dom";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchLearningProgress,
  fetchTrainingProgress,
  fetchWeaknessProfile,
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

function barWidth(value) {
  if (value === null || value === undefined) return null;
  return `${Math.min(100, Math.max(0, value))}%`;
}

export default function ProgressWidget() {
  const navigate = useNavigate();
  const [training, setTraining] = useState(null);
  const [learning, setLearning] = useState(null);
  const [weakness, setWeakness] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!getAccessToken()) {
      setTraining(null);
      setLearning(null);
      setWeakness(null);
      setLoading(false);
      setError(null);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const [trainingData, learningData, weaknessData] = await Promise.all([
        fetchTrainingProgress(),
        fetchLearningProgress(),
        fetchWeaknessProfile(),
      ]);
      setTraining(trainingData);
      setLearning(learningData);
      setWeakness(weaknessData);
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
    () => (weakness?.topics || []).filter((topic) => topic.attempts > 0),
    [weakness],
  );

  const selectionRule = weakness?.selection_rule;
  const weakShare = Number(selectionRule?.weak_share ?? 0) * 100;
  const strongShare = Number(selectionRule?.strong_share ?? 0) * 100;

  if (!getAccessToken()) return null;

  if (loading && !training && !learning && !weakness) {
    return (
      <section className="progress-widget progress-widget--loading">
        Загружаем прогресс…
      </section>
    );
  }

  if (error && !training && !learning && !weakness) {
    return (
      <section className="progress-widget progress-widget--error">
        {error}
      </section>
    );
  }

  const modules = training?.modules;
  const trainingAttempts = training?.attempts;

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
            <strong>{training?.streak ?? "—"}</strong>
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
            {modules ? `${modules.completed} / ${modules.total}` : "—"}
          </div>
          <div className="progress-track-card__caption">модулей пройдено</div>
          {modules?.percent != null && (
            <div className="progress-meter" aria-hidden="true">
              <span style={{ width: barWidth(modules.percent) }} />
            </div>
          )}
          <div className="progress-track-card__footer">
            <span>Точность</span>
            <strong>{percent(trainingAttempts?.accuracy)}</strong>
          </div>
        </article>

        <article className="progress-track-card">
          <div className="progress-track-card__title">
            <span>♜</span>
            <strong>Пазлы</strong>
          </div>
          <div className="progress-track-card__value">
            {learning?.solved ?? "—"}
          </div>
          <div className="progress-track-card__caption">
            решено · {learning?.attempted ?? "—"} попробовано
          </div>
          {learning?.accuracy != null && (
            <div className="progress-meter" aria-hidden="true">
              <span style={{ width: barWidth(learning.accuracy) }} />
            </div>
          )}
          <div className="progress-track-card__footer">
            <span>Точность попыток</span>
            <strong>{percent(learning?.accuracy)}</strong>
          </div>
        </article>
      </div>

      <div className="progress-widget__adaptive">
        <div>
          <strong>Персональная тренировка</strong>
          <span>
            {Number.isFinite(weakShare) && Number.isFinite(strongShare)
              ? `${Math.round(weakShare)}% задач по слабым темам, ${Math.round(
                  strongShare,
                )}% — закрепление сильных.`
              : "Тренируем слабые темы, закрепляем сильные."}
          </span>
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
              <div className="progress-topic" key={topic.slug}>
                <span className="progress-topic__icon">
                  {TOPIC_ICONS[topic.slug] || "♟"}
                </span>
                <span className="progress-topic__name">{topic.title}</span>
                {topic.accuracy != null && (
                  <span className="progress-topic__bar" aria-hidden="true">
                    <span style={{ width: barWidth(topic.accuracy) }} />
                  </span>
                )}
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