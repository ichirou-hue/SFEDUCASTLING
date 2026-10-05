import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchAssessmentStatus,
  submitAssessmentOnboarding,
} from "../api.js";
import LevelTestPage from "./LevelTestPage.jsx";
import "./Assessment.css";

const Q1 = [
  ["never", "Никогда не играл(а)"],
  ["know_moves", "Знаю, как ходят фигуры"],
  ["sometimes", "Играю иногда"],
  ["regularly", "Играю регулярно"],
  ["tournaments", "Играю в турнирах"],
];

const GOALS = [
  ["friends_family", "Играть с друзьями и семьёй"],
  ["online_rating", "Повысить онлайн-рейтинг"],
  ["tournaments", "Готовиться к турнирам"],
  ["child", "Заниматься вместе с ребёнком"],
];

const FORMATS = [
  ["puzzles", "Задачи"],
  ["games", "Партии"],
  ["lessons", "Уроки"],
];

const initialForm = {
  q1: "",
  q2: {
    has_rating: false,
    platform: "lichess",
    username: "",
    rating_type: "blitz",
    rating: "",
  },
  q3: [],
  q4: "",
  q5: ["puzzles", "games", "lessons"],
  q6: "",
  q7: "",
  q8: "",
  parental_consent: false,
  guardian_contact: "",
};

function ChoiceGrid({ options, value, onChange }) {
  return (
    <div className="assessment-choice-grid">
      {options.map(([id, label]) => (
        <button
          type="button"
          key={id}
          className={`assessment-choice ${value === id ? "is-selected" : ""}`}
          onClick={() => onChange(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function OnboardingForm({ onDone }) {
  const [form, setForm] = useState(initialForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState(null);

  const canSubmit = useMemo(() => {
    const q2ok = !form.q2.has_rating || Boolean(form.q2.rating);
    const consentOk = form.q7 !== "under10" || form.parental_consent;
    return Boolean(
      form.q1 &&
      q2ok &&
      form.q3.length >= 1 &&
      form.q3.length <= 2 &&
      form.q4 &&
      form.q5.length === 3 &&
      form.q6 &&
      form.q7 &&
      consentOk
    );
  }, [form]);

  const toggleGoal = (id) => {
    setForm((prev) => {
      const exists = prev.q3.includes(id);
      if (exists) return { ...prev, q3: prev.q3.filter((x) => x !== id) };
      if (prev.q3.length >= 2) return prev;
      return { ...prev, q3: [...prev.q3, id] };
    });
  };

  const changeRank = (index, value) => {
    setForm((prev) => {
      const next = [...prev.q5];
      const otherIndex = next.indexOf(value);
      if (otherIndex >= 0) [next[index], next[otherIndex]] = [next[otherIndex], next[index]];
      else next[index] = value;
      return { ...prev, q5: next };
    });
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError("");
    try {
      const payload = {
        ...form,
        q2: {
          ...form.q2,
          username: form.q2.username || null,
          rating: form.q2.has_rating ? Number(form.q2.rating) : null,
        },
        q8: form.q8 || null,
        guardian_contact: form.guardian_contact || null,
      };
      const data = await submitAssessmentOnboarding(payload);
      setFeedback(data);
      window.dispatchEvent(new Event("sfedu-assessment-updated"));
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось сохранить анкету");
    } finally {
      setSubmitting(false);
    }
  };

  if (feedback) {
    return (
      <section className="assessment-card assessment-feedback-card">
        <div className="assessment-kicker">Анкета завершена</div>
        <h1>Предварительный уровень определён</h1>
        <div className="assessment-rating-chip">
          ≈ {feedback.rating_estimate} · {feedback.rating_group?.key}
        </div>
        <div className="assessment-feedback-list">
          {(feedback.feedback || []).map((item) => <div key={item}>• {item}</div>)}
        </div>
        <button className="assessment-primary" onClick={() => onDone(feedback)}>
          Перейти к персональному тесту
        </button>
      </section>
    );
  }

  return (
    <form className="assessment-card assessment-form" onSubmit={submit}>
      <div className="assessment-kicker">Шаг 1 из 2 · около 2 минут</div>
      <h1>Расскажите немного о своей игре</h1>
      <p className="assessment-lead">
        Ответы нужны только для стартовой персонализации. После анкеты вы получите 20 задач,
        подобранных под ваш предполагаемый уровень и накопленные метрики.
      </p>

      <fieldset>
        <legend>Q1. Играли ли вы в шахматы раньше?</legend>
        <ChoiceGrid options={Q1} value={form.q1} onChange={(q1) => setForm({ ...form, q1 })} />
      </fieldset>

      <fieldset>
        <legend>Q2. Есть ли у вас внешний шахматный рейтинг?</legend>
        <div className="assessment-inline-toggle">
          <label><input type="radio" checked={!form.q2.has_rating} onChange={() => setForm({ ...form, q2: { ...form.q2, has_rating: false } })} /> Нет</label>
          <label><input type="radio" checked={form.q2.has_rating} onChange={() => setForm({ ...form, q2: { ...form.q2, has_rating: true } })} /> Да</label>
        </div>
        {form.q2.has_rating && (
          <div className="assessment-q2-grid">
            <select value={form.q2.platform} onChange={(e) => setForm({ ...form, q2: { ...form.q2, platform: e.target.value } })}>
              <option value="lichess">Lichess</option>
              <option value="chesscom">Chess.com</option>
            </select>
            <select value={form.q2.rating_type} onChange={(e) => setForm({ ...form, q2: { ...form.q2, rating_type: e.target.value } })}>
              <option value="blitz">Blitz</option>
              <option value="rapid">Rapid</option>
              <option value="bullet">Bullet</option>
            </select>
            <input value={form.q2.username} placeholder="Логин (необязательно)" onChange={(e) => setForm({ ...form, q2: { ...form.q2, username: e.target.value } })} />
            <input type="number" min="0" max="3500" required value={form.q2.rating} placeholder="Рейтинг" onChange={(e) => setForm({ ...form, q2: { ...form.q2, rating: e.target.value } })} />
          </div>
        )}
      </fieldset>

      <fieldset>
        <legend>Q3. Зачем вам шахматы? Выберите до двух вариантов.</legend>
        <div className="assessment-choice-grid">
          {GOALS.map(([id, label]) => (
            <button
              type="button"
              key={id}
              className={`assessment-choice ${form.q3.includes(id) ? "is-selected" : ""}`}
              onClick={() => toggleGoal(id)}
            >{label}</button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Q4. Сколько времени готовы заниматься в неделю?</legend>
        <ChoiceGrid
          options={[["lt1", "< 1 часа"], ["1_3", "1–3 часа"], ["3_5", "3–5 часов"], ["5plus", "5+ часов"]]}
          value={form.q4}
          onChange={(q4) => setForm({ ...form, q4 })}
        />
      </fieldset>

      <fieldset>
        <legend>Q5. Расставьте форматы по интересу.</legend>
        <div className="assessment-ranking">
          {form.q5.map((value, index) => (
            <label key={index}>
              <span>{index + 1} место</span>
              <select value={value} onChange={(e) => changeRank(index, e.target.value)}>
                {FORMATS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
              </select>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Q6. Нужны ли подсказки во время заданий?</legend>
        <ChoiceGrid
          options={[["yes", "Да"], ["stuck", "Только если застряну"], ["no", "Нет"]]}
          value={form.q6}
          onChange={(q6) => setForm({ ...form, q6 })}
        />
      </fieldset>

      <fieldset>
        <legend>Q7. Возрастная группа?</legend>
        <ChoiceGrid
          options={[["under10", "До 10 лет"], ["10_16", "10–16 лет"], ["17plus", "17+"]]}
          value={form.q7}
          onChange={(q7) => setForm({ ...form, q7 })}
        />
        {form.q7 === "under10" && (
          <div className="assessment-parental">
            <label>
              <input
                type="checkbox"
                checked={form.parental_consent}
                onChange={(e) => setForm({ ...form, parental_consent: e.target.checked })}
              />
              Есть согласие родителя/опекуна на сохранение результатов обучения
            </label>
            <input
              value={form.guardian_contact}
              placeholder="Контакт родителя/опекуна (необязательно для пилота)"
              onChange={(e) => setForm({ ...form, guardian_contact: e.target.value })}
            />
          </div>
        )}
      </fieldset>

      <fieldset>
        <legend>Q8. Как вы узнали о платформе? <span className="assessment-optional">необязательно</span></legend>
        <ChoiceGrid
          options={[["coach", "Тренер"], ["friends", "Друзья"], ["internet", "Интернет"], ["school", "Школа"], ["other", "Другое"]]}
          value={form.q8}
          onChange={(q8) => setForm({ ...form, q8 })}
        />
      </fieldset>

      {error && <div className="assessment-error">{String(error)}</div>}
      <button className="assessment-primary" disabled={!canSubmit || submitting}>
        {submitting ? "Сохраняем..." : "Получить предварительную оценку"}
      </button>
    </form>
  );
}

export default function AssessmentPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadStatus = async () => {
    setLoading(true);
    setError("");
    try {
      setStatus(await fetchAssessmentStatus());
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось получить состояние оценки");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadStatus(); }, []);

  if (loading) return <main className="assessment-page"><div className="assessment-card">Загружаем стартовую оценку...</div></main>;
  if (error) return <main className="assessment-page"><div className="assessment-card assessment-error">{String(error)}</div></main>;
  if (!status) return null;

  if (status.phase === "not_applicable") {
    return (
      <main className="assessment-page">
        <section className="assessment-card"><h1>Стартовая оценка не требуется</h1><button className="assessment-primary" onClick={() => navigate("/")}>На главную</button></section>
      </main>
    );
  }

  if (status.phase === "completed") {
    return (
      <main className="assessment-page">
        <section className="assessment-card assessment-feedback-card">
          <div className="assessment-kicker">Оценка завершена</div>
          <h1>Ваш стартовый профиль готов</h1>
          <div className="assessment-rating-chip">Рейтинг {status.elo ?? "—"} · уровень {status.skill_band ?? "—"} из 4</div>
          <p>Напоминание больше показываться не будет. Дальнейшая сложность будет корректироваться по вашей практике.</p>
          <button className="assessment-primary" onClick={() => navigate("/training")}>Перейти к обучению</button>
        </section>
      </main>
    );
  }

  if (status.phase === "onboarding") {
    return (
      <main className="assessment-page">
        <OnboardingForm onDone={() => loadStatus()} />
      </main>
    );
  }

  return (
    <div className="assessment-test-stage">
      <LevelTestPage onCompleted={() => loadStatus()} />
    </div>
  );
}
