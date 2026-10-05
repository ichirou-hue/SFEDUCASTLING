import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchAssessmentStatus,
  fetchLinkedChessAccounts,
  linkChessAccount,
  submitAssessmentOnboarding,
  submitAssessmentUserFeedback,
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
    rating_type: null,
    rating: null,
    rating_scale: null,
    rating_usable: false,
    linked_account: false,
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

function platformTitle(platform) {
  return platform === "chesscom" ? "Chess.com" : "Lichess";
}

function ratingTypeTitle(value) {
  if (value === "rapid") return "Rapid";
  if (value === "bullet") return "Bullet";
  return "Blitz";
}

function OnboardingForm({ onDone }) {
  const [form, setForm] = useState(initialForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [linking, setLinking] = useState(false);
  const [linkError, setLinkError] = useState("");
  const [linkedResult, setLinkedResult] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetchLinkedChessAccounts()
      .then((data) => {
        if (cancelled) return;
        const accounts = data?.items || [];
        const account = accounts.find((item) => item.platform === "lichess") || accounts[0];
        if (!account) return;
        setForm((prev) => ({
          ...prev,
          q2: {
            ...prev.q2,
            has_rating: true,
            platform: account.platform,
            username: account.username || "",
            rating_type: account.rating_type || null,
            rating: account.rating ?? null,
            rating_scale: account.rating_scale || null,
            rating_usable: Boolean(account.rating_usable),
            linked_account: true,
          },
        }));
        setLinkedResult({ account, profile: null, assessment_rating: null });
      })
      .catch(() => {
        // Привязка необязательна; ошибка чтения не блокирует анкету.
      });
    return () => { cancelled = true; };
  }, []);

  const canSubmit = useMemo(() => {
    const q2ok = !form.q2.has_rating || Boolean(form.q2.linked_account && form.q2.username);
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

  const changeExternalPlatform = (platform) => {
    setLinkedResult(null);
    setLinkError("");
    setForm((prev) => ({
      ...prev,
      q2: {
        ...prev.q2,
        platform,
        username: "",
        rating_type: null,
        rating: null,
        rating_scale: null,
        rating_usable: false,
        linked_account: false,
      },
    }));
  };

  const changeExternalUsername = (username) => {
    setLinkedResult(null);
    setLinkError("");
    setForm((prev) => ({
      ...prev,
      q2: {
        ...prev.q2,
        username,
        rating_type: null,
        rating: null,
        rating_scale: null,
        rating_usable: false,
        linked_account: false,
      },
    }));
  };

  const linkExternalAccount = async () => {
    const username = form.q2.username.trim();
    if (!username) {
      setLinkError("Введите логин шахматного аккаунта.");
      return;
    }
    setLinking(true);
    setLinkError("");
    try {
      const data = await linkChessAccount(username, form.q2.platform);
      const account = data.account;
      setLinkedResult(data);
      setForm((prev) => ({
        ...prev,
        q2: {
          ...prev.q2,
          has_rating: true,
          platform: account.platform,
          username: account.username,
          rating_type: account.rating_type || null,
          rating: account.rating ?? null,
          rating_scale: account.rating_scale || null,
          rating_usable: Boolean(account.rating_usable),
          linked_account: true,
        },
      }));
    } catch (e) {
      setLinkError(e?.response?.data?.detail || e?.message || "Не удалось привязать аккаунт");
    } finally {
      setLinking(false);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError("");
    try {
      const payload = {
        ...form,
        q2: form.q2.has_rating
          ? {
              ...form.q2,
              username: form.q2.username || null,
              rating: form.q2.rating == null ? null : Number(form.q2.rating),
            }
          : {
              has_rating: false,
              platform: null,
              username: null,
              rating_type: null,
              rating: null,
              rating_scale: null,
              rating_usable: false,
              linked_account: false,
            },
        q8: form.q8 || null,
        guardian_contact: form.guardian_contact || null,
      };
      const data = await submitAssessmentOnboarding(payload);
      window.dispatchEvent(new Event("sfedu-assessment-updated"));
      onDone?.(data);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось сохранить анкету");
    } finally {
      setSubmitting(false);
    }
  };


  const linkedAccount = linkedResult?.account;
  const ratingInfo = linkedResult?.assessment_rating;
  const profilePerfs = linkedResult?.profile?.perfs;

  return (
    <form className="assessment-card assessment-form" onSubmit={submit}>
      <div className="assessment-kicker">Шаг 1 из 3 · около 2 минут</div>
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
        <legend>Q2. Есть ли у вас аккаунт Lichess или Chess.com?</legend>
        <div className="assessment-inline-toggle">
          <label>
            <input
              type="radio"
              checked={!form.q2.has_rating}
              onChange={() => setForm((prev) => ({ ...prev, q2: { ...prev.q2, has_rating: false } }))}
            />
            Нет / не хочу использовать
          </label>
          <label>
            <input
              type="radio"
              checked={form.q2.has_rating}
              onChange={() => setForm((prev) => ({ ...prev, q2: { ...prev.q2, has_rating: true } }))}
            />
            Да, привязать
          </label>
        </div>

        {form.q2.has_rating && (
          <div className="assessment-account-link">
            <div className="assessment-q2-grid assessment-q2-grid--link">
              <select
                value={form.q2.platform}
                onChange={(e) => changeExternalPlatform(e.target.value)}
                disabled={linking}
              >
                <option value="lichess">Lichess</option>
                <option value="chesscom">Chess.com</option>
              </select>
              <input
                value={form.q2.username}
                placeholder={`Логин ${platformTitle(form.q2.platform)}`}
                onChange={(e) => changeExternalUsername(e.target.value)}
                disabled={linking}
              />
            </div>
            <button
              type="button"
              className="assessment-secondary assessment-link-btn"
              onClick={linkExternalAccount}
              disabled={linking || !form.q2.username.trim()}
            >
              {linking ? "Проверяем профиль..." : "Найти и привязать аккаунт"}
            </button>

            {linkError && <div className="assessment-error">{String(linkError)}</div>}

            {linkedAccount && (
              <div className={`assessment-linked-account ${linkedAccount.rating_usable ? "is-usable" : "is-warning"}`}>
                <div className="assessment-linked-head">
                  <div>
                    <strong>{platformTitle(linkedAccount.platform)} · {linkedAccount.username}</strong>
                    <span>Аккаунт связан с вашим профилем SFEDUCASTLING</span>
                  </div>
                  <span className="assessment-linked-badge">Привязан</span>
                </div>

                {profilePerfs && (
                  <div className="assessment-perfs">
                    {["blitz", "rapid", "bullet"].map((type) => {
                      const perf = profilePerfs[type];
                      if (!perf?.rating) return null;
                      return (
                        <div key={type}>
                          <span>{ratingTypeTitle(type)}</span>
                          <strong>{perf.rating}</strong>
                          <small>{perf.games || 0} партий</small>
                        </div>
                      );
                    })}
                  </div>
                )}

                {linkedAccount.rating != null && (
                  <p className="assessment-linked-rating">
                    Для оценки выбран: <strong>{ratingTypeTitle(linkedAccount.rating_type)} {linkedAccount.rating}</strong>
                    {linkedAccount.games ? ` · ${linkedAccount.games} партий` : ""}
                    {linkedAccount.rating_deviation != null ? ` · RD ${Math.round(linkedAccount.rating_deviation)}` : ""}
                  </p>
                )}

                {linkedAccount.rating_usable ? (
                  <p className="assessment-link-ok">
                    Этот рейтинг будет использован как предварительный при подборе входного теста.
                  </p>
                ) : (
                  <p className="assessment-link-warning">
                    Профиль сохранён, но рейтинг пока не подходит для стартовой оценки. Будет использован ответ Q1.
                  </p>
                )}

                {(ratingInfo?.warnings || []).map((warning) => (
                  <small className="assessment-link-note" key={warning}>{warning}</small>
                ))}
                <small className="assessment-link-note">
                  Это мягкая привязка по публичному профилю: без OAuth SFEDUCASTLING не может доказать владение аккаунтом.
                </small>
              </div>
            )}
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
        {submitting ? "Сохраняем..." : "Сохранить анкету и перейти к задачам"}
      </button>
    </form>
  );
}


function PostTestFeedback({ onDone }) {
  const [text, setText] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const complete = async (skipped) => {
    if (saving) return;
    const value = text.trim();
    if (!skipped && !value) return;

    setSaving(true);
    setError("");
    try {
      await submitAssessmentUserFeedback(skipped ? null : value, skipped);
      window.dispatchEvent(new Event("sfedu-assessment-updated"));
      await onDone?.();
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Не удалось сохранить отзыв");
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="assessment-page">
      <section className="assessment-card assessment-feedback-card">
        <div className="assessment-kicker">Шаг 3 из 3 · тест завершён</div>
        <h1>Помогите нам улучшить SFEDUCASTLING</h1>
        <p className="assessment-lead">
          Что бы вы улучшили в платформе или что хотели бы видеть в следующих версиях?
          Отзыв необязателен и не влияет на ваш рейтинг или результат теста.
        </p>

        <div className="assessment-user-feedback">
          <textarea
            value={text}
            maxLength={2000}
            rows={6}
            disabled={saving}
            placeholder="Например: хотелось бы больше разборов партий, новые типы задач, подробную статистику прогресса..."
            onChange={(e) => {
              setText(e.target.value);
              setError("");
            }}
          />
          <div className="assessment-user-feedback__footer">
            <small>{text.length}/2000</small>
          </div>
        </div>

        {error && <div className="assessment-error">{String(error)}</div>}

        <div className="assessment-modal-actions">
          <button
            type="button"
            className="assessment-primary"
            disabled={!text.trim() || saving}
            onClick={() => complete(false)}
          >
            {saving ? "Сохраняем..." : "Отправить и посмотреть результат"}
          </button>
          <button
            type="button"
            className="assessment-secondary"
            disabled={saving}
            onClick={() => complete(true)}
          >
            Пропустить и посмотреть результат
          </button>
        </div>
      </section>
    </main>
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

  if (status.phase === "feedback") {
    return <PostTestFeedback onDone={loadStatus} />;
  }

  if (status.phase === "completed") {
    const testResult = status.test_result || {};
    const score = testResult.score || {};
    const result = testResult.result || score.result || {};
    const finalRating = result.rating ?? score.final_rating ?? status.elo;
    const level = result.level ?? result.band ?? status.skill_band;
    const total = score.total ?? 20;
    const correct = score.correct;

    return (
      <main className="assessment-page">
        <section className="assessment-card assessment-feedback-card">
          <div className="assessment-kicker">Оценка завершена</div>
          <h1>Ваш стартовый профиль готов</h1>
          <div className="assessment-rating-chip">
            Рейтинг {finalRating ?? "—"} · уровень {level ?? "—"} из 4
          </div>

          {correct != null && (
            <p><strong>{correct}</strong> из {total} задач решено точно.</p>
          )}

          {(testResult.feedback || score.feedback || []).length > 0 && (
            <div className="assessment-feedback-list">
              {(testResult.feedback || score.feedback || []).map((item) => (
                <div key={item}>• {item}</div>
              ))}
            </div>
          )}

          <p>Стартовая оценка завершена. Напоминание больше показываться не будет, а дальнейшая сложность будет корректироваться по вашей практике.</p>
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
