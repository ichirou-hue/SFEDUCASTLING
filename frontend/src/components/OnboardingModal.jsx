import { useState } from "react";
import { saveOnboarding } from "../api.js";
import "./OnboardingModal.css";

// Дословно из «Анкеты оценки уровня игры», раздел 1 / приложение Д.1.
// Ключи — те же коды, что уходит на бэкенд (тексты могут меняться, коды — нет).

const Q1 = [
  ["never", "Никогда"],
  ["know_moves", "Знаю ходы"],
  ["sometimes", "Играю иногда"],
  ["regular", "Регулярно"],
  ["tournaments", "Турниры"],
];

const Q3 = [
  ["family", "Друзья и семья"],
  ["online_rating", "Онлайн-рейтинг"],
  ["tournaments", "Турниры"],
  ["child", "Занимается ребёнок"],
];

const Q4 = [
  ["lt1", "< 1 ч"],
  ["h1_3", "1–3 ч"],
  ["h3_5", "3–5 ч"],
  ["h5plus", "5+ ч"],
];

const Q5_LABELS = {
  puzzles: "Задачи",
  games: "Партии",
  lessons: "Уроки",
};

const Q6 = [
  ["always", "Да"],
  ["stuck", "Только если застряну"],
  ["never", "Нет"],
];

const Q7 = [
  ["u10", "До 10"],
  ["age10_16", "10–16"],
  ["age17plus", "17+"],
];

const Q8 = [
  ["coach", "Тренер"],
  ["friends", "Друзья"],
  ["internet", "Интернет"],
  ["school", "Школа"],
];

const TIME_CONTROLS = [
  ["blitz", "Блиц"],
  ["rapid", "Рапид"],
  ["bullet", "Пуля"],
];

const INITIAL = {
  q1: null,
  q2: { platform: "lichess", login: "", time_control: "blitz", rating: "" },
  useQ2: false,
  q3: [],
  q4: null,
  q5: ["puzzles", "games", "lessons"],
  q6: null,
  q7: null,
  q8: null,
  consent: false,
  consentContact: "",
};

function RadioGroup({ value, options, onChange, name }) {
  return (
    <div className="onb-options" role="radiogroup">
      {options.map(([code, label]) => (
        <button
          key={code}
          type="button"
          role="radio"
          aria-checked={value === code}
          className={`onb-option ${value === code ? "onb-option--active" : ""}`}
          onClick={() => onChange(code)}
        >
          <span className="onb-option-mark" aria-hidden="true" />
          {label}
        </button>
      ))}
    </div>
  );
}

export default function OnboardingModal({ isOpen, onClose, onDone }) {
  const [a, setA] = useState(INITIAL);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!isOpen) return null;

  const set = (patch) => setA((prev) => ({ ...prev, ...patch }));

  const toggleQ3 = (code) =>
    setA((prev) => {
      const has = prev.q3.includes(code);
      if (has) return { ...prev, q3: prev.q3.filter((x) => x !== code) };
      if (prev.q3.length >= 2) return prev; // «до 2»
      return { ...prev, q3: [...prev.q3, code] };
    });

  const moveQ5 = (index, delta) =>
    setA((prev) => {
      const next = [...prev.q5];
      const target = index + delta;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return { ...prev, q5: next };
    });

  const validate = () => {
    if (!a.q1) return "Ответьте на вопрос 1";
    if (a.useQ2 && !a.q2.login.trim())
      return "Укажите логин на платформе или снимите галочку «Есть аккаунт»";
    if (!a.q3.length) return "Выберите хотя бы один ответ на вопрос 3";
    if (!a.q4) return "Ответьте на вопрос 4";
    if (!a.q6) return "Ответьте на вопрос 6";
    if (!a.q7) return "Ответьте на вопрос 7";
    if (a.q7 === "u10" && !a.consent)
      return "Для возраста до 10 лет нужно согласие родителя/опекуна";
    return "";
  };

  const submit = async () => {
    if (submitting) return;
    const problem = validate();
    if (problem) {
      setError(problem);
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const payload = {
        q1: a.q1,
        q2: a.useQ2
          ? {
              platform: a.q2.platform,
              login: a.q2.login.trim(),
              time_control: a.q2.time_control,
              rating: a.q2.rating === "" ? null : Number(a.q2.rating),
            }
          : null,
        q3: a.q3,
        q4: a.q4,
        q5: a.q5,
        q6: a.q6,
        q7: a.q7,
        q8: a.q8,
        parental_consent:
          a.q7 === "u10"
            ? { granted: true, contact: a.consentContact.trim() || null }
            : null,
      };
      const user = await saveOnboarding(payload);
      onDone?.(user);
      onClose();
      setA(INITIAL);
    } catch (e) {
      const d = e?.response?.data?.detail;
      if (Array.isArray(d)) {
        setError(
          d
            .map((x) => String(x?.msg || "").replace(/^Value error,\s*/i, ""))
            .filter(Boolean)
            .join("; "),
        );
      } else if (typeof d === "string") {
        setError(d);
      } else {
        setError("Не удалось сохранить анкету. Попробуйте ещё раз.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="onb-overlay" onClick={onClose}>
      <div
        className="onb-card"
        role="dialog"
        aria-modal="true"
        aria-label="Анкета оценки уровня игры"
        onClick={(e) => e.stopPropagation()}
      >
        <button className="onb-close" onClick={onClose} aria-label="Закрыть">
          ✕
        </button>

        <div className="onb-head">
          <span className="onb-brand">
            <span className="onb-brand-icon" aria-hidden="true">♛</span>
            <span className="onb-brand-name">GIGACHESS</span>
          </span>
          <h2 className="onb-title">Анкета оценки уровня игры</h2>
          <p className="onb-subtitle">
            Восемь коротких вопросов. От неё зависит, с чего начнётся обучение.
          </p>
          <div className="onb-divider" />
        </div>

        <div className="onb-body">
          <section className="onb-q">
            <h3 className="onb-q-title">
              1. Играли ли вы в шахматы раньше?
            </h3>
            <RadioGroup name="q1" value={a.q1} options={Q1} onChange={(v) => set({ q1: v })} />
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">
              2. Аккаунт + платформа + тип рейтинга?{" "}
              <span className="onb-optional">необязательно</span>
            </h3>
            <label className="onb-check">
              <input
                type="checkbox"
                checked={a.useQ2}
                onChange={(e) => set({ useQ2: e.target.checked })}
              />
              У меня есть аккаунт на Lichess или Chess.com
            </label>

            {a.useQ2 && (
              <div className="onb-q2">
                <div className="onb-row">
                  <span className="onb-row-label">Платформа</span>
                  <div className="onb-pills">
                    {[
                      ["lichess", "Lichess"],
                      ["chesscom", "Chess.com"],
                    ].map(([code, label]) => (
                      <button
                        key={code}
                        type="button"
                        className={`onb-pill ${a.q2.platform === code ? "onb-pill--active" : ""}`}
                        onClick={() => set({ q2: { ...a.q2, platform: code } })}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="onb-row">
                  <span className="onb-row-label">Логин</span>
                  <input
                    className="onb-input"
                    value={a.q2.login}
                    maxLength={64}
                    placeholder="ваш ник на платформе"
                    onChange={(e) => set({ q2: { ...a.q2, login: e.target.value } })}
                  />
                </div>

                <div className="onb-row">
                  <span className="onb-row-label">Тип рейтинга</span>
                  <div className="onb-pills">
                    {TIME_CONTROLS.map(([code, label]) => (
                      <button
                        key={code}
                        type="button"
                        className={`onb-pill ${a.q2.time_control === code ? "onb-pill--active" : ""}`}
                        onClick={() => set({ q2: { ...a.q2, time_control: code } })}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="onb-row">
                  <span className="onb-row-label">Рейтинг</span>
                  <input
                    className="onb-input onb-input--short"
                    type="number"
                    min="0"
                    max="4000"
                    value={a.q2.rating}
                    placeholder="необязательно"
                    onChange={(e) => set({ q2: { ...a.q2, rating: e.target.value } })}
                  />
                </div>
              </div>
            )}
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">
              3. Зачем вам шахматы?{" "}
              <span className="onb-optional">можно выбрать до 2</span>
            </h3>
            <div className="onb-options onb-options--wrap">
              {Q3.map(([code, label]) => {
                const active = a.q3.includes(code);
                const disabled = !active && a.q3.length >= 2;
                return (
                  <button
                    key={code}
                    type="button"
                    disabled={disabled}
                    aria-pressed={active}
                    className={`onb-option ${active ? "onb-option--active" : ""} ${disabled ? "onb-option--disabled" : ""}`}
                    onClick={() => toggleQ3(code)}
                  >
                    <span className="onb-option-mark" aria-hidden="true" />
                    {label}
                  </button>
                );
              })}
            </div>
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">4. Сколько готовы заниматься в неделю?</h3>
            <RadioGroup name="q4" value={a.q4} options={Q4} onChange={(v) => set({ q4: v })} />
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">
              5. Что интереснее?{" "}
              <span className="onb-optional">расставьте по порядку</span>
            </h3>
            <ol className="onb-rank">
              {a.q5.map((code, i) => (
                <li key={code} className="onb-rank-item">
                  <span className="onb-rank-pos">{i + 1}</span>
                  <span className="onb-rank-label">{Q5_LABELS[code]}</span>
                  <span className="onb-rank-arrows">
                    <button
                      type="button"
                      className="onb-rank-btn"
                      disabled={i === 0}
                      onClick={() => moveQ5(i, -1)}
                      aria-label={`Поднять «${Q5_LABELS[code]}»`}
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      className="onb-rank-btn"
                      disabled={i === a.q5.length - 1}
                      onClick={() => moveQ5(i, 1)}
                      aria-label={`Опустить «${Q5_LABELS[code]}»`}
                    >
                      ▼
                    </button>
                  </span>
                </li>
              ))}
            </ol>
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">6. Подсказки во время заданий?</h3>
            <RadioGroup name="q6" value={a.q6} options={Q6} onChange={(v) => set({ q6: v })} />
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">7. Возрастная группа?</h3>
            <RadioGroup name="q7" value={a.q7} options={Q7} onChange={(v) => set({ q7: v })} />

            {a.q7 === "u10" && (
              <div className="onb-consent">
                <label className="onb-check">
                  <input
                    type="checkbox"
                    checked={a.consent}
                    onChange={(e) => set({ consent: e.target.checked })}
                  />
                  Есть согласие родителя/опекуна на участие
                </label>
                <input
                  className="onb-input"
                  value={a.consentContact}
                  maxLength={255}
                  placeholder="контакт родителя (необязательно)"
                  onChange={(e) => set({ consentContact: e.target.value })}
                />
              </div>
            )}
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">
              8. Как узнали о платформе?{" "}
              <span className="onb-optional">необязательно</span>
            </h3>
            <RadioGroup name="q8" value={a.q8} options={Q8} onChange={(v) => set({ q8: v })} />
          </section>

          {error && <div className="onb-error">{error}</div>}
        </div>

        <div className="onb-footer">
          <button className="onb-skip" onClick={onClose}>
            Позже
          </button>
          <button
            className="onb-submit"
            onClick={submit}
            disabled={submitting}
          >
            {submitting ? "Сохраняем…" : "Сохранить"}
          </button>
        </div>
      </div>
    </div>
  );
}
