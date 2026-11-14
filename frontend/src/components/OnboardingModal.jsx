import { useEffect, useRef, useState } from "react";
import {
  fetchLinkedChessAccounts,
  saveOnboarding,
} from "../api.js";
import { formatApiError } from "../apiErrors.js";
import "./OnboardingModal.css";

// Дословно из «Анкеты оценки уровня игры», раздел 1 / приложение Д.1.
// Ключи - те же коды, что уходит на бэкенд (тексты могут меняться, коды - нет).

const Q1 = [
  ["never", "Никогда"],
  ["know_moves", "Знаю ходы"],
  ["sometimes", "Играю иногда"],
  ["regular", "Регулярно"],
  ["tournaments", "Турниры"],
];

// Отдельный вопрос про правила: ответ "no" уводит новичка в модуль
// «Шахматная азбука» сразу после анкеты, минуя входной тест.
const KNOWS_RULES = [
  ["yes", "Да"],
  ["no", "Нет"],
];

const Q3 = [
  ["family", "Друзья и семья"],
  ["online_rating", "Онлайн-рейтинг"],
  ["tournaments", "Турниры"],
  ["child", "Занимается ребёнок"],
];

const Q4 = [
  ["lt1", "< 1 ч"],
  ["h1_3", "1-3 ч"],
  ["h3_5", "3-5 ч"],
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
  ["age10_16", "10-16"],
  ["age17plus", "17+"],
];

const Q8 = [
  ["coach", "Тренер"],
  ["friends", "Друзья"],
  ["internet", "Интернет"],
  ["school", "Школа"],
];

const INITIAL = {
  q1: null,
  knowsRules: null,
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
  const [linkedAccount, setLinkedAccount] = useState(null);

  // Анкета использует уже привязанный профиль без повторного запроса привязки.
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setLinkedAccount(null);
    fetchLinkedChessAccounts()
      .then((accounts) => {
        if (!cancelled) setLinkedAccount(accounts?.items?.find((item) => item.rating_usable) || null);
      })
      .catch(() => { if (!cancelled) setLinkedAccount(null); });
    return () => { cancelled = true; };
  }, [isOpen]);

  // Хуки перетаскивания Q5 - строго до раннего return ниже:
  // условные хуки ломают порядок вызовов React и роняют модалку.
  const rankRowRefs = useRef({});
  const flipFromRef = useRef(null);
  const dragFromRef = useRef(null);
  const [dragIndex, setDragIndex] = useState(null);
  const [dropIndex, setDropIndex] = useState(null);

  // FLIP-анимация перестановки строк Q5: после рендера сдвигаем каждую
  // строку на измеренную разницу и плавно отпускаем в ноль.
  useEffect(() => {
    const from = flipFromRef.current;
    if (!from) return;
    flipFromRef.current = null;
    for (const code of a.q5) {
      const el = rankRowRefs.current[code];
      if (!el || from[code] === undefined) continue;
      const dy = from[code] - el.getBoundingClientRect().top;
      if (!dy) continue;
      el.style.transition = "none";
      el.style.transform = `translateY(${dy}px)`;
      requestAnimationFrame(() => {
        el.style.transition = "transform 0.28s ease";
        el.style.transform = "";
      });
    }
  }, [a.q5]);

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

  // Замер Y-координат строк Q5 до перестановки; сама FLIP-анимация -
  // в эффекте выше (хуки обязаны стоять до раннего return).
  const measureRankFlip = () => {
    const from = {};
    for (const code of a.q5) {
      const el = rankRowRefs.current[code];
      if (el) from[code] = el.getBoundingClientRect().top;
    }
    flipFromRef.current = from;
  };

  const moveQ5Step = (index, delta) => {
    const target = index + delta;
    if (target < 0 || target >= a.q5.length) return;
    measureRankFlip();
    moveQ5(index, delta);
  };

  const moveQ5To = (fromIndex, toIndex) => {
    if (
      fromIndex === null ||
      fromIndex === undefined ||
      fromIndex === toIndex ||
      toIndex < 0 ||
      toIndex >= a.q5.length
    ) {
      return;
    }
    measureRankFlip();
    setA((prev) => {
      const next = [...prev.q5];
      const [moved] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, moved);
      return { ...prev, q5: next };
    });
  };

  const validate = () => {
    if (!a.q1) return "Ответьте на вопрос 1";
    if (!a.knowsRules) return "Ответьте на вопрос 2";
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
      // Q2 собирается только из проверенной привязки: логин - канонический
      // с платформы, рейтинг и тип контроля - из профиля. Руками рейтинг
      // не вводится. Нет рейтинга в профиле - шлём null, бэкенд откатится к Q1.
      const account = linkedAccount || {};
      const verifiedTimeControl = ["blitz", "rapid", "bullet"].includes(
        account.rating_type,
      )
        ? account.rating_type
        : "blitz";
      const payload = {
        q1: a.q1,
        knows_rules: a.knowsRules,
        q2: linkedAccount
          ? {
              platform: account.platform,
              login: account.username,
              time_control: verifiedTimeControl,
              rating: account.rating ?? null,
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
      // Ошибки валидации бэкенда переводим на русский (см. apiErrors.js).
      setError(
        formatApiError(
          e?.response?.data?.detail,
          "Не удалось сохранить анкету. Попробуйте ещё раз.",
        ),
      );
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
            Девять коротких вопросов. От неё зависит, с чего начнётся обучение.
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
              2. Знаете ли вы правила игры в шахматы?
            </h3>
            <RadioGroup
              name="knowsRules"
              value={a.knowsRules}
              options={KNOWS_RULES}
              onChange={(v) => set({ knowsRules: v })}
            />
            {a.knowsRules === "no" && (
              <p className="onb-question-note">
                Назначим минимальный стартовый уровень — тест проходить не
                нужно. Начнём с модуля «Шахматная азбука»: доска, фигуры,
                шах и мат, специальные правила.
              </p>
            )}
          </section>

          <section className="onb-q">
            <h3 className="onb-q-title">
              3. Зачем вам шахматы?{" "}
              <span className="onb-optional">можно выбрать до 2</span>
            </h3>
            <div className="onb-options onb-options--wrap onb-options--multi">
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
                <li
                  key={code}
                  ref={(el) => {
                    if (el) rankRowRefs.current[code] = el;
                    else delete rankRowRefs.current[code];
                  }}
                  className={[
                    "onb-rank-item",
                    dragIndex === i ? "onb-rank-item--dragging" : "",
                    dropIndex === i && dragIndex !== i
                      ? "onb-rank-item--drop-target"
                      : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  draggable
                  onDragStart={(e) => {
                    dragFromRef.current = i;
                    setDragIndex(i);
                    e.dataTransfer.effectAllowed = "move";
                    try {
                      e.dataTransfer.setData("text/plain", String(i));
                    } catch {
                      // Firefox требует setData, остальные игнорируют.
                    }
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = "move";
                    if (dropIndex !== i) setDropIndex(i);
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    moveQ5To(dragFromRef.current, i);
                    dragFromRef.current = null;
                    setDragIndex(null);
                    setDropIndex(null);
                  }}
                  onDragEnd={() => {
                    dragFromRef.current = null;
                    setDragIndex(null);
                    setDropIndex(null);
                  }}
                >
                  <span className="onb-rank-pos">{i + 1}</span>
                  <span className="onb-rank-label">{Q5_LABELS[code]}</span>
                  <span className="onb-rank-arrows">
                    <button
                      type="button"
                      className="onb-rank-btn"
                      disabled={i === 0}
                      onClick={() => moveQ5Step(i, -1)}
                      aria-label={`Поднять «${Q5_LABELS[code]}»`}
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      className="onb-rank-btn"
                      disabled={i === a.q5.length - 1}
                      onClick={() => moveQ5Step(i, 1)}
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
