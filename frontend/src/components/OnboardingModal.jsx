import { useEffect, useRef, useState } from "react";
import { linkChessAccount, saveOnboarding } from "../api.js";
import { formatApiError } from "../apiErrors.js";
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

const INITIAL = {
  q1: null,
  // Q2 — только привязка: платформа и логин для поиска профиля.
  // Рейтинг и тип контроля в форму не вводятся — берутся из проверенной
  // привязки в submit. Руками присвоенный рейтинг изменить нельзя.
  q2: { platform: "lichess", login: "" },
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
  // Мягкая привязка внешнего аккаунта (Q2): ищет публичный профиль через
  // POST /api/chess-profile/link и подставляет проверенный рейтинг в форму.
  // Непривязанная отправка по-прежнему допустима — рейтинг тогда идёт на веру.
  const [linking, setLinking] = useState(false);
  const [linkError, setLinkError] = useState("");
  const [linked, setLinked] = useState(null);

  // Хуки перетаскивания Q5 — строго до раннего return ниже:
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

  // Смена платформы/логина инвалидирует предыдущую привязку.
  const setQ2 = (patch) => {
    setLinked(null);
    setLinkError("");
    setA((prev) => ({ ...prev, q2: { ...prev.q2, ...patch } }));
  };

  const linkExternalAccount = async () => {
    const login = a.q2.login.trim();
    if (!login) {
      setLinkError("Укажите логин на платформе");
      return;
    }
    if (linking) return;
    setLinking(true);
    setLinkError("");
    try {
      const data = await linkChessAccount(login, a.q2.platform);
      setLinked(data);
      // Ничего в форму не пишем: рейтинг и тип контроля берутся только из
      // проверенной привязки в момент отправки (см. submit). Руками рейтинг
      // изменить нельзя - только перепривязать другой аккаунт.
    } catch (e) {
      setLinked(null);
      setLinkError(
        e?.response?.data?.detail || e?.message || "Не удалось найти профиль",
      );
    } finally {
      setLinking(false);
    }
  };

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

  // Замер Y-координат строк Q5 до перестановки; сама FLIP-анимация —
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
    if (a.useQ2 && !a.q2.login.trim())
      return "Укажите логин на платформе или снимите галочку «Есть аккаунт»";
    // Q2 — только через привязку: рейтинг всегда проверенный, руками не вводится.
    if (a.useQ2 && !linked?.account)
      return "Привяжите аккаунт кнопкой «Найти и привязать аккаунт» или снимите галочку «Есть аккаунт»";
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
      // Q2 собирается только из проверенной привязки: логин — канонический
      // с платформы, рейтинг и тип контроля — из профиля. Руками рейтинг
      // не вводится. Нет рейтинга в профиле — шлём null, бэкенд откатится к Q1.
      const linkedAccount = linked?.account || {};
      const verifiedTimeControl = ["blitz", "rapid", "bullet"].includes(
        linkedAccount.rating_type,
      )
        ? linkedAccount.rating_type
        : "blitz";
      const payload = {
        q1: a.q1,
        q2: a.useQ2
          ? {
              platform: linkedAccount.platform || a.q2.platform,
              login: (linkedAccount.username || a.q2.login).trim(),
              time_control: verifiedTimeControl,
              rating: linkedAccount.rating ?? null,
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
      if (a.useQ2) {
        await linkChessAccount(a.q2.login.trim(), a.q2.platform);
      }
      const user = await saveOnboarding(payload);
      onDone?.(user);
      onClose();
      setA(INITIAL);
      setLinked(null);
      setLinkError("");
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
                onChange={(e) => {
                  setLinked(null);
                  setLinkError("");
                  set({ useQ2: e.target.checked });
                }}
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
                        onClick={() => setQ2({ platform: code })}
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
                    onChange={(e) => setQ2({ login: e.target.value })}
                  />
                </div>

                <button
                  type="button"
                  className="onb-link-btn"
                  onClick={linkExternalAccount}
                  disabled={linking || !a.q2.login.trim()}
                >
                  {linking ? "Проверяем профиль…" : "Найти и привязать аккаунт"}
                </button>

                {linkError && <div className="onb-link-error">{linkError}</div>}

                {linked?.account && (
                  <div
                    className={`onb-linked ${
                      linked.account.rating_usable ? "onb-linked--ok" : "onb-linked--warn"
                    }`}
                  >
                    <div className="onb-linked-head">
                      <strong>
                        {linked.account.platform === "chesscom" ? "Chess.com" : "Lichess"}
                        {" · "}
                        {linked.account.username}
                      </strong>
                      <span className="onb-linked-badge">Привязан</span>
                    </div>
                    {linked.account.rating != null ? (
                      <p className="onb-linked-rating">
                        Рейтинг для оценки: <strong>{linked.account.rating}</strong>
                        {linked.account.rating_type ? ` (${linked.account.rating_type})` : ""}
                        {linked.account.games ? ` · ${linked.account.games} партий` : ""}
                      </p>
                    ) : (
                      <p className="onb-linked-rating">
                        У профиля пока нет рейтинга в блиц/рапид/пулю.
                      </p>
                    )}
                    {linked.account.rating_usable ? (
                      <p className="onb-linked-note">
                        Проверенный рейтинг подставлен в анкету и будет
                        использован для стартовой оценки. Изменить его вручную
                        нельзя - только перепривязать другой аккаунт.
                      </p>
                    ) : (
                      <p className="onb-linked-note">
                        Аккаунт привязан, но его рейтинг не прошёл критерии
                        надёжности - для оценки возьмём ответ на вопрос 1.
                      </p>
                    )}
                    <p className="onb-linked-note onb-linked-note--dim">
                      Мягкая привязка по публичному профилю: без OAuth владение
                      аккаунтом не подтверждено.
                    </p>
                  </div>
                )}
              </div>
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
