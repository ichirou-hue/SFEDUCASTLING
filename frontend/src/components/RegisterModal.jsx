import { useEffect, useState } from "react";
import {
  fetchCurrentUser,
  fetchPlayerProfile,
  linkChessAccount,
  login as apiLogin,
  register as apiRegister,
} from "../api.js";
import { formatApiError } from "../apiErrors.js";
import { LEGAL_DOCS } from "../legalDocs.js";

const PLATFORM_LABELS = {
  lichess: "Lichess",
  "chess.com": "Chess.com",
  chesscom: "Chess.com",
};

const PERF_LABELS = {
  bullet: "Пуля",
  blitz: "Блиц",
  rapid: "Рапид",
  classical: "Классика",
  daily: "Днев.",
};

function normalizePlatform(value) {
  return value === "chess.com" || value === "chesscom" ? "chesscom" : "lichess";
}

export default function AuthModal({ isOpen, onClose, onSuccess, initialMode = "register", notice = "" }) {
  const [mode, setMode] = useState("register"); // "register" | "login"
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitPhase, setSubmitPhase] = useState(""); // "" | "register" | "link"
  const [cooldown, setCooldown] = useState(0);
  const [authError, setAuthError] = useState("");
  // Обязательное согласие с документами при регистрации (только UI).
  const [consentAccepted, setConsentAccepted] = useState(false);

  // Подтягивание профиля с Lichess / Chess.com (предпросмотр до регистрации)
  const [platform, setPlatform] = useState("lichess");
  const [username, setUsername] = useState("");
  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");

  useEffect(() => {
    if (!isOpen) return;
    setMode(initialMode);
    setAuthError(notice || "");
    setPassword("");
    setConsentAccepted(false);
    setCooldown(0);
  }, [isOpen, initialMode, notice]);

  // Обратный отсчёт после 429: сервер вернул Retry-After, и повторять
  // запрос раньше срока бессмысленно — кнопка остаётся заблокированной.
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((value) => value - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  if (!isOpen) return null;

  const isRegister = mode === "register";

  // Регистрация / вход через бэкенд (задача 64)
  const handleSubmit = async () => {
    if (submitting || cooldown > 0) return;
    const nick = login.trim();
    if (!nick || !password) {
      setAuthError("Введите логин и пароль");
      return;
    }
    if (isRegister && !consentAccepted) {
      setAuthError("Подтвердите согласие с условиями использования и документами по персональным данным");
      return;
    }
    setSubmitting(true);
    setAuthError("");
    try {
      if (!isRegister) {
        const user = await apiLogin(nick, password);
        onSuccess?.(user);
        onClose();
        return;
      }

      setSubmitPhase("register");
      const user = await apiRegister(nick, password);

      // После создания пользователя привязываем выбранный при регистрации
      // профиль. Повторный экран привязки не нужен.
      const importedNick = username.trim();
      if (importedNick) {
        setSubmitPhase("link");
        try {
          await linkChessAccount(importedNick, normalizePlatform(platform));
        } catch {
          // Регистрация завершена; повторная привязка доступна в профиле.
        }
      }

      const fresh = await fetchCurrentUser().catch(() => null);
      onSuccess?.(fresh || user);
      onClose();
    } catch (e) {
      // Ошибки валидации бэкенда переводим на русский (см. apiErrors.js).
      const status = e?.response?.status;
      const retryAfter = Number(
        e?.response?.headers?.["retry-after"] ??
          e?.response?.headers?.["Retry-After"],
      );
      if (status === 429 && Number.isFinite(retryAfter) && retryAfter > 0) {
        setCooldown(Math.ceil(retryAfter));
      }
      setAuthError(formatApiError(e?.response?.data?.detail));
    } finally {
      setSubmitting(false);
      setSubmitPhase("");
    }
  };

  const handleSwitchMode = () => {
    setMode(isRegister ? "login" : "register");
    setAuthError("");
    // У режимов разные лимиты (регистрация не троттлится), поэтому снимаем паузу.
    setCooldown(0);
  };

  const handleFetchProfile = async () => {
    const nick = username.trim();
    if (!nick) return;
    setProfileLoading(true);
    setProfileError("");
    setProfile(null);
    try {
      const data = await fetchPlayerProfile(nick, platform);
      if (data?.error) {
        setProfileError(data.error);
      } else {
        setProfile(data);
      }
    } catch (e) {
      setProfileError("Не удалось получить профиль. Проверьте ник.");
    } finally {
      setProfileLoading(false);
    }
  };

  const ratings = profile?.perfs
    ? Object.entries(profile.perfs).filter(([, v]) => v && v.rating)
    : [];


  const submitLabel = submitting
    ? submitPhase === "link"
      ? "Привязываем профиль…"
      : isRegister
        ? "Создаём аккаунт…"
        : "Входим…"
    : isRegister
      ? "Зарегистрироваться"
      : "Войти";

  return (
    <div className="auth-overlay" onClick={onClose}>
      <div className="auth-card" onClick={(e) => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose} aria-label="Закрыть">
          ✕
        </button>

        {/* Левая панель - заголовок и импорт профиля */}
        <div className="auth-side auth-side--brand">
          <h2 className="auth-title">
            {isRegister ? "Регистрация" : "Вход в аккаунт"}
          </h2>
          <p className="auth-subtitle">
            {isRegister
              ? "Создайте аккаунт и при желании привяжите шахматный профиль. Привязка произойдёт сразу после регистрации."
              : "С возвращением на доску"}
          </p>

          {isRegister && (
            <div className="auth-section">
              <div className="auth-section-head">
                <span className="auth-section-label">Шахматный аккаунт</span>
                <span className="auth-optional">необязательно</span>
              </div>

              <div className="auth-providers">
                <button
                  className={`auth-provider ${platform === "lichess" ? "active" : ""}`}
                  onClick={() => {
                    setPlatform("lichess");
                    setProfile(null);
                  }}
                >
                  <span className="auth-provider-glyph">♞</span>
                  Lichess
                </button>
                <button
                  className={`auth-provider ${platform === "chess.com" ? "active" : ""}`}
                  onClick={() => {
                    setPlatform("chess.com");
                    setProfile(null);
                  }}
                >
                  <span className="auth-provider-glyph">♕</span>
                  Chess.com
                </button>
              </div>

              <div className="auth-import-row">
                <input
                  type="text"
                  className="auth-input auth-import-input"
                  placeholder="Ваш ник на платформе"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleFetchProfile()}
                />
                <button
                  className="auth-import-btn"
                  onClick={handleFetchProfile}
                  disabled={profileLoading || !username.trim()}
                >
                  {profileLoading ? "Ищем…" : "Найти"}
                </button>
              </div>

              {profileError && <p className="auth-error">{profileError}</p>}

              {profile && (
                <div className="auth-profile-card">
                  <div className="auth-profile-top">
                    {profile.avatar ? (
                      <img className="auth-profile-img" src={profile.avatar} alt="" />
                    ) : (
                      <div className="auth-profile-img auth-profile-img--fallback">♟</div>
                    )}
                    <div className="auth-profile-meta">
                      <div className="auth-profile-name">
                        {profile.title && (
                          <span className="auth-profile-title">{profile.title}</span>
                        )}
                        {profile.username}
                      </div>
                      <div className="auth-profile-sub">
                        {profile.name ||
                          `${PLATFORM_LABELS[profile.platform] || profile.platform} · рейтинги`}
                      </div>
                    </div>
                  </div>
                  <div className="auth-perfs">
                    {ratings.map(([key, v]) => (
                      <div className="auth-perf" key={key}>
                        <span className="auth-perf-label">{PERF_LABELS[key] || key}</span>
                        <span className="auth-perf-value">{v.rating}</span>
                      </div>
                    ))}
                  </div>
                  <div className="auth-profile-stats">
                    Партий: <b>{profile.counts?.all ?? 0}</b>
                    <span className="auth-stat-sep" />
                    Побед: <b>{profile.counts?.wins ?? 0}</b>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Правая панель - форма */}
        <div className="auth-side auth-side--form">
          <div className="auth-fields">
            <label className="auth-label">
              <span className="auth-label-text">{isRegister ? "Логин" : "Логин или e-mail"}</span>
              <div className="auth-input-wrapper">
                <input
                  type="text"
                  className="auth-input"
                  placeholder="Введите логин"
                  value={login}
                  onChange={(e) => setLogin(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                />
              </div>
            </label>

            <label className="auth-label">
              <span className="auth-label-text">Пароль</span>
              <div className="auth-input-wrapper">
                <input
                  type={showPassword ? "text" : "password"}
                  className="auth-input"
                  placeholder={isRegister ? "Минимум 8 символов" : "Введите пароль"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                />
                <button
                  className="auth-eye"
                  onClick={() => setShowPassword((v) => !v)}
                  type="button"
                  aria-label={showPassword ? "Скрыть пароль" : "Показать пароль"}
                >
                  {showPassword ? (
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </label>

            {authError && <p className="auth-error">{authError}</p>}
          </div>

          {isRegister && (
            <div className="auth-consent">
              <input
                id="auth-consent"
                type="checkbox"
                className="auth-consent-input"
                checked={consentAccepted}
                onChange={(e) => {
                  setConsentAccepted(e.target.checked);
                  if (e.target.checked) setAuthError("");
                }}
              />
              <span className="auth-consent-text">
                <label htmlFor="auth-consent">Я принимаю</label>{" "}
                <a
                  href={LEGAL_DOCS.terms.href}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {LEGAL_DOCS.terms.label}
                </a>
                ,{" "}
                <a
                  href={LEGAL_DOCS.privacy.href}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {LEGAL_DOCS.privacy.label.replaceAll(" ", "\u00A0")}
                </a>{" "}
                и даю{" "}
                <a
                  href={LEGAL_DOCS.consent.href}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {LEGAL_DOCS.consent.label}
                </a>
              </span>
            </div>
          )}

          <button
            className="auth-submit"
            onClick={handleSubmit}
            disabled={
              submitting || cooldown > 0 || (isRegister && !consentAccepted)
            }
          >
            {cooldown > 0 ? `Подождите ${cooldown} с` : submitLabel}
          </button>

          <p className="auth-footer">
            {isRegister ? (
              <>
                Уже есть аккаунт?{" "}
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    handleSwitchMode();
                  }}
                >
                  Войти
                </a>
              </>
            ) : (
              <>
                Нет аккаунта?{" "}
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    handleSwitchMode();
                  }}
                >
                  Зарегистрироваться
                </a>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
