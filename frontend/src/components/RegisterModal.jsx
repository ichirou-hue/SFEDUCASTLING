import { useEffect, useState } from "react";
import {
  fetchCurrentUser,
  fetchPlayerProfile,
  linkChessAccount,
  login as apiLogin,
  register as apiRegister,
} from "../api.js";
import { formatApiError } from "../apiErrors.js";
import ChessAccountsSection from "./ChessAccountsSection.jsx";

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

// Ссылка на документ внизу формы регистрации (открывается в новой вкладке).
// TODO: подставить реальный URL и подпись, когда документ будет приложен.
const REG_DOC_URL = "#";
const REG_DOC_LABEL = "Документ для ознакомления";

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
  const [authError, setAuthError] = useState("");

  // Шаг после успешной регистрации: привязка шахматного аккаунта.
  // Привязка требует авторизации, поэтому выполняется уже после создания
  // пользователя, тем же компонентом, что и на вкладке профиля.
  const [step, setStep] = useState("auth"); // "auth" | "link"
  const [createdUser, setCreatedUser] = useState(null);
  const [autoLinkNotice, setAutoLinkNotice] = useState("");
  const [autoLinkError, setAutoLinkError] = useState("");
  const [finishing, setFinishing] = useState(false);

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
    setStep("auth");
    setCreatedUser(null);
    setAutoLinkNotice("");
    setAutoLinkError("");
    setFinishing(false);
  }, [isOpen, initialMode, notice]);

  if (!isOpen) return null;

  const isRegister = mode === "register";

  // Регистрация / вход через бэкенд (задача 64)
  const handleSubmit = async () => {
    if (submitting) return;
    const nick = login.trim();
    if (!nick || !password) {
      setAuthError("Введите логин и пароль");
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

      // Импортированный профиль сразу привязываем: только привязка внешнего
      // аккаунта даёт стартовую оценку выше ограничения самооценки.
      // Ошибку не глотаем — показываем её на шаге привязки, аккаунт уже создан.
      const importedNick = username.trim();
      let linkedNotice = "";
      let linkError = "";
      if (importedNick) {
        setSubmitPhase("link");
        try {
          const linked = await linkChessAccount(importedNick, normalizePlatform(platform));
          const linkedName = linked?.account?.username || importedNick;
          linkedNotice = `Аккаунт ${linkedName} привязан.`;
          if (!linked?.account?.rating_trusted) {
            linkedNotice +=
              " Владение пока не подтверждено: пройдите OAuth, чтобы рейтинг учитывался в стартовой оценке.";
          }
        } catch (e) {
          const detail = e?.response?.data?.detail;
          linkError = `Аккаунт создан, но привязать профиль не удалось: ${
            detail
              ? formatApiError(detail)
              : "проверьте ник и попробуйте снова."
          }`;
        }
      }

      setCreatedUser(user);
      setAutoLinkNotice(linkedNotice);
      setAutoLinkError(linkError);
      setStep("link");
    } catch (e) {
      // Ошибки валидации бэкенда переводим на русский (см. apiErrors.js).
      setAuthError(formatApiError(e?.response?.data?.detail));
    } finally {
      setSubmitting(false);
      setSubmitPhase("");
    }
  };

  // Завершает регистрацию уже после шага привязки: подтягиваем свежего
  // пользователя (OAuth мог обновить рейтинг) и отдаём его приложению.
  const finishRegistration = async () => {
    if (finishing) return;
    setFinishing(true);
    try {
      const fresh = await fetchCurrentUser();
      onSuccess?.(fresh || createdUser);
    } catch {
      onSuccess?.(createdUser);
    } finally {
      onClose();
      setFinishing(false);
    }
  };

  // После регистрации токены уже сохранены, поэтому закрытие модалки не
  // должно оставлять приложение в состоянии «гость».
  const handleClose = () => {
    if (submitting) return;
    if (step === "link" && createdUser) {
      finishRegistration();
      return;
    }
    onClose();
  };

  const handleSwitchMode = () => {
    setMode(isRegister ? "login" : "register");
    setAuthError("");
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

  if (step === "link" && createdUser) {
    return (
      <div className="auth-overlay" onClick={handleClose}>
        <div className="auth-card auth-card--link" onClick={(e) => e.stopPropagation()}>
          <button className="auth-close" onClick={handleClose} aria-label="Закрыть">
            ✕
          </button>

          <div className="auth-link-panel">
            <div className="auth-link-head">
              <div className="auth-brand">
                <span className="auth-brand-icon">♛</span>
                <span className="auth-brand-name">GIGACHESS</span>
              </div>
              <h2 className="auth-title">Аккаунт создан</h2>
              <p className="auth-subtitle">
                Привяжите шахматный профиль — это необязательно. Подтверждённый
                рейтинг Lichess или Chess.com используется для стартовой оценки.
              </p>
            </div>

            {autoLinkNotice && (
              <div className="auth-link-notice auth-link-notice--ok" role="status">
                {autoLinkNotice}
              </div>
            )}
            {autoLinkError && (
              <div className="auth-link-notice auth-link-notice--error" role="alert">
                {autoLinkError}
              </div>
            )}

            <ChessAccountsSection
              user={createdUser}
              embedded
              initialPlatform={username.trim() ? normalizePlatform(platform) : undefined}
              initialUsername={username.trim()}
            />

            <div className="auth-link-actions">
              <button
                className="auth-submit"
                onClick={finishRegistration}
                disabled={finishing}
              >
                {finishing ? "Сохраняем…" : "Продолжить"}
              </button>
              <p className="auth-link-hint">
                Привязать аккаунт можно позже в разделе «Профиль».
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
    <div className="auth-overlay" onClick={handleClose}>
      <div className="auth-card" onClick={(e) => e.stopPropagation()}>
        <button className="auth-close" onClick={handleClose} aria-label="Закрыть">
          ✕
        </button>

        {/* Левая панель — бренд и импорт профиля */}
        <div className="auth-side auth-side--brand">
          <div className="auth-brand">
            <span className="auth-brand-icon">♛</span>
            <span className="auth-brand-name">GIGACHESS</span>
          </div>
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

        {/* Правая панель — форма */}
        <div className="auth-side auth-side--form">
          <div className="auth-fields">
            <label className="auth-label">
              <span className="auth-label-text">{isRegister ? "Логин" : "Логин или e-mail"}</span>
              <div className="auth-input-wrapper">
                <span className="auth-input-icon">👤</span>
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
                <span className="auth-input-icon">🔒</span>
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
                  aria-label="Показать пароль"
                >
                  {showPassword ? "👁" : "👁‍🗨"}
                </button>
              </div>
            </label>

            {authError && <p className="auth-error">{authError}</p>}
          </div>

          <button
            className="auth-submit"
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitLabel}
          </button>

          {isRegister && (
            <a
              className="auth-reg-doc"
              href={REG_DOC_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              {REG_DOC_LABEL}
            </a>
          )}

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
