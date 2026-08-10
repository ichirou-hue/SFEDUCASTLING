import { useState } from "react";
import { fetchPlayerProfile } from "../api.js";

const PLATFORM_LABELS = {
  lichess: "Lichess",
  "chess.com": "Chess.com",
};

const PERF_LABELS = {
  bullet: "Пуля",
  blitz: "Блиц",
  rapid: "Рапид",
  classical: "Классика",
  daily: "Днев.",
};

export default function AuthModal({ isOpen, onClose }) {
  const [mode, setMode] = useState("register"); // "register" | "login"
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // Подтягивание профиля с Lichess / Chess.com
  const [platform, setPlatform] = useState("lichess");
  const [username, setUsername] = useState("");
  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");

  if (!isOpen) return null;

  const isRegister = mode === "register";

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

  return (
    <div className="auth-overlay" onClick={onClose}>
      <div className="auth-card" onClick={(e) => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose} aria-label="Закрыть">
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
              ? "Создайте аккаунт и импортируйте ваш шахматный профиль"
              : "С возвращением на доску"}
          </p>

          {isRegister && (
            <div className="auth-section">
              <div className="auth-section-head">
                <span className="auth-section-label">Импортировать профиль</span>
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
                    Партий: <b>{profile.counts?.all ?? "—"}</b>
                    <span className="auth-stat-sep" />
                    Побед: <b>{profile.counts?.wins ?? "—"}</b>
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
                  placeholder="Введите пароль"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
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
          </div>

          <button className="auth-submit">
            {isRegister ? "Зарегистрироваться" : "Войти"}
          </button>

          <p className="auth-footer">
            {isRegister ? (
              <>
                Уже есть аккаунт?{" "}
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setMode("login");
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
                    setMode("register");
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