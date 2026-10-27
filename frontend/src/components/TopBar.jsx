import { Link, NavLink } from "react-router-dom";
import { logout as apiLogout } from "../api.js";
import MiniProfilePopover from "./MiniProfilePopover.jsx";

export default function TopBar({ user, onUserChange, onRegister, onOpenOnboarding }) {
  const handleLogout = async () => {
    try {
      await apiLogout();
    } finally {
      window.location.reload();
    }
  };

  // Плашка видна, пока анкета не заполнена.
  const needsOnboarding = Boolean(user) && !user?.onboarding_completed;
  const navItems = [
    { to: "/", label: "Главная", mobileLabel: "Главная", end: true, icon: "M3 10.5 12 3l9 7.5M5.5 9v11h13V9M9 20v-6h6v6" },
    { to: "/play", label: "Игра", mobileLabel: "Игра", icon: "M7 21c-2.2 0-4-1.8-4-4 0-1.9 1.3-3.5 3-3.9V11a3 3 0 1 1 3-3h2a3 3 0 1 1 3 3v2.1c1.7.4 3 2 3 3.9 0 2.2-1.8 4-4 4H7Z" },
    { to: "/puzzles", label: "Задачи", mobileLabel: "Задачи", icon: "M4 4h6v5a2 2 0 1 0 4 0V4h6v6h-5a2 2 0 1 0 0 4h5v6h-6v-5a2 2 0 1 0-4 0v5H4v-6h5a2 2 0 1 0 0-4H4V4Z" },
    { to: "/training", label: "Обучение", mobileLabel: "Учёба", icon: "M3 5.5A2.5 2.5 0 0 1 5.5 3H12v17H5.5A2.5 2.5 0 0 0 3 22V5.5ZM21 5.5A2.5 2.5 0 0 0 18.5 3H12v17h6.5A2.5 2.5 0 0 1 21 22V5.5Z" },
  ];
  if (user?.is_admin === true || user?.role === "admin") {
    navItems.push({ to: "/admin", label: "Администрирование", mobileLabel: "Админ", icon: "M12 3v2m0 14v2m9-9h-2M5 12H3m15.36-6.36-1.42 1.42M7.06 16.94l-1.42 1.42m12.72 0-1.42-1.42M7.06 7.06 5.64 5.64M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z" });
  }

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <Link to="/" className="logo-wrapper" aria-label="GigaChess: главная">
          <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
        </Link>
      </div>
      <nav className="top-nav" aria-label="Основная навигация">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
            }
          >
            <svg className="mobile-nav-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d={item.icon} />
            </svg>
            <span className="top-nav-label">{item.label}</span>
            <span className="top-nav-label-mobile">{item.mobileLabel}</span>
          </NavLink>
        ))}
      </nav>
      {user ? (
        <div className="user-box">
          <Link
            to="/gigacoins"
            className="topbar-gigacoins"
            title="Твой баланс GigaCoin"
            aria-label={`Баланс: ${user.gigacoins ?? 0} GigaCoin`}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="10" cy="10" r="7.2" />
              <path d="M12.3 7.5c-.4-.7-1.1-1-2.2-1-1.2 0-2 .5-2 1.4 0 2.2 4.8.8 4.8 3.3 0 1-.9 1.7-2.5 1.7-1.2 0-2.1-.4-2.6-1.2M10 4.9v1.4m0 7v1.7" />
            </svg>
            <strong>{Number(user.gigacoins || 0).toLocaleString("ru-RU")}</strong>
            <small>GigaCoin</small>
          </Link>
          {needsOnboarding && (
            <button
              className="onb-badge"
              onClick={() => onOpenOnboarding?.()}
              title="Пройти онбординг-анкету"
            >
              Заполните анкету
            </button>
          )}
          <MiniProfilePopover user={user} onUserChange={onUserChange} onLogout={handleLogout} />
          <button
            type="button"
            className="reg-btn reg-btn--ghost top-bar-logout-compact"
            onClick={handleLogout}
            aria-label="Выйти из аккаунта"
          >
            Выйти
          </button>
        </div>
      ) : (
        <button className="reg-btn" onClick={onRegister}>Регистрация</button>
      )}
    </div>
  );
}
