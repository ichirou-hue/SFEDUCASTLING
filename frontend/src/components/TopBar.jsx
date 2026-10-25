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

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <Link to="/" className="logo-wrapper" aria-label="GigaChess — главная">
          <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
        </Link>
      </div>
      <nav className="top-nav" aria-label="Основная навигация">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
          }
        >
          Главная
        </NavLink>
        <NavLink
          to="/play"
          className={({ isActive }) =>
            `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
          }
        >
          Игра
        </NavLink>
        <NavLink
          to="/puzzles"
          className={({ isActive }) =>
            `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
          }
        >
          Задачи
        </NavLink>
        <NavLink
          to="/training"
          className={({ isActive }) =>
            `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
          }
        >
          Обучение
        </NavLink>
        {user?.is_admin === true || user?.role === "admin" ? (
          <NavLink
            to="/admin"
            className={({ isActive }) =>
              `top-nav-link ${isActive ? "top-nav-link--active" : ""}`
            }
          >
            Администрирование
          </NavLink>
        ) : null}
      </nav>
      {user ? (
        <div className="user-box">
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
        </div>
      ) : (
        <button className="reg-btn" onClick={onRegister}>Регистрация</button>
      )}
    </div>
  );
}
