import { logout as apiLogout } from "../api.js";

export default function TopBar({ user, onRegister, onMenuClick }) {
  const handleLogout = async () => {
    try {
      await apiLogout();
    } finally {
      window.location.reload();
    }
  };

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <button className="hamburger-btn" onClick={onMenuClick} title="Меню">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <div className="logo-wrapper">
          <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
        </div>
      </div>
      {user ? (
        <div className="user-box">
          <span className="user-name" title={user.email || ""}>
            {user.is_admin ? "♛ " : "👤 "}
            {user.login}
          </span>
          <button className="reg-btn reg-btn--ghost" onClick={handleLogout}>
            Выйти
          </button>
        </div>
      ) : (
        <button className="reg-btn" onClick={onRegister}>Регистрация</button>
      )}
    </div>
  );
}
