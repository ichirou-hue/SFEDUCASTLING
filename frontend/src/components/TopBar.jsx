import { logout as apiLogout } from "../api.js";

export default function TopBar({ user, onRegister }) {
  const handleLogout = async () => {
    try {
      await apiLogout();
    } finally {
      window.location.reload();
    }
  };

  return (
    <div className="top-bar">
      <div className="logo-wrapper">
        <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
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
