import { useState } from "react";

export default function AuthModal({ isOpen, onClose }) {
  const [mode, setMode] = useState("register"); // "register" | "login"
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  if (!isOpen) return null;

  const isRegister = mode === "register";

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          ✕
        </button>

        <div className="modal-icon">♛</div>
        <h2 className="modal-title">{isRegister ? "Регистрация" : "Вход"}</h2>

        <div className="modal-divider" />

        <label className="modal-label">Логин</label>
        <div className="modal-input-wrapper">
          <span className="modal-input-icon">👤</span>
          <input
            type="text"
            className="modal-input"
            placeholder="Введите логин"
            value={login}
            onChange={(e) => setLogin(e.target.value)}
          />
        </div>

        <label className="modal-label">Пароль</label>
        <div className="modal-input-wrapper">
          <span className="modal-input-icon">🔒</span>
          <input
            type={showPassword ? "text" : "password"}
            className="modal-input"
            placeholder="Введите пароль"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button
            className="modal-eye"
            onClick={() => setShowPassword((v) => !v)}
            type="button"
          >
            {showPassword ? "👁" : "👁‍🗨"}
          </button>
        </div>

        <button className="modal-submit">
          {isRegister ? "Зарегистрироваться" : "Войти"}
        </button>

        <p className="modal-footer">
          {isRegister ? (
            <>
              Уже есть аккаунт?{" "}
              <a href="#" onClick={() => setMode("login")}>
                Войти
              </a>
            </>
          ) : (
            <>
              Нет аккаунта?{" "}
              <a href="#" onClick={() => setMode("register")}>
                Зарегистрироваться
              </a>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
