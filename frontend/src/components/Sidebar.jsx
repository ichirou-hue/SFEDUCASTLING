import { NavLink } from "react-router-dom";

export default function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}
      <div className={`sidebar ${isOpen ? "sidebar--open" : ""}`}>
        <div className="sidebar-header">
          <span className="sidebar-title">Меню</span>
          <button className="sidebar-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <nav className="sidebar-nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "sidebar-link--active" : ""}`
            }
            onClick={onClose}
          >
            <span className="sidebar-icon">♟</span>
            Основная страница
          </NavLink>
          <NavLink
            to="/puzzles"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "sidebar-link--active" : ""}`
            }
            onClick={onClose}
          >
            <span className="sidebar-icon">♜</span>
            Шахматные задачи
          </NavLink>
          <NavLink
            to="/training"
            className={({ isActive }) =>
              `sidebar-link ${isActive ? "sidebar-link--active" : ""}`
            }
            onClick={onClose}
          >
            <span className="sidebar-icon">♘</span>
            Обучение
          </NavLink>
        </nav>
      </div>
    </>
  );
}
