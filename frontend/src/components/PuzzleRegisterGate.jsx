import "./PuzzleRegisterGate.css";

// Гейт регистрации на предпоследнем ходу в пазле для незарегистрированных.
// Отказаться нельзя: нет кнопки закрытия, нет onClick на оверлее,
// Escape-слушателей в проекте нет. Единственный путь — зарегистрироваться.
export default function PuzzleRegisterGate({ open, onRegister }) {
  if (!open) return null;

  return (
    <div className="prg-overlay">
      <div
        className="prg-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="prg-title"
        aria-describedby="prg-text"
      >
        <div className="prg-brand">
          <span className="prg-brand-icon" aria-hidden="true">
            ♛
          </span>
          <span className="prg-brand-name">GIGACHESS</span>
        </div>
        <h2 className="prg-title" id="prg-title">
          Остался один ход
        </h2>
        <p className="prg-text" id="prg-text">
          Решение не сохранится без аккаунта. Позиция останется на доске —
          зарегистрируйтесь и сделайте последний ход.
        </p>
        <button className="prg-submit" onClick={onRegister} autoFocus>
          Зарегистрироваться
        </button>
      </div>
    </div>
  );
}
