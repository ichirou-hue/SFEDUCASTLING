import "./PuzzleRegisterGate.css";
import "./PuzzlePreviewGate.css";

// Гейт пробного доступа до анкеты. Показывается на последнем ходу третьей
// решённой за сегодня задачи (и если дневной лимит уже выбран). Зарегистрированный,
// но не прошедший анкету пользователь решает не больше трёх лёгких задач в день,
// дальше его просят заполнить анкету. В отличие от PuzzleRegisterGate закрыть
// можно («Позже»), но доску завершить нельзя — это мягкий онбординг-напоминатель.
export default function PuzzlePreviewGate({ onOpenOnboarding, onLater }) {
  return (
    <div className="prg-overlay">
      <div
        className="prg-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ppg-title"
        aria-describedby="ppg-text"
      >
        <div className="prg-brand">
          <span className="prg-brand-icon" aria-hidden="true">
            ♛
          </span>
          <span className="prg-brand-name">GIGACHESS</span>
        </div>
        <h2 className="prg-title" id="ppg-title">
          Нужно пройти анкету
        </h2>
        <p className="prg-text" id="ppg-text">
          Чтобы продолжить решать задачи, заполните короткую анкету. Это займёт
          минуту. Откроются все задачи и персональные тренировки. Или
          возвращайтесь завтра.
        </p>
        <div className="ppg-actions">
          <button className="prg-submit" onClick={onOpenOnboarding} autoFocus>
            Пройти анкету
          </button>
          <button type="button" className="ppg-later" onClick={onLater}>
            Позже
          </button>
        </div>
      </div>
    </div>
  );
}
