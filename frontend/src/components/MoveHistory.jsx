import { useState, useEffect, useRef } from "react";

export default function MoveHistory({
  moveHistory,
  positionSnapshots,
  viewIndex,
  isViewMode,
  onNavigate,
  maiaRating = 1700,
  userRating = "?",
}) {
  const listRef = useRef(null);

  // Скролл теперь будет следить не только за новыми ходами, но и за переключением активного хода
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [moveHistory, viewIndex]);

  return (
    <div className="move-history-panel">
      <h3 className="history-title">История ходов</h3>
      <div className="title-separator"></div>

      {/* Блок профиля */}
      <div className="player-info">
        <div className="player-avatar">
          <img src="/user-icon.svg" alt="User" />
          <span className="player-name">
            Вы <small>{userRating}</small>
          </span>
        </div>
        <span className="vs-text">VS</span>
        <div className="bot-avatar">
          <span className="bot-name">
            Maia 2<br />
            <small>{maiaRating}</small>
          </span>
          <img src="/bot-icon.svg" alt="Bot" />
        </div>
      </div>

      {/* Список ходов */}
      <div className="move-list" ref={listRef}>
        {moveHistory.map((pair, i) => {
          // Логика активной зеленой строки (зависит от индекса текущего просмотра)
          const isActive = viewIndex - 1 === i;

          return (
            <div className={`move-pair ${isActive ? "active" : ""}`} key={i}>
              <span className="num">{pair.num}.</span>
              <span className="move-w">{pair.w}</span>
              <span className="move-b">{pair.b || ""}</span>
            </div>
          );
        })}
      </div>

      {/* Навигация */}
      <div className="nav-controls">
        <button onClick={() => onNavigate("prev")} className="icon-btn">
          <img src="/arrow-left.svg" alt="Назад" />
        </button>
        <span className="move-counter">
          {viewIndex}/{moveHistory.length}
        </span>
        <button onClick={() => onNavigate("next")} className="icon-btn">
          <img src="/arrow-right.svg" alt="Вперёд" />
        </button>
      </div>

      {/* Нижние кнопки */}
      <div className="action-buttons">
        <button className="btn-import">Импорт</button>
        <input type="text" className="input-fen" placeholder="Введите FEN..." />
      </div>
    </div>
  );
}
