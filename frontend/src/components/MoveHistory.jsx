import { useState, useEffect, useRef } from "react";

export default function MoveHistory({
  moveHistory,
  positionSnapshots,
  viewIndex,
  isViewMode,
  onNavigate,
  onImport,
  maiaRating = 1700,
  userRating = "?",
}) {
  const listRef = useRef(null);
  const fileInputRef = useRef(null);

  // Скролл теперь будет следить не только за новыми ходами, но и за переключением активного хода
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [moveHistory, viewIndex]);
  // Если moveHistory вдруг пришел как undefined, считаем его пустым массивом
  const safeHistory = moveHistory || [];
  // Если мы отмотали в самое начало (viewIndex === -1 И мы в режиме просмотра), то показываем 0.
  // Иначе показываем текущий ход или максимум.
  const currentHalfMove =
    viewIndex === -1 && !isViewMode ? safeHistory.length : viewIndex;
  const currentFullMove = Math.ceil(currentHalfMove / 2);
  const totalFullMoves = Math.ceil(safeHistory.length / 2);

  const isPrevDisabled = safeHistory.length === 0 || viewIndex === 0;
  const isNextDisabled = safeHistory.length === 0 || viewIndex === -1;

  // НОВОЕ: Группируем плоский массив строк в красивые пары (белые/черные)
  const movePairs = [];
  for (let i = 0; i < safeHistory.length; i += 2) {
    movePairs.push({
      num: Math.floor(i / 2) + 1,
      w: safeHistory[i],
      b: safeHistory[i + 1] || null,
      wIndex: i, // Индекс полухода белых
      bIndex: i + 1, // Индекс полухода черных
    });
  }
  // Функция для программного клика по скрытому загрузчику файлов
  const handleImportClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  return (
    <div className="move-history-panel">
      <h3 className="history-title">История ходов</h3>
      <div className="title-separator"></div>

      {/* Блок профиля */}
      <div className="player-info">
        {/* Левая часть: Пользователь */}
        <div className="player-avatar">
          <img src="\user-icon.svg" alt="Вы" />
        </div>
        <div className="player-details">
          <span className="player-rating">{userRating || "1500"}</span>
          <span className="player-name">Вы</span>
        </div>

        <div className="vs-text">VS</div>

        {/* Правая часть: Бот Maia */}
        <div className="bot-details">
          <span className="bot-name">Maia 2</span>
          {/* Сюда мы передаем проп с рейтингом от доски */}
          <span className="bot-rating">{maiaRating}</span>
        </div>
        <div className="bot-avatar">
          <img src="\bot-icon.svg" alt="Maia 2" />
        </div>
      </div>

      {/* Список ходов */}
      <div className="move-list" ref={listRef}>
        {movePairs.map((item, index) => (
          <div key={index} className="move-pair">
            <span className="num">{item.num}.</span>

            {/* Ход белых */}
            <span
              className={`move-w ${viewIndex === item.wIndex || (viewIndex === -1 && item.wIndex === safeHistory.length - 1) ? "active" : ""}`}
              onClick={() => onNavigate(item.wIndex)}
              style={{ cursor: "pointer" }}
            >
              {item.w}
            </span>

            {/* Ход черных */}
            {item.b && (
              <span
                className={`move-b ${viewIndex === item.bIndex || (viewIndex === -1 && item.bIndex === safeHistory.length - 1) ? "active" : ""}`}
                onClick={() => onNavigate(item.bIndex)}
                style={{ cursor: "pointer" }}
              >
                {item.b}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Навигация */}
      <div className="nav-row">
        <button
          onClick={() => onNavigate("prev")}
          className="arrow-btn"
          aria-label="Назад"
          disabled={isPrevDisabled}
          style={{
            opacity: isPrevDisabled ? 0.3 : 1,
            cursor: isPrevDisabled ? "default" : "pointer",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="55"
            height="44"
            viewBox="0 0 55 44"
            fill="none"
          >
            <rect
              width="55"
              height="44"
              rx="22"
              fill="#76B451"
              fill-opacity="0.3"
            />
            <path
              d="M22.651 39.1787C24.2588 38.9233 25.5233 37.8556 25.9964 36.3483C26.3649 35.1843 26.189 33.9073 25.5149 32.8647C25.3474 32.6051 24.6775 31.9101 22.1402 29.3727L18.9748 26.199L32.7459 26.1864L46.5212 26.178L46.8687 26.0817C48.5937 25.6212 49.7828 24.252 49.9755 22.5228C50.1848 20.6261 49.0376 18.8466 47.2036 18.2185C46.5086 17.9799 47.2665 17.9924 32.5365 17.9924H18.9706L22.136 14.8229C24.1039 12.8508 25.3683 11.5528 25.4688 11.3979C26.5826 9.72725 26.3816 7.58349 24.979 6.18503C24.2002 5.40206 23.233 5.00429 22.1108 5.0001C21.4032 4.99592 20.8757 5.11734 20.2476 5.43137C19.5442 5.78726 19.9713 5.36856 11.7731 13.617L5.7396 19.6882L5.52188 20.0859C4.70122 21.61 4.85614 23.3811 5.93221 24.7084C6.06619 24.8717 9.16459 27.9952 12.824 31.6463C18.5561 37.3699 19.5233 38.3204 19.8331 38.5256C20.4905 38.9526 21.152 39.1662 21.9434 39.1955C22.1946 39.208 22.517 39.1997 22.651 39.1787Z"
              fill="#3E7451"
            />
          </svg>
        </button>

        <span className="move-counter">
          {currentFullMove} / {totalFullMoves}
        </span>

        <button
          onClick={() => onNavigate("next")}
          className="arrow-btn"
          aria-label="Вперёд"
          disabled={isNextDisabled}
          style={{
            opacity: isNextDisabled ? 0.3 : 1,
            cursor: isNextDisabled ? "default" : "pointer",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="55"
            height="44"
            viewBox="0 0 55 44"
            fill="none"
          >
            <rect
              width="55"
              height="44"
              rx="22"
              fill="#76B451"
              fillOpacity="0.3"
            />
            <path
              d="M32.3491 5.02243C30.7412 5.27784 29.4768 6.34553 29.0036 7.85286C28.6352 9.01685 28.811 10.2939 29.4851 11.3365C29.6526 11.5961 30.3225 12.2911 32.8599 14.8284L36.0253 18.0022L22.2542 18.0148L8.47887 18.0231L8.13135 18.1194C6.40629 18.58 5.21718 19.9492 5.02458 21.6784C4.81523 23.5751 5.96247 25.3546 7.79638 25.9826C8.49143 26.2213 7.73358 26.2087 22.4635 26.2087H36.0295L32.8641 29.3783C30.8962 31.3504 29.6317 32.6484 29.5312 32.8033C28.4175 34.4739 28.6184 36.6177 30.0211 38.0161C30.7999 38.7991 31.7671 39.1969 32.8892 39.2011C33.5968 39.2053 34.1244 39.0838 34.7524 38.7698C35.4558 38.4139 35.0288 38.8326 43.2269 30.5842L49.2604 24.513L49.4782 24.1152C50.2988 22.5912 50.1439 20.8201 49.0678 19.4928C48.9338 19.3295 45.8354 16.206 42.176 12.5549C36.444 6.83122 35.4768 5.88077 35.1669 5.67561C34.5096 5.24853 33.848 5.03499 33.0567 5.00568C32.8055 4.99312 32.483 5.0015 32.3491 5.02243Z"
              fill="#3E7451"
            />
          </svg>
        </button>
      </div>

      {/* Нижние кнопки */}
      <div className="action-buttons">
        {/* Скрытый инпут для файлов */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          accept=".fen,.pgn,.png"
          onChange={onImport}
        />

        {/* Красивая кнопка, которая триггерит инпут */}
        <button className="btn-import" onClick={handleImportClick}>
          Импорт
        </button>
      </div>
    </div>
  );
}
