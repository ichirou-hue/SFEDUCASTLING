import { useState, useEffect, useRef } from 'react'

export default function MoveHistory({ moveHistory, positionSnapshots, viewIndex, isViewMode, onNavigate }) {
  const listRef = useRef(null)

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [moveHistory])

  return (
    <div className="middle-section">
      <h3>История ходов</h3>
      <div className="move-list" ref={listRef}>
        {moveHistory.map((pair, i) => (
          <div className="move-pair" key={i}>
            <span className="num">{pair.num}.</span>
            <span>{pair.w}</span>
            {pair.b && <span>{pair.b}</span>}
          </div>
        ))}
      </div>
      <div className="nav-controls">
        <button onClick={() => onNavigate('first')} title="Начало">⏮</button>
        <button onClick={() => onNavigate('prev')} title="Назад (←)">◀</button>
        <button onClick={() => onNavigate('next')} title="Вперёд (→)">▶</button>
        <button onClick={() => onNavigate('last')} title="Конец">⏭</button>
      </div>
    </div>
  )
}
