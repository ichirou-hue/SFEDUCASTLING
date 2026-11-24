import { useEffect, useRef } from "react";

/*
 * Выбор фигуры при превращении пешки.
 *
 * Общий диалог для страницы задач и страницы игры: react-chessboard
 * показывает свой выбор только при перетаскивании, а при ходе кликами
 * пешка молча становилась ферзём. Одна разметка на оба хода решает и это,
 * и разнобой во внешнем виде.
 */
const PIECE_OPTIONS = [
  ["q", "Ферзь"],
  ["r", "Ладья"],
  ["b", "Слон"],
  ["n", "Конь"],
];

const PIECE_IMAGES = {
  wQ: "/pieces/white_queen.svg",
  wR: "/pieces/white_rook.svg",
  wB: "/pieces/white_bishop.svg",
  wN: "/pieces/white_knight.svg",
  bQ: "/pieces/black_queen.svg",
  bR: "/pieces/black_rook.svg",
  bB: "/pieces/black_bishop.svg",
  bN: "/pieces/black_knight.svg",
};

export default function PromotionDialog({ color = "w", onSelect, onCancel }) {
  const firstButtonRef = useRef(null);
  const cancelRef = useRef(onCancel);

  useEffect(() => {
    cancelRef.current = onCancel;
  }, [onCancel]);

  /*
   * Фокус сразу в диалоге: клавиатурный игрок не должен искать его
   * табом, а Escape - привычный выход.
   */
  useEffect(() => {
    firstButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      cancelRef.current?.();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  const side = color === "b" ? "b" : "w";

  return (
    <div className="promotion-overlay">
      <section
        className="promotion-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="promotion-title"
      >
        <h2 id="promotion-title">Во что превратить пешку?</h2>
        <p>Выберите фигуру, которой хотите продолжить.</p>
        <div className="promotion-options">
          {PIECE_OPTIONS.map(([piece, label], index) => (
            <button
              type="button"
              key={piece}
              ref={index === 0 ? firstButtonRef : null}
              onClick={() => onSelect?.(piece)}
              aria-label={`Превратить пешку в фигуру: ${label.toLowerCase()}`}
            >
              <img src={PIECE_IMAGES[`${side}${piece.toUpperCase()}`]} alt="" draggable={false} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <button type="button" className="promotion-cancel" onClick={() => onCancel?.()}>
          Отмена
        </button>
      </section>
    </div>
  );
}
