import { useEffect, useRef } from "react";

export default function EvalBar({ diff = 0, flipped, height }) {
  const barRef = useRef(null);
  const MAX = 10;
  const signed = Number.isFinite(diff) ? diff : 0;
  const clamped = Math.max(-MAX, Math.min(MAX, signed));

  /*
   * Низ шкалы — та сторона, которая стоит внизу доски: её цвет и её
   * доля позиции. Когда доска развёрнута, цвета меняются местами —
   * тёмный мрамор (нижняя сторона) уходит вниз, бумага наверх.
   * Поэтому и цифра считается от нижней стороны.
   */
  const bottomAdv = flipped ? -clamped : clamped;
  const fillPct = 50 + (bottomAdv / MAX) * 50;

  const label = (() => {
    const v = Math.round(bottomAdv);
    if (v === 0) return "";
    return v > 0 ? `+${v}` : `${v}`;
  })();

  // Подсказка при наведении: шкала отражает материальный перевес.
  const title = label
    ? `Материальный перевес: ${label}`
    : "Материальный перевес: 0";

  /* Keep the bar locked to the playable board, independent of card chrome. */
  useEffect(() => {
    const bar = barRef.current;
    const card = document.querySelector(".board-wrapper");
    const board = document.querySelector(".eval-board-surface");

    if (!bar || !card || !board) return undefined;

    const measure = () => {
      const boardRect = board.getBoundingClientRect();
      const currentY =
        parseFloat(bar.style.getPropertyValue("--eval-shift-y")) || 0;
      const barTop = bar.getBoundingClientRect().top - currentY;
      const shiftY = Math.round(boardRect.top - barTop);

      bar.style.height = `${Math.round(boardRect.height)}px`;
      bar.style.setProperty("--eval-shift-y", `${shiftY}px`);
    };

    /*
     * Замер сразу, без requestAnimationFrame: ResizeObserver вызывает
     * нас уже после раскладки и до отрисовки, поэтому шкала едет в том
     * же кадре, что и доска (иначе она отстаёт на кадр и «догоняет»).
     */
    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(card);
    observer.observe(board);
    const stage = document.querySelector(".board-stage");
    if (stage) observer.observe(stage);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      bar.style.height = "";
      bar.style.removeProperty("--eval-shift-y");
    };
  }, []);

  return (
    <div
      ref={barRef}
      className={`eval-bar${
        flipped ? " eval-bar--bottom-black" : ""
      }`}
      style={{ height: height || undefined, "--eval-fill": `${fillPct}%` }}
      title={title}
      aria-label={title}
      role="img"
    >
      <div
        className="eval-bar-fill"
        style={{ height: `${fillPct}%` }}
      >
        {label && <span className="eval-bar-label">{label}</span>}
      </div>
    </div>
  );
}
