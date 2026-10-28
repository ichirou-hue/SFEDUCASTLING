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

  /*
   * Шкала стоит посередине между панелью истории и левым краем доски.
   * Свободное место в строке зависит от ширины окна и от того, свёрнут
   * ли чат (доска внутри своей колонки центрируется), поэтому позицию
   * считаем по живым замерам. Двигаем через transform: раскладку строки
   * это не меняет, и замеры не «плывут» от собственного сдвига.
   */
  useEffect(() => {
    const bar = barRef.current;
    const section = document.querySelector(".board-section");
    const card = document.querySelector(".board-wrapper");

    if (!bar || !section || !card) return undefined;

    const measure = () => {
      const sectionRect = section.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      const shift = Math.max(
        0,
        Math.round((cardRect.left - sectionRect.left) / 2)
      );
      bar.style.setProperty("--eval-shift", `${shift}px`);
    };

    /*
     * Замер сразу, без requestAnimationFrame: ResizeObserver вызывает
     * нас уже после раскладки и до отрисовки, поэтому шкала едет в том
     * же кадре, что и доска (иначе она отстаёт на кадр и «догоняет»).
     */
    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(section);
    observer.observe(card);
    window.addEventListener("resize", measure);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      bar.style.removeProperty("--eval-shift");
    };
  }, []);

  return (
    <div
      ref={barRef}
      className={`eval-bar${
        flipped ? " eval-bar--bottom-black" : ""
      }`}
      style={{ height: height || undefined }}
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
