import { useEffect, useRef } from "react";

export default function EvalBar({ diff = 0, flipped, height }) {
  const barRef = useRef(null);

  /*
   * Первые 10 очков перевеса занимают 40% шкалы - так обычная игра
   * читается хорошо. Дальше шкала сжимается: 10..39 очков добавляют
   * оставшиеся 10%, поэтому подавляющий перевес тоже виден.
   * 39 - стоимость полного комплекта без короля.
   */
  const BASE_MAX = 10;
  const EXTENDED_MAX = 39;
  const signed = Number.isFinite(diff) ? diff : 0;

  /*
   * Низ шкалы - та сторона, которая стоит внизу доски: её цвет и её
   * доля позиции. Когда доска развёрнута, цвета меняются местами -
   * тёмный мрамор (нижняя сторона) уходит вниз, бумага наверх.
   * Поэтому и цифра считается от нижней стороны.
   */
  const bottomAdv = flipped ? -signed : signed;
  const magnitude = Math.min(Math.abs(bottomAdv), EXTENDED_MAX);
  const basePart = Math.min(magnitude, BASE_MAX) / BASE_MAX;
  const extraPart =
    Math.max(0, magnitude - BASE_MAX) / (EXTENDED_MAX - BASE_MAX);
  const fillPct =
    50 + Math.sign(bottomAdv) * (basePart * 40 + extraPart * 10);

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
   * Шкала держится рядом с левым краем доски, а по вертикали - ровно
   * напротив доски, а не карточки целиком: сверху в карточке есть
   * панель противника, снизу - кнопки.
   *
   * Горизонтальный зазор до доски постоянный: flex-gap строки,
   * margin-right шкалы и левый padding колонки доски. Когда чат
   * сворачивается, доска уезжает вправо, и шкала едет вместе с ней,
   * а не остаётся посередине освободившегося просвета. Позицию
   * считаем по живым замерам и двигаем через transform: раскладку
   * строки это не меняет, и замеры не «плывут» от собственного сдвига.
   */
  useEffect(() => {
    const bar = barRef.current;
    const section = document.querySelector(".board-section");
    const card = document.querySelector(".board-wrapper");

    if (!bar || !section || !card) return undefined;

    const measure = () => {
      const cardRect = card.getBoundingClientRect();

      /*
       * Натуральный правый край шкалы: текущий rect включает transform,
       * поэтому вычитаем уже применённый сдвиг.
       */
      const currentX =
        parseFloat(bar.style.getPropertyValue("--eval-shift")) || 0;
      const naturalRight = bar.getBoundingClientRect().right - currentX;

      const mainStyles = getComputedStyle(section.parentElement);
      const barStyles = getComputedStyle(bar);
      const sectionStyles = getComputedStyle(section);

      const clearance =
        (parseFloat(mainStyles.columnGap) ||
          parseFloat(mainStyles.gap) ||
          0) +
        (parseFloat(barStyles.marginRight) || 0) +
        (parseFloat(sectionStyles.paddingLeft) || 0);

      // Сдвиг по горизонтали: постоянный зазор до левого края доски.
      const shiftX = Math.max(
        0,
        Math.round(cardRect.left - naturalRight - clearance)
      );

      /*
       * Сдвиг по вертикали: шкала той же высоты, что и доска, поэтому
       * достаточно опустить её до верхнего края доски - тогда середины
       * совпадут. Положение доски внутри карточки берём у самой доски.
       * Текущий transform вычитаем, чтобы замер не зависел от него.
       */
      const board = card.querySelector('div[style*="perspective"]');
      const currentY =
        parseFloat(bar.style.getPropertyValue("--eval-shift-y")) || 0;
      const barTop = bar.getBoundingClientRect().top - currentY;
      const shiftY = board
        ? Math.round(board.getBoundingClientRect().top - barTop)
        : 0;

      bar.style.setProperty("--eval-shift", `${shiftX}px`);
      bar.style.setProperty("--eval-shift-y", `${shiftY}px`);
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
