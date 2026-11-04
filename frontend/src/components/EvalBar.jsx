import { useEffect, useRef } from "react";

/*
 * Шкала двух режимов:
 *
 * - "material" - материальный перевес в пешках (как раньше);
 * - "engine"   - оценка Stockfish, переведённая в вероятность победы
 *                по кривой Lichess: разница в пешку почти не двигает
 *                шкалу, а разгромная позиция прижимает её к краю.
 *
 * Переключатель живёт внутри капсулы: отдельная кнопка рядом занимала
 * бы место в раскладке строки и сдвигала доску.
 */

// Материальный перевес, при котором шкала упирается в край.
const MAX_MATERIAL = 10;

// Оценка в сантипешках -> доля позиции от -1 (чёрные выигрывают) до 1.
function engineAdvantage(engineEval) {
  if (!engineEval) return null;
  if (engineEval.type === "mate") {
    const value = Number(engineEval.value) || 0;
    if (value === 0) return 0;
    return value > 0 ? 1 : -1;
  }
  const cp = Number(engineEval.value) || 0;
  return 2 / (1 + Math.exp(-0.00368208 * cp)) - 1;
}

// Подпись на шкале: "+1,2", "-0,4", "#" (мат на доске).
// Для мата, который ещё не на доске, показываем предельную оценку
// ±10,0: точное число ходов на коротком поиске недостоверно, а число
// не спорит с историей ходов. Расшифровка - в подсказке.
function engineLabel(engineEval, bottomSign) {
  if (!engineEval) return "";

  if (engineEval.type === "mate") {
    const mate = Number(engineEval.value) || 0;
    if (mate === 0) return "#";
    const sign = mate > 0 ? 1 : -1;
    return sign * bottomSign > 0 ? "+10,0" : "-10,0";
  }

  const pawns = ((Number(engineEval.value) || 0) / 100) * bottomSign;
  if (Math.abs(pawns) < 0.05) return "0,0";
  const text = (Math.round(pawns * 10) / 10).toFixed(1).replace(".", ",");
  return pawns > 0 ? `+${text}` : text;
}

// Склонение «ход» в подсказке: 1 ход, 2 хода, 5 ходов.
function movesWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;

  if (mod10 === 1 && mod100 !== 11) return "ход";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "хода";
  return "ходов";
}

// Расшифровка оценки для подсказки и aria-label: у мата нужна сторона.
function engineLabelTitle(engineEval) {
  if (!engineEval) return "";

  if (engineEval.type === "mate") {
    const mate = Number(engineEval.value) || 0;
    if (mate === 0) return "мат на доске";
    const side = mate > 0 ? "белых" : "чёрных";
    const count = Math.abs(mate);
    return `мат в ${count} ${movesWord(count)}, ставит сторона ${side}`;
  }

  return engineLabel(engineEval, 1);
}

export default function EvalBar({
  diff = 0,
  engineEval = null,
  mode = "material",
  onToggleMode,
  flipped,
  height,
}) {
  const barRef = useRef(null);

  // Выбор пользователя и фактический режим шкалы расходятся, пока не
  // пришла оценка движка: пустая капсула выглядела бы как сломанная,
  // поэтому до первого ответа показываем материал.
  const showEngine = mode === "engine" && Boolean(engineEval);
  const signed = Number.isFinite(diff) ? diff : 0;

  /*
   * Низ шкалы - та сторона, которая стоит внизу доски: её цвет и её
   * доля позиции. Когда доска развёрнута, цвета меняются местами -
   * тёмный мрамор (нижняя сторона) уходит вниз, бумага наверх.
   * Поэтому и цифра считается от нижней стороны.
   */
  let bottomAdv;
  let fillPct;
  let label;

  if (showEngine) {
    const whiteAdv = engineAdvantage(engineEval) ?? 0;
    bottomAdv = flipped ? -whiteAdv : whiteAdv;
    fillPct = 50 + bottomAdv * 50;
    label = engineLabel(engineEval, flipped ? -1 : 1);
  } else {
    const clamped = Math.max(-MAX_MATERIAL, Math.min(MAX_MATERIAL, signed));
    bottomAdv = flipped ? -clamped : clamped;
    fillPct = 50 + (bottomAdv / MAX_MATERIAL) * 50;
    const rounded = Math.round(bottomAdv);
    label = rounded === 0 ? "" : rounded > 0 ? `+${rounded}` : `${rounded}`;
  }

  const title = showEngine
    ? `Оценка движка: ${engineLabelTitle(engineEval) || "0,0"}`
    : mode === "engine"
      ? "Оценка движка недоступна - показан материальный перевес"
      : `Материальный перевес: ${label || "0"}`;

  /*
   * Шкала стоит посередине между панелью истории и левым краем доски
   * и по вертикали - ровно напротив доски, а не карточки целиком:
   * сверху в карточке есть панель противника, снизу - кнопки.
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

      // Сдвиг по горизонтали: середина просвета между историей и доской.
      const shiftX = Math.max(
        0,
        Math.round((cardRect.left - sectionRect.left) / 2)
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
      }${showEngine ? " eval-bar--engine" : ""}`}
      style={{ height: height || undefined, "--eval-fill": `${fillPct}%` }}
      title={title}
      aria-label={title}
      role="group"
    >
      {/*
       * Переключатель режима. Буква показывает, что на шкале сейчас:
       * «М» - материал, «Д» - движок.
       */}
      <button
        type="button"
        className="eval-bar-mode"
        onClick={onToggleMode}
        disabled={!onToggleMode}
        title={
          mode === "engine"
            ? "Показать материальный перевес"
            : "Показать оценку движка"
        }
        aria-label={
          mode === "engine"
            ? "Переключить шкалу на материальный перевес"
            : "Переключить шкалу на оценку движка"
        }
      >
        {mode === "engine" ? "Д" : "М"}
      </button>

      <div
        className="eval-bar-fill"
        style={{ height: `${fillPct}%` }}
        aria-hidden="true"
      >
        {label && <span className="eval-bar-label">{label}</span>}
      </div>
    </div>
  );
}
