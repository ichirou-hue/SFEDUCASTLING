import { useEffect, useRef } from "react";

/*
 * Шкала оценки движка: оценка Stockfish переводится в вероятность
 * победы по кривой Lichess. Разница в пешку почти не двигает шкалу,
 * а разгромная позиция прижимает её к краю.
 *
 * Материального режима нет: материал движок уже учитывает в своей
 * оценке, отдельный пересчёт только дублировал бы его.
 */

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
  engineEval = null,
  flipped,
  height,
}) {
  const barRef = useRef(null);

  /*
   * Низ шкалы - та сторона, которая стоит внизу доски: её цвет и её
   * доля позиции. Когда доска развёрнута, цвета меняются местами -
   * тёмный мрамор (нижняя сторона) уходит вниз, бумага наверх.
   * Поэтому и цифра считается от нижней стороны.
   */
  const whiteAdv = engineAdvantage(engineEval);
  const hasEval = whiteAdv !== null;
  const bottomAdv = hasEval ? (flipped ? -whiteAdv : whiteAdv) : 0;
  const fillPct = hasEval ? 50 + bottomAdv * 50 : 50;
  const label = hasEval ? engineLabel(engineEval, flipped ? -1 : 1) : "";

  const title = hasEval
    ? `Оценка движка: ${engineLabelTitle(engineEval) || "0,0"}`
    : "Оценка движка недоступна";

  /*
   * Шкала держится за игровое поле, а не за карточку целиком: сверху в
   * карточке есть панель противника, снизу - кнопки, и по карточке
   * капсула съезжала бы относительно доски. Высоту и вертикальный
   * сдвиг считаем по живым замерам: они меняются от ширины окна и от
   * того, свёрнут ли чат.
   */
  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return undefined;

    const measure = () => {
      // Доска может пересоздаваться (react-chessboard меняет key при
      // смене позиции), поэтому берём актуальный узел на каждый замер,
      // а не тот, что был при монтировании шкалы.
      const board = document.querySelector(".eval-board-surface");
      if (!board) return false;

      const boardRect = board.getBoundingClientRect();
      // Пока доска ещё не разложена, её высота равна нулю: не схлопываем
      // шкалу, дождёмся следующего замера.
      if (!boardRect.height) return false;

      bar.style.height = `${Math.round(boardRect.height)}px`;

      // На узких экранах шкала лежит над доской, и CSS выключает сдвиг
      // (transform: none). В этом режиме не копим --eval-shift-y: иначе
      // при каждом замере он растёт и скачком применится на широком.
      if (getComputedStyle(bar).transform === "none") {
        bar.style.setProperty("--eval-shift-y", "0px");
        return true;
      }

      const currentY =
        parseFloat(bar.style.getPropertyValue("--eval-shift-y")) || 0;
      const barTop = bar.getBoundingClientRect().top - currentY;
      const shiftY = Math.round(boardRect.top - barTop);

      bar.style.setProperty("--eval-shift-y", `${shiftY}px`);
      return true;
    };

    /*
     * Шкала и доска - соседние компоненты, и к моменту первого эффекта
     * карточки доски может ещё не быть в DOM. Поэтому ждём её в кадрах:
     * как только появились и карточка, и поле с ненулевой высотой,
     * ставим замер и подписываемся на изменения.
     */
    let observer = null;
    let frame = 0;
    let timer = 0;

    const setup = () => {
      const card = document.querySelector(".board-wrapper");
      if (card && !observer) {
        observer = new ResizeObserver(measure);
        observer.observe(card);
        const stage = document.querySelector(".board-stage");
        if (stage) observer.observe(stage);
        window.addEventListener("resize", measure);
      }

      // Замер сразу: ResizeObserver вызывает нас уже после раскладки и
      // до отрисовки, поэтому шкала едет в том же кадре, что и доска
      // (иначе она отстаёт на кадр и «догоняет»).
      if ((!card || !measure()) && frame < 40) {
        frame += 1;
        // setTimeout, а не requestAnimationFrame: в фоновой вкладке
        // кадры не идут, а таймеры работают.
        timer = setTimeout(setup, 50);
      }
    };
    setup();

    return () => {
      clearTimeout(timer);
      if (observer) observer.disconnect();
      window.removeEventListener("resize", measure);
      bar.style.height = "";
      bar.style.removeProperty("--eval-shift-y");
    };
  }, []);

  return (
    <div
      ref={barRef}
      className={`eval-bar eval-bar--engine${
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
        aria-hidden="true"
      >
        {label && <span className="eval-bar-label">{label}</span>}
      </div>
    </div>
  );
}
