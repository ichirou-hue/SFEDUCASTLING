import { Chess } from "chess.js";

/*
 * Описание шаха и мата для визуального слоя.
 *
 * Работаем по отображаемому FEN, а не по «живому» объекту партии:
 * при листании истории и загрузке позиции доска может показывать
 * не текущую партию, а произвольный кадр.
 */

const NO_CHECK = Object.freeze({
  inCheck: false,
  isCheckmate: false,
  kingSquare: null,
});

const CHECK_KING_STYLE = Object.freeze({
  backgroundImage:
    "radial-gradient(ellipse at center, rgba(220, 30, 30, 0.88) 0%, rgba(220, 30, 30, 0.72) 55%, rgba(220, 30, 30, 0.42) 100%)",
  backgroundSize: "100% 100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
});

const CHECKMATE_KING_STYLE = Object.freeze({
  backgroundImage:
    "radial-gradient(ellipse at center, rgba(170, 0, 0, 0.92) 0%, rgba(175, 10, 10, 0.78) 55%, rgba(160, 0, 0, 0.5) 100%)",
  backgroundSize: "100% 100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
});

export function findKingSquare(game, color) {
  if (!game) return null;

  try {
    for (const row of game.board()) {
      for (const cell of row) {
        if (cell && cell.type === "k" && cell.color === color) {
          return cell.square;
        }
      }
    }
  } catch {
    return null;
  }

  return null;
}

export function getCheckState(position) {
  if (!position) return NO_CHECK;

  let game;

  try {
    game = typeof position === "string" ? new Chess(position) : position;
  } catch {
    return NO_CHECK;
  }

  try {
    /* В шахматной позиции мат всегда означает шах, поэтому одной
       проверки inCheck() достаточно, чтобы отсечь спокойные позиции. */
    if (!game.inCheck()) return NO_CHECK;

    return {
      inCheck: true,
      isCheckmate: game.isCheckmate(),
      kingSquare: findKingSquare(game, game.turn()),
    };
  } catch {
    return NO_CHECK;
  }
}

export function getKingCheckStyle(isCheckmate) {
  return isCheckmate ? CHECKMATE_KING_STYLE : CHECK_KING_STYLE;
}
