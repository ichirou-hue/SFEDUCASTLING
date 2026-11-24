/*
 * Превращение пешки в позиции.
 *
 * Возвращает цвет превращающейся пешки ("w"/"b"), если ход from->to
 * действительно превращение, иначе null. Проверка идёт по списку
 * легальных ходов chess.js: одного взгляда на горизонталь мало, потому
 * что ход может быть недопустим (например, король под шахом).
 */
export function promotionColorForMove(game, from, to) {
  if (!game || typeof game.get !== "function") return null;

  const piece = game.get(from);
  if (!piece || piece.type !== "p") return null;

  const moves = game.moves({ square: from, verbose: true }) || [];
  const isPromotion = moves.some((move) => move.to === to && move.promotion);

  return isPromotion ? piece.color : null;
}
