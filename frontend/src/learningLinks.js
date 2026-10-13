import { Chess } from "chess.js";

export const COURSE_TOPIC_META = {
  pawn: { title: "Пешка", icon: "♙" },
  knight: { title: "Конь", icon: "♘" },
  bishop: { title: "Слон", icon: "♗" },
  rook: { title: "Ладья", icon: "♖" },
  queen: { title: "Ферзь", icon: "♕" },
  king: { title: "Король", icon: "♔" },
  "special-rules": { title: "Специальные правила", icon: "♙" },
  "check-mate-stalemate": { title: "Шах, мат и пат", icon: "♚" },
  endgames: { title: "Эндшпиль", icon: "♔" },
};

// Модули курса названы по фигурам: пазл относится к такому модулю, только
// если фигура реально участвует в решении (делает хотя бы один ход). Тактики
// (pin, skewer, fork...) выполняет любая фигура, а не обязательно «своя».
const PIECE_BY_MODULE = {
  pawn: "p",
  knight: "n",
  bishop: "b",
  rook: "r",
  queen: "q",
  king: "k",
};

function fenHasPiece(fen, symbol) {
  if (!fen) return true;
  const placement = fen.split(" ")[0].toLowerCase();
  return placement.includes(symbol);
}

function piecesUsedInSolution(fen, moves) {
  if (!fen || !moves) return null;
  const sequence = String(moves).replace(/\s+/g, "");
  if (sequence.length < 4) return null;
  let game;
  try {
    game = new Chess(fen);
  } catch {
    return null;
  }
  const used = new Set();
  let parsedAny = false;
  for (let i = 0; i + 3 < sequence.length; i += 4) {
    const from = sequence.slice(i, i + 2);
    const to = sequence.slice(i + 2, i + 4);
    const promotion = sequence[i + 4];
    const piece = game.get(from);
    if (!piece) break;
    try {
      game.move({ from, to, promotion });
    } catch {
      break;
    }
    used.add(piece.type);
    parsedAny = true;
  }
  return parsedAny ? used : null;
}

function modulePieceOk(puzzle, slug) {
  const symbol = PIECE_BY_MODULE[slug];
  if (!symbol) return true;
  const used = piecesUsedInSolution(puzzle?.fen, puzzle?.moves);
  if (used) return used.has(symbol);
  return fenHasPiece(puzzle?.fen, symbol);
}

const THEME_TO_COURSE_TOPIC = [
  ["enPassant", "special-rules"],
  ["promotion", "special-rules"],
  ["underPromotion", "special-rules"],

  ["pawnEndgame", "pawn"],
  ["advancedPawn", "pawn"],
  ["knightEndgame", "knight"],
  ["bishopEndgame", "bishop"],
  ["rookEndgame", "rook"],
  ["queenEndgame", "queen"],

  ["mateIn1", "check-mate-stalemate"],
  ["mateIn2", "check-mate-stalemate"],
  ["mateIn3", "check-mate-stalemate"],
  ["mateIn4", "check-mate-stalemate"],
  ["mateIn5", "check-mate-stalemate"],
  ["mate", "check-mate-stalemate"],
  ["smotheredMate", "check-mate-stalemate"],
  ["backRankMate", "check-mate-stalemate"],
  ["doubleCheck", "check-mate-stalemate"],

  ["fork", "knight"],
  ["pin", "bishop"],
  ["skewer", "bishop"],
  ["attraction", "queen"],
  ["sacrifice", "queen"],
  ["exposedKing", "king"],
  ["kingsideAttack", "king"],
  ["defensiveMove", "king"],

  ["opposition", "endgames"],
  ["passedPawn", "endgames"],
  ["endgame", "endgames"],
];

export function getCourseTopicMeta(slug) {
  if (!slug) return null;
  const meta = COURSE_TOPIC_META[slug];
  return meta ? { slug, ...meta } : null;
}

export function getCourseTopicForPuzzle(puzzle) {
  const themes = new Set(puzzle?.themes || []);
  const match = THEME_TO_COURSE_TOPIC.find(
    ([theme, slug]) => themes.has(theme) && modulePieceOk(puzzle, slug),
  );
  return match ? getCourseTopicMeta(match[1]) : null;
}
