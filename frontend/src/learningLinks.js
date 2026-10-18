import { Chess } from "chess.js";

export const COURSE_TOPIC_META = {
  pieces: { title: "Фигуры: ходы и взятия", icon: "♟" },
  "king-safety": { title: "Шах, мат и безопасность короля", icon: "♚" },
  endgames: { title: "Эндшпиль", icon: "♔" },
  tactics: { title: "Тактические приёмы", icon: "♞" },
  openings: { title: "Дебюты", icon: "♜" },
};

// Раньше модули назывались по фигурам и пазл относился к модулю, только
// если фигура реально ходила в решении. После слияния модулей (10 -> 5)
// профильных фигур у тем нет — словарь пуст, modulePieceOk всегда true.
const PIECE_BY_MODULE = {};

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
  ["enPassant", "endgames"],
  ["promotion", "endgames"],
  ["underPromotion", "endgames"],

  ["pawnEndgame", "pieces"],
  ["advancedPawn", "pieces"],
  ["knightEndgame", "pieces"],
  ["bishopEndgame", "pieces"],
  ["rookEndgame", "pieces"],
  ["queenEndgame", "pieces"],

  ["mateIn1", "king-safety"],
  ["mateIn2", "king-safety"],
  ["mateIn3", "king-safety"],
  ["mateIn4", "king-safety"],
  ["mateIn5", "king-safety"],
  ["mate", "king-safety"],
  ["smotheredMate", "king-safety"],
  ["backRankMate", "king-safety"],

  ["fork", "tactics"],
  ["pin", "tactics"],
  ["skewer", "tactics"],
  ["attraction", "tactics"],
  ["sacrifice", "tactics"],
  ["doubleCheck", "tactics"],

  ["opening", "openings"],

  ["exposedKing", "king-safety"],
  ["kingsideAttack", "king-safety"],
  ["defensiveMove", "king-safety"],

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
