export const COURSE_TOPIC_META = {
  pawn: { title: "Пешка", icon: "♙" },
  knight: { title: "Конь", icon: "♘" },
  bishop: { title: "Слон", icon: "♗" },
  rook: { title: "Ладья", icon: "♖" },
  queen: { title: "Ферзь", icon: "♕" },
  king: { title: "Король", icon: "♔" },
  "special-rules": { title: "Специальные правила", icon: "♙" },
  "check-mate-stalemate": { title: "Шах, мат и пат", icon: "♚" },
};

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
];

export function getCourseTopicMeta(slug) {
  if (!slug) return null;
  const meta = COURSE_TOPIC_META[slug];
  return meta ? { slug, ...meta } : null;
}

export function getCourseTopicForPuzzle(puzzle) {
  const themes = new Set(puzzle?.themes || []);
  const match = THEME_TO_COURSE_TOPIC.find(([theme]) => themes.has(theme));
  return match ? getCourseTopicMeta(match[1]) : null;
}
