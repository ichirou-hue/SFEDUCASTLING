import { Chessboard as ReactChessboard } from "react-chessboard";

const PIECE_IMAGES = {
  wK: "/pieces/white_king.svg",
  wQ: "/pieces/white_queen.svg",
  wR: "/pieces/white_rook.svg",
  wB: "/pieces/white_bishop.svg",
  wN: "/pieces/white_knight.svg",
  wP: "/pieces/white_pawn.svg",
  bK: "/pieces/black_king.svg",
  bQ: "/pieces/black_queen.svg",
  bR: "/pieces/black_rook.svg",
  bB: "/pieces/black_bishop.svg",
  bN: "/pieces/black_knight.svg",
  bP: "/pieces/black_pawn.svg",
};

const customPieces = Object.fromEntries(
  Object.entries(PIECE_IMAGES).map(([code, src]) => [
    code,
    () => <img src={src} style={{ width: "100%", height: "100%" }} />,
  ]),
);

const STOCKFISH_GREEN = "rgba(0, 150, 50, 0.85)";
const MAIA_ORANGE = "rgba(220, 120, 20, 0.9)";

export default function MiniBoard({ fen, width = 220, lastMove = null, arrows = null }) {
  const customSquareStyles = {};

  if (lastMove && lastMove.from && lastMove.to) {
    customSquareStyles[lastMove.from] = {
      backgroundColor: "rgba(201, 169, 110, 0.45)",
    };
    customSquareStyles[lastMove.to] = {
      backgroundColor: "rgba(201, 169, 110, 0.6)",
    };
  }

  const customArrows = (arrows || [])
    .filter((a) => a && a.from && a.to)
    .map((a) => [a.from, a.to, a.color || (a.role === "maia" ? MAIA_ORANGE : STOCKFISH_GREEN)]);

  return (
    <div className="chat-board">
      <ReactChessboard
        position={fen}
        boardWidth={width}
        arePiecesDraggable={false}
        animationDuration={0}
        boardOrientation="white"
        showBoardNotation={true}
        areArrowsAllowed={true}
        customArrows={customArrows}
        customArrowColor={STOCKFISH_GREEN}
        customPieces={customPieces}
        customSquareStyles={customSquareStyles}
        customBoardStyle={{
          borderRadius: "10px",
          backgroundImage:
            "linear-gradient(0deg, rgba(74, 178, 45, 0.43) 0%, rgba(74, 178, 45, 0.43) 100%), url(/textures/green-marble.png)",
          backgroundSize: "100% 100%",
          backgroundRepeat: "no-repeat",
          backgroundColor: "lightgray",
        }}
        customDarkSquareStyle={{
          boxShadow:
            "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
          backgroundColor: "rgba(223, 239, 252, 0.20)",
        }}
        customLightSquareStyle={{
          boxShadow:
            "inset 1px 0 0 rgba(226, 213, 124, 0.5), inset 0 1px 0 rgba(226, 213, 124, 0.5)",
          backgroundImage: "url(/textures/white-marble.png)",
          backgroundPosition: "50%",
          backgroundSize: "cover",
          backgroundRepeat: "no-repeat",
          backgroundColor: "rgba(255, 255, 255, 0.75)",
        }}
      />
    </div>
  );
}