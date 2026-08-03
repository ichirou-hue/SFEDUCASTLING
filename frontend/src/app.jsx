import { useState, useRef, useCallback } from "react";
import TopBar from "./components/TopBar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import { fetchGigaChatAnalysis } from "./api.js";

export default function App() {
  const boardRef = useRef(null);
  const [boardState, setBoardState] = useState({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    moveHistory: [],
    turn: "w",
    positionSnapshots: [],
    viewIndex: -1,
    isViewMode: false,
    evalScore: null,
  });

  const handleAnalyze = useCallback(async () => {
    return await fetchGigaChatAnalysis(boardState.fen);
  }, [boardState.fen]);

  return (
    <>
      <TopBar />
      <div className="main-area">
        {/* 1. Левая панель */}
        <MoveHistory
          moveHistory={boardState.moveHistory}
          positionSnapshots={boardState.positionSnapshots}
          viewIndex={boardState.viewIndex}
          isViewMode={boardState.isViewMode}
          onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
        />

        {/* 1.5 Eval Bar */}
        <EvalBar evaluation={boardState.evalScore} />

        {/* 2. Центральная панель */}
        <ChessboardComponent ref={boardRef} onStateChange={setBoardState} />

        {/* 3. Правая панель (ЧАТ) */}
        <ChatPanel onAnalyze={handleAnalyze} />
      </div>
    </>
  );
}
