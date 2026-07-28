import { useState, useRef, useCallback } from "react";
import TopBar from "./components/TopBar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
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
  });

  const handleAnalyze = useCallback(async () => {
    return await fetchGigaChatAnalysis(boardState.fen);
  }, [boardState.fen]);

  return (
    <>
      <TopBar />
      {/* Обертка для всех панелей и доски */}
      <div className="main-area">
        {/* 1. Левая панель */}
        <MoveHistory
          moveHistory={boardState.moveHistory}
          positionSnapshots={boardState.positionSnapshots}
          viewIndex={boardState.viewIndex}
          isViewMode={boardState.isViewMode}
          onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
        />

        {/* 2. Доска (по центру) */}
        <ChessboardComponent ref={boardRef} onStateChange={setBoardState} />

        {/* 3. Правая панель с чатом */}
        <ChatPanel onAnalyze={handleAnalyze} />
      </div>
    </>
  );
}
