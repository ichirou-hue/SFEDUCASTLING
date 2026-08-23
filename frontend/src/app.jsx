import { useState, useRef, useCallback, useEffect } from "react";
import TopBar from "./components/TopBar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import RegisterModal from "./components/RegisterModal.jsx";
import { fetchGigaChatAnalysis, getStoredUser } from "./api.js";

export default function App() {
  const boardRef = useRef(null);
  const [showRegister, setShowRegister] = useState(false);
  const [user, setUser] = useState(null);

  // Восстанавливаем сессию из localStorage при загрузке
  useEffect(() => {
    const stored = getStoredUser();
    if (stored) setUser(stored);
  }, []);
  const [boardState, setBoardState] = useState({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    moveHistory: [],
    turn: "w",
    positionSnapshots: [],
    viewIndex: -1,
    isViewMode: false,
    evalScore: null,
    maiaRating: 1500,
  });

  const handleAnalyze = useCallback(async () => {
    return await fetchGigaChatAnalysis(boardState.fen);
  }, [boardState.fen]);

  return (
    <>
      <TopBar
        user={user}
        onRegister={() => setShowRegister(true)}
      />
      <div className="main-area">
        {/* 1. Левая панель */}
        <MoveHistory
          moveHistory={boardState.moveHistory}
          positionSnapshots={boardState.positionSnapshots}
          viewIndex={boardState.viewIndex}
          isViewMode={boardState.isViewMode}
          maiaRating={boardState.maiaRating}
          onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
        />

        {/* 1.5 Eval Bar */}
        <EvalBar
          diff={boardState.materialDiff}
          flipped={boardState.boardFlipped}
          height={boardState.boardHeight}
        />

        {/* 2. Центральная панель */}
        <ChessboardComponent ref={boardRef} onStateChange={setBoardState} />

        {/* 3. Правая панель (ЧАТ) */}
        <ChatPanel onAnalyze={handleAnalyze} />
      </div>
      <RegisterModal
        isOpen={showRegister}
        onClose={() => setShowRegister(false)}
        onSuccess={(u) => setUser(u)}
      />
    </>
  );
}
