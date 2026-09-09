import { useState, useRef, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import RegisterModal from "./components/RegisterModal.jsx";
import PuzzlesPage from "./components/PuzzlesPage.jsx";
import TrainingPage from "./components/TrainingPage.jsx";
import { getStoredUser } from "./api.js";

function MainPage() {
  const boardRef = useRef(null);
  const [boardState, setBoardState] = useState({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    moveHistory: [],
    turn: "w",
    positionSnapshots: [],
    viewIndex: -1,
    isViewMode: false,
    evalScore: null,
    maiaRating: 1500,
    hintMessage: null,
  });

  return (
    <div className="main-area">
      <MoveHistory
        moveHistory={boardState.moveHistory}
        positionSnapshots={boardState.positionSnapshots}
        viewIndex={boardState.viewIndex}
        isViewMode={boardState.isViewMode}
        maiaRating={boardState.maiaRating}
        onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
      />
      <EvalBar
        diff={boardState.materialDiff}
        flipped={boardState.boardFlipped}
        height={boardState.boardHeight}
      />
      <ChessboardComponent ref={boardRef} onStateChange={setBoardState} />
      <ChatPanel
        hintMessage={boardState.hintMessage}
        currentFen={boardState.fen}
        currentMoves={boardState.moveHistory}
      />
    </div>
  );
}

export default function App() {
  const [showRegister, setShowRegister] = useState(false);
  const [user, setUser] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const stored = getStoredUser();
    if (stored) setUser(stored);
  }, []);

  return (
    <>
      <TopBar
        user={user}
        onRegister={() => setShowRegister(true)}
        onMenuClick={() => setSidebarOpen(true)}
      />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/puzzles" element={<PuzzlesPage />} />
        <Route path="/training" element={<TrainingPage />} />
      </Routes>
      <RegisterModal
        isOpen={showRegister}
        onClose={() => setShowRegister(false)}
        onSuccess={(u) => setUser(u)}
      />
    </>
  );
}
