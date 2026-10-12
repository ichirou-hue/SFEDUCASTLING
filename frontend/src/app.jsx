import { useState, useRef, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import RegisterModal from "./components/RegisterModal.jsx";
import PuzzlesPage from "./components/PuzzlesPage.jsx";
import TrainingPage from "./components/TrainingPage.jsx";
import UserProfilePage from "./components/UserProfilePage.jsx";
import AdminPage from "./components/AdminPage.jsx";
import LevelTestPage from "./components/LevelTestPage.jsx";
import {
  AUTH_EXPIRED_EVENT,
  fetchCurrentUser,
  getStoredUser,
} from "./api.js";

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
  const [authMode, setAuthMode] = useState("register");
  const [authNotice, setAuthNotice] = useState("");
  const [user, setUser] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Пока не прочитали localStorage, роль неизвестна — нельзя редиректить с /admin.
  const [authChecked, setAuthChecked] = useState(false);

  const isAdmin = user?.is_admin === true || user?.role === "admin";

  useEffect(() => {
    const stored = getStoredUser();
    if (stored) {
      setUser(stored);
      // Обновляем профиль с сервера: роль могла измениться с прошлого визита.
      // Ошибка сети не должна разлогинивать — молча оставляем копию из localStorage.
      fetchCurrentUser()
        .then((fresh) => {
          if (fresh) setUser(fresh);
        })
        .catch(() => {});
    }
    setAuthChecked(true);

    const handleAuthExpired = () => {
      setUser(null);
      setSidebarOpen(false);
      setAuthMode("login");
      setAuthNotice("Сессия истекла. Войдите в аккаунт снова.");
      setShowRegister(true);
    };

    window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
  }, []);

  return (
    <>
      <TopBar
        user={user}
        onUserChange={setUser}
        onRegister={() => {
          setAuthMode("register");
          setAuthNotice("");
          setShowRegister(true);
        }}
        onMenuClick={() => setSidebarOpen(true)}
      />
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        user={user}
      />
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route
          path="/puzzles"
          element={
            <PuzzlesPage
              user={user}
              onRegister={() => {
                setAuthMode("register");
                setAuthNotice("");
                setShowRegister(true);
              }}
            />
          }
        />
        <Route
          path="/training"
          element={
            <TrainingPage
              user={user}
              onRegister={() => {
                setAuthMode("register");
                setAuthNotice("");
                setShowRegister(true);
              }}
            />
          }
        />
        <Route
          path="/profile"
          element={<UserProfilePage user={user} onUserChange={setUser} />}
        />
        <Route
          path="/level-test"
          element={
            !authChecked ? null : user ? (
              <LevelTestPage />
            ) : (
              // Тест уровня требует аккаунт: сервер вернёт 401 без токена.
              <div className="training-page training-page--centered">
                <div className="training-register-wall">
                  <div className="training-kicker">Проверка уровня</div>
                  <h1>Тест доступен после регистрации</h1>
                  <p>
                    Персональный тест из 20 задач помогает определить ваш уровень
                    и подобрать подходящее обучение.
                  </p>
                  <button
                    type="button"
                    className="training-primary-btn"
                    onClick={() => {
                      setAuthMode("register");
                      setAuthNotice("");
                      setShowRegister(true);
                    }}
                  >
                    Зарегистрироваться
                  </button>
                </div>
              </div>
            )
          }
        />
        <Route
          path="/admin"
          element={
            !authChecked ? null : isAdmin ? (
              <AdminPage user={user} onUserChange={setUser} />
            ) : (
              // Гость/ученик не должны видеть админку — уводим в профиль.
              <Navigate to="/profile" replace />
            )
          }
        />
      </Routes>
      <RegisterModal
        isOpen={showRegister}
        initialMode={authMode}
        notice={authNotice}
        onClose={() => {
          setShowRegister(false);
          setAuthNotice("");
        }}
        onSuccess={(u) => {
          setUser(u);
          setAuthNotice("");
        }}
      />
    </>
  );
}
