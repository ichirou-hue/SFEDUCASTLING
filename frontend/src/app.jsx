import { useState, useRef, useEffect } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import SiteFooter from "./components/SiteFooter.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import RegisterModal from "./components/RegisterModal.jsx";
import OnboardingModal from "./components/OnboardingModal.jsx";
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

function MainPage({ user }) {
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
    isComparing: false,
  });

  /*
   * Чат можно свернуть в узкую полосу - тогда освободившаяся ширина
   * уходит доске (см. updateBoardSize в Chessboard.jsx).
   */
  const [chatCollapsed, setChatCollapsed] = useState(false);

  return (
    // Своя скролл-колонка: футер в потоке контента (внизу при проскролле),
    // а не зафиксированной полосой поверх страницы.
    <div className="main-page-scroll">
      <div className="main-area">
        <MoveHistory
          moveHistory={boardState.moveHistory}
          positionSnapshots={boardState.positionSnapshots}
          viewIndex={boardState.viewIndex}
          isViewMode={boardState.isViewMode}
          maiaRating={boardState.maiaRating}
          userRating={user?.elo || undefined}
          onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
        />
        <EvalBar
          diff={boardState.materialDiff}
          flipped={boardState.boardFlipped}
          height={boardState.boardHeight}
        />
        <ChessboardComponent
          ref={boardRef}
          onStateChange={setBoardState}
          chatCollapsed={chatCollapsed}
        />
        <ChatPanel
          hintMessage={boardState.hintMessage}
          currentFen={boardState.fen}
          currentMoves={boardState.moveHistory}
          isComparing={boardState.isComparing}
          onAnalyze={() => boardRef.current?.requestBestMove?.()}
          collapsed={chatCollapsed}
          onToggleCollapsed={() => setChatCollapsed((v) => !v)}
        />
      </div>
      <SiteFooter />
    </div>
  );
}

function AccessWall({ kicker, title, text, onRegister }) {
  return (
    <div className="training-page training-page--centered">
      <div className="training-center-wrap">
        <div className="training-register-wall">
          <div className="training-kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{text}</p>
          <button
            type="button"
            className="training-primary-btn"
            onClick={onRegister}
          >
            Зарегистрироваться
          </button>
        </div>
      </div>
      <SiteFooter />
    </div>
  );
}

export default function App() {
  const navigate = useNavigate();
  const [showRegister, setShowRegister] = useState(false);
  const [authMode, setAuthMode] = useState("register");
  const [authNotice, setAuthNotice] = useState("");
  const [user, setUser] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Пока не прочитали localStorage, роль неизвестна — нельзя редиректить с /admin.
  const [authChecked, setAuthChecked] = useState(false);
  // Анкета открывается автоматически после регистрации/входа, пока не заполнена.
  const [showOnboarding, setShowOnboarding] = useState(false);
  // После анкеты сразу запускаем входной тест на /level-test.
  const [autoStartTest, setAutoStartTest] = useState(false);

  const isAdmin = user?.is_admin === true || user?.role === "admin";

  const openRegister = () => {
    setAuthMode("register");
    setAuthNotice("");
    setShowRegister(true);
  };

  // Пользователь без заполненной анкеты (и не админ) должен увидеть её сразу.
  const needsOnboarding = (u) =>
    Boolean(u) && !u.onboarding_completed && !(u.is_admin === true || u.role === "admin");

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
    const refreshAfterAssessment = () => {
      fetchCurrentUser()
        .then((fresh) => {
          if (fresh) setUser(fresh);
        })
        .catch(() => {});
    };

    window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
    window.addEventListener("sfedu-assessment-updated", refreshAfterAssessment);
    return () => {
      window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
      window.removeEventListener("sfedu-assessment-updated", refreshAfterAssessment);
    };
  }, []);

  const handleAuthSuccess = (u) => {
    setUser(u);
    setAuthNotice("");
    // Автоматический онбординг: анкета, затем входной тест.
    if (needsOnboarding(u)) {
      setAutoStartTest(false);
      setShowOnboarding(true);
    }
  };

  return (
    <>
      <TopBar
        user={user}
        onUserChange={setUser}
        onOpenOnboarding={() => setShowOnboarding(true)}
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
        <Route path="/" element={<MainPage user={user} />} />
        <Route
          path="/puzzles"
          element={
            !authChecked ? null : user ? (
              <PuzzlesPage
                user={user}
                onOpenOnboarding={() => setShowOnboarding(true)}
                onRegister={openRegister}
              />
            ) : (
              <AccessWall
                kicker="Шахматные задачи"
                title="Задачи доступны после регистрации"
                text="Создайте аккаунт, чтобы решать задачи, сохранять прогресс и получать персональные тренировки. Это займёт минуту."
                onRegister={openRegister}
              />
            )
          }
        />
        <Route
          path="/training"
          element={
            !authChecked ? null : user ? (
              <TrainingPage user={user} onRegister={openRegister} />
            ) : (
              <AccessWall
                kicker="Обучение"
                title="Обучение доступно после регистрации"
                text="Создайте аккаунт, чтобы проходить уроки, выполнять задания и сохранять прогресс обучения. Это займёт минуту."
                onRegister={openRegister}
              />
            )
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
              <LevelTestPage
                autoStart={autoStartTest}
                onAutoStartHandled={() => setAutoStartTest(false)}
              />
            ) : (
              // Тест уровня требует аккаунт: сервер вернёт 401 без токена.
              <div className="training-page training-page--centered">
                <div className="training-center-wrap">
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
                <SiteFooter />
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
        onSuccess={handleAuthSuccess}
      />
      <OnboardingModal
        isOpen={showOnboarding}
        onClose={() => setShowOnboarding(false)}
        onDone={(freshUser) => {
          if (freshUser) setUser(freshUser);
          setShowOnboarding(false);
          // Сразу после анкеты — входной тест (страница сама его запускает).
          setAutoStartTest(true);
          navigate("/level-test");
        }}
      />
    </>
  );
}
