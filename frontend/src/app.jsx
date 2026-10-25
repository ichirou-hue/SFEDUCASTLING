import { useState, useRef, useEffect } from "react";
import { Routes, Route, Navigate, Link, useNavigate } from "react-router-dom";
import TopBar from "./components/TopBar.jsx";
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
import "./home.css";
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

function HomePage({ user, onRegister, onOpenOnboarding }) {
  return (
    <main className="home-page">
      <section className="home-hero">
        <div className="home-hero-copy">
          <h1>
            <span className="home-title-line">Думай на ход</span>
            <br />
            <em>вперёд.</em>
          </h1>
          <p className="home-lead">
            Играй, решай задачи и учись вместе с ИИ-тренером. Персональный путь
            к более сильной игре начинается здесь.
          </p>
          <div className="home-actions">
            <Link className="home-primary" to="/play">
              Начать играть <span aria-hidden="true">↗</span>
            </Link>
            {!user ? (
              <button type="button" className="home-secondary" onClick={onRegister}>
                Создать аккаунт
              </button>
            ) : !user.onboarding_completed ? (
              <button
                type="button"
                className="home-secondary"
                onClick={onOpenOnboarding}
              >
                Подобрать обучение
              </button>
            ) : null}
          </div>
          <div className="home-proof">
            <span className="home-proof-mark">♞</span>
            <span>Твой темп. Твоя стратегия. Твой прогресс.</span>
          </div>
        </div>
        <div className="home-hero-aside" aria-label="Возможности GigaChess">
          <div className="home-aside-heading">
            <span>Твой прогресс</span>
            <span className="home-online-dot" />
          </div>
          <div className="home-progress-chart" aria-hidden="true">
            <div className="home-chart-label">
              <span>Точность ходов</span>
              <span>+12%</span>
            </div>
            <svg viewBox="0 0 320 130" preserveAspectRatio="none">
              <defs>
                <linearGradient id="home-chart-fill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#74d8bf" stopOpacity=".32" />
                  <stop offset="100%" stopColor="#74d8bf" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path
                d="M0 112 C24 99 31 78 54 84 S83 100 106 67 S145 81 165 56 S199 61 220 40 S247 67 269 48 S294 28 320 12 V130 H0Z"
                fill="url(#home-chart-fill)"
              />
              <path
                d="M0 112 C24 99 31 78 54 84 S83 100 106 67 S145 81 165 56 S199 61 220 40 S247 67 269 48 S294 28 320 12"
                fill="none"
                stroke="#74d8bf"
                strokeWidth="3"
                strokeLinecap="round"
              />
            </svg>
            <div className="home-chart-days">
              <span>ПН</span><span>ВТ</span><span>СР</span><span>ЧТ</span>
              <span>ПТ</span><span>СБ</span><span>ВС</span>
            </div>
          </div>
          <div className="home-aside-caption">
            Каждая партия — возможность стать сильнее.
          </div>
          <div className="home-stats">
            <div>
              <span>Играй</span>
              <strong>Свои партии</strong>
              <small>Игра с Maia и разбор</small>
            </div>
            <div>
              <span>Тренируйся</span>
              <strong>Тактический расчёт</strong>
              <small>Задачи под твой уровень</small>
            </div>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}


export default function App() {
  const navigate = useNavigate();
  const [showRegister, setShowRegister] = useState(false);
  const [authMode, setAuthMode] = useState("register");
  const [authNotice, setAuthNotice] = useState("");
  const [user, setUser] = useState(null);
  // Пока не прочитали localStorage, роль неизвестна — нельзя редиректить с /admin.
  const [authChecked, setAuthChecked] = useState(false);
  // Анкета открывается автоматически после регистрации/входа, пока не заполнена.
  const [showOnboarding, setShowOnboarding] = useState(false);
  // После анкеты сразу запускаем входной тест на /level-test.
  const [autoStartTest, setAutoStartTest] = useState(false);

  const isAdmin = user?.is_admin === true || user?.role === "admin";

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
      />
      <Routes>
        <Route
          path="/"
          element={
            <HomePage
              user={user}
              onRegister={() => {
                setAuthMode("register");
                setAuthNotice("");
                setShowRegister(true);
              }}
              onOpenOnboarding={() => setShowOnboarding(true)}
            />
          }
        />
        <Route path="/play" element={<MainPage user={user} />} />
        <Route
          path="/puzzles"
          element={
            <PuzzlesPage
              user={user}
              onOpenOnboarding={() => setShowOnboarding(true)}
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
