import { useState, useRef, useEffect } from "react";
import { Routes, Route, Navigate, Link, useNavigate } from "react-router-dom";
import TopBar from "./components/TopBar.jsx";
import SiteFooter from "./components/SiteFooter.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import MaiaLevelProgress from "./components/MaiaLevelProgress.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import EvalBar from "./components/EvalBar.jsx";
import RegisterModal from "./components/RegisterModal.jsx";
import OnboardingModal from "./components/OnboardingModal.jsx";
import PuzzlesPage from "./components/PuzzlesPage.jsx";
import TrainingPage from "./components/TrainingPage.jsx";
import UserProfilePage from "./components/UserProfilePage.jsx";
import AdminPage from "./components/AdminPage.jsx";
import LevelTestPage from "./components/LevelTestPage.jsx";
import UserRecordsWidget from "./components/UserRecordsWidget.jsx";
import "./home.css";
import "./mobile.css";
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
    maiaVictory: null,
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
        <div className="game-history-column">
          <MoveHistory
            moveHistory={boardState.moveHistory}
            positionSnapshots={boardState.positionSnapshots}
            viewIndex={boardState.viewIndex}
            isViewMode={boardState.isViewMode}
            maiaRating={boardState.maiaRating}
            userRating={user?.elo || undefined}
            onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
          />
          <MaiaLevelProgress
            userId={user?.id}
            currentLevel={boardState.maiaRating}
            lastWin={boardState.maiaVictory}
            onSelectLevel={(level) => boardRef.current?.setMaiaLevel?.(level)}
          />
        </div>
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
        <div className="home-hero-aside" aria-label="Статистика пользователя">
          {user ? (
            <UserRecordsWidget variant="home" />
          ) : (
            <section className="home-stats-guest">
              <span className="home-stats-guest__mark" aria-hidden="true">♞</span>
              <div>
                <div className="user-profile-kicker">Твой прогресс</div>
                <h2>Твои партии и рекорды — в одном месте</h2>
                <p>Создай аккаунт, чтобы сохранять результаты и видеть настоящую статистику.</p>
                <button type="button" className="home-stats-guest__button" onClick={onRegister}>
                  Создать аккаунт <span aria-hidden="true">↗</span>
                </button>
              </div>
            </section>
          )}
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
  const [gigacoinToast, setGigacoinToast] = useState(null);

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
    const updatePuzzleStreak = (event) => {
      const streak = Number(event?.detail?.streak);
      if (Number.isFinite(streak)) {
        setUser((current) => current ? { ...current, puzzle_streak: streak } : current);
      }
    };
    let gigacoinToastTimer;
    const showGigacoinReward = (event) => {
      const amount = Number(event?.detail?.amount || 0);
      if (amount <= 0) return;
      const balance = Number(event?.detail?.balance);
      if (Number.isFinite(balance)) {
        setUser((current) => current ? { ...current, gigacoins: balance } : current);
      }
      setGigacoinToast({
        amount,
        balance: Number.isFinite(balance) ? balance : null,
      });
      window.clearTimeout(gigacoinToastTimer);
      gigacoinToastTimer = window.setTimeout(() => setGigacoinToast(null), 4000);
    };

    window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
    window.addEventListener("sfedu-assessment-updated", refreshAfterAssessment);
    window.addEventListener("sfedu-gigacoins-earned", showGigacoinReward);
    window.addEventListener("sfedu-puzzle-streak-updated", updatePuzzleStreak);
    return () => {
      window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired);
      window.removeEventListener("sfedu-assessment-updated", refreshAfterAssessment);
      window.removeEventListener("sfedu-gigacoins-earned", showGigacoinReward);
      window.removeEventListener("sfedu-puzzle-streak-updated", updatePuzzleStreak);
      window.clearTimeout(gigacoinToastTimer);
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
      {gigacoinToast?.amount > 0 && (
        <div className="gigacoin-reward-toast" role="status" aria-live="polite">
          <span className="gigacoin-reward-toast__icon" aria-hidden="true">✦</span>
          <span className="gigacoin-reward-toast__copy">
            <strong>Получено +{gigacoinToast.amount} GigaCoin</strong>
            {gigacoinToast.balance != null && (
              <small>Баланс: {gigacoinToast.balance.toLocaleString("ru-RU")}</small>
            )}
          </span>
        </div>
      )}
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
