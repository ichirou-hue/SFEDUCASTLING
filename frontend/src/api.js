import axios from "axios";

const api = axios.create({ baseURL: "" });

// === Auth: хранение токенов (задача 64) ===

const TOKEN_KEY = "gigachess_access";
const REFRESH_KEY = "gigachess_refresh";
const USER_KEY = "gigachess_user";
export const AUTH_EXPIRED_EVENT = "sfedu-auth-expired";

export function getAccessToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export async function fetchCurrentUser() {
  const { data } = await api.get("/api/auth/me");
  const user = data?.user ?? data;
  if (user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }
  return user;
}

function saveAuth(data) {
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(REFRESH_KEY, data.refresh_token);
  localStorage.setItem(USER_KEY, JSON.stringify(data.user));
  api.defaults.headers.common["Authorization"] = `Bearer ${data.access_token}`;
  return data.user;
}

export function clearAuth() {
  [TOKEN_KEY, REFRESH_KEY, USER_KEY].forEach((k) => localStorage.removeItem(k));
  delete api.defaults.headers.common["Authorization"];
}

function getRefreshToken() {
  return localStorage.getItem(REFRESH_KEY);
}

function refreshWasRejected(error) {
  const status = error?.response?.status;
  return status === 401 || status === 403;
}

function requireLogin(reason = "session_expired") {
  clearAuth();
  if (typeof window !== "undefined") {
    window.dispatchEvent(
      new CustomEvent(AUTH_EXPIRED_EVENT, { detail: { reason } }),
    );
  }
}

function authRequestMustNotAutoRefresh(url = "") {
  const value = String(url);
  return [
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/logout",
  ].some((path) => value.startsWith(path));
}

function accessTokenExpiresSoon(token, skewSeconds = 60) {
  if (!token) return true;
  try {
    const base64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
    const payload = JSON.parse(atob(padded));
    if (!payload.exp) return true;
    return payload.exp * 1000 <= Date.now() + skewSeconds * 1000;
  } catch {
    return true;
  }
}

// Одна общая refresh-операция на все параллельные запросы этой вкладки.
let refreshing = null;
async function refreshSession() {
  const refreshToken = getRefreshToken();
  if (!refreshToken) throw new Error("Refresh token is missing");

  if (!refreshing) {
    refreshing = axios
      .post("/api/auth/refresh", { refresh_token: refreshToken })
      .then(({ data }) => {
        saveAuth(data);
        return data;
      })
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

// Подставляем сохранённый access-token сразу после загрузки приложения.
if (getAccessToken()) {
  api.defaults.headers.common["Authorization"] = `Bearer ${getAccessToken()}`;
}

// Если access-token уже истёк или истечёт в ближайшую минуту, обновляем его
// ДО запроса. Так пользователь не замечает истечение короткого JWT.
api.interceptors.request.use(async (config) => {
  if (authRequestMustNotAutoRefresh(config.url)) return config;

  const accessToken = getAccessToken();
  const refreshToken = getRefreshToken();

  if (refreshToken && accessTokenExpiresSoon(accessToken)) {
    try {
      const data = await refreshSession();
      config.headers = config.headers || {};
      config.headers["Authorization"] = `Bearer ${data.access_token}`;
      return config;
    } catch (error) {
      // Только 401/403 от refresh означает, что сессия действительно закончилась.
      // Сетевой сбой/500 не должен разлогинивать пользователя.
      if (refreshWasRejected(error)) {
        requireLogin("refresh_rejected");
      }
      return Promise.reject(error);
    }
  }

  if (accessToken) {
    config.headers = config.headers || {};
    config.headers["Authorization"] = `Bearer ${accessToken}`;
  }
  return config;
});

// Запасной механизм: если сервер всё-таки вернул 401, один раз обновляем
// сессию и повторяем исходный запрос. Важно: /api/auth/me НЕ исключён,
// поэтому профиль тоже автоматически оживает после истечения access-token.
api.interceptors.response.use(
  (resp) => resp,
  async (error) => {
    const original = error.config || {};

    if (
      error.response?.status === 401 &&
      !original._retried &&
      !authRequestMustNotAutoRefresh(original.url)
    ) {
      const refreshToken = getRefreshToken();

      // Access недействителен, а refresh уже отсутствует: нужна повторная авторизация.
      if (!refreshToken) {
        requireLogin("refresh_missing");
        throw error;
      }

      original._retried = true;
      try {
        const data = await refreshSession();
        original.headers = original.headers || {};
        original.headers["Authorization"] = `Bearer ${data.access_token}`;
        return api(original);
      } catch (refreshError) {
        // Именно отказ refresh endpoint завершает локальную сессию.
        // Обычный 403 от бизнес-endpoint (например, нет роли admin) сюда не попадает.
        if (refreshWasRejected(refreshError)) {
          requireLogin("refresh_rejected");
        }
        throw refreshError;
      }
    }

    throw error;
  }
);

export async function register(login, password, email = null, elo = null) {
  const { data } = await api.post("/api/auth/register", {
    login,
    password,
    email,
    elo,
  });
  return saveAuth(data);
}

export async function login(loginOrEmail, password) {
  const { data } = await api.post("/api/auth/login", {
    login: loginOrEmail,
    password,
  });
  return saveAuth(data);
}

export async function logout() {
  const refresh = localStorage.getItem(REFRESH_KEY);
  try {
    if (refresh) {
      await api.post("/api/auth/logout", { refresh_token: refresh });
    }
  } finally {
    clearAuth();
  }
}

// === Онбординг-анкета Q1-Q8 ===

export async function saveOnboarding(answers) {
  const { data } = await api.post("/api/auth/onboarding", answers);
  const user = data?.user;
  if (user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }
  return user;
}

export async function fetchAdminUsers(limit = 100) {
  const { data } = await api.get("/api/auth/admin/users", { params: { limit } });
  return data;
}

export async function updateUserRole(userId, role) {
  const { data } = await api.patch(`/api/auth/admin/users/${userId}/role`, { role });
  return data;
}

export async function fetchMaiaMove(fen, elo, moves = []) {
  const { data } = await api.post("/api/maia-move", {
    fen,
    elo,
    moves,
    engine: "maia3",
  });
  return data;
}

export async function fetchStockfishAnalysis(fen) {
  const { data } = await api.post("/api/stockfish-analysis", {
    fen,
    elo: 1500,
    engine: "stockfish",
  });
  return data;
}

export async function fetchCompareMoves(fen, elo, moves = []) {
  const { data } = await api.post("/api/compare-moves", {
    fen,
    elo,
    moves,
  });
  return data;
}

export async function fetchEval(fen) {
  const { data } = await api.post("/api/eval", { fen });
  return data;
}

export async function fetchGigaChatAnalysis(fen) {
  const { data } = await api.post("/api/analyze", {
    fen,
    elo: 1500,
  });
  return data;
}

export async function saveMoveToDataset(fen, move, userId, gameId) {
  const { data } = await api.post("/api/save-move-to-dataset", {
    fen,
    move,
    user_id: userId,
    game_id: gameId,
  });
  return data;
}

export async function finishGame({ moves, userId, elo, engine, result, status }) {
  const { data } = await api.post("/api/game/finish", {
    moves,
    user_id: userId ?? null,
    elo: elo ?? null,
    engine: engine || "maia3",
    result: result || "*",
    status: status || "playing",
  });
  return data;
}

export async function fetchRandomOpening() {
  const { data } = await api.get("/api/knowledge/random-opening");
  return data;
}

export async function sendChatMessage(message, role = "user") {
  const { data } = await api.post("/api/chat/ingest", { message, role });
  return data;
}

export async function fetchChatMessages(after = 0) {
  const { data } = await api.get("/api/chat/messages", { params: { after } });
  return data;
}

export async function askLLM(message, fen, moves = [], isGreeting = false) {
  const { data } = await api.post("/api/chat/ask", {
    message,
    fen,
    moves,
    is_greeting: isGreeting,
  });
  return data;
}

export async function fetchPlayerProfile(username, platform = "lichess") {
  const { data } = await api.get("/api/chess-profile", {
    params: { username, platform },
  });
  return data;
}

export async function fetchExplainMove(
  fen,
  move,
  elo = 1500,
) {
  const { data } = await api.post("/api/explain-move", {
    fen,
    move,
    elo,
  });

  return data;
}

export async function fetchPuzzles(count = 20, topic = null) {
  const params = { count };
  if (topic) params.topic = topic;
  const { data } = await api.get("/api/learning/puzzles", { params });
  return data;
}

export async function checkPuzzle(puzzleId, move) {
  const { data } = await api.post("/api/learning/puzzle/check", {
    puzzle_id: puzzleId,
    move,
  });
  return data;
}

export async function recordPuzzleAttempt(puzzleId, correct) {
  const { data } = await api.post("/api/learning/puzzle/attempt", {
    puzzle_id: puzzleId,
    correct,
  });
  return data;
}

export async function fetchLearningProgress() {
  const { data } = await api.get("/api/learning/progress");
  return data;
}

export async function fetchWeaknessProfile() {
  const { data } = await api.get("/api/learning/weaknesses");
  return data;
}

export async function fetchAdaptivePuzzles(count = 20) {
  const { data } = await api.get("/api/learning/puzzles/adaptive", {
    params: { count },
  });
  return data;
}


// === Учебная подсистема ===

export async function fetchTrainingModules() {
  const { data } = await api.get("/api/training/modules");
  return data;
}

export async function fetchTrainingModule(slug) {
  const { data } = await api.get(`/api/training/modules/${slug}`);
  return data;
}

export async function fetchTrainingLesson(lessonId) {
  const { data } = await api.get(`/api/training/lessons/${lessonId}`);
  return data;
}

export async function checkTrainingTask(
  taskId,
  answer,
  hintsUsed = 0,
  responseTimeMs = null,
) {
  const { data } = await api.post(`/api/training/tasks/${taskId}/check`, {
    answer,
    hints_used: hintsUsed,
    response_time_ms: responseTimeMs,
  });
  return data;
}


export async function fetchTrainingProgress() {
  const { data } = await api.get("/api/training/progress");
  return data;
}

// === Стартовая оценка пользователя: персональный level-test ===

export async function startLevelTest() {
  const { data } = await api.post("/api/learning/level-test/start");
  return data;
}

export async function checkLevelTestAnswer(testId, puzzleId, move, responseTimeMs = null) {
  const { data } = await api.post("/api/learning/level-test/check", {
    test_id: testId,
    puzzle_id: puzzleId,
    move,
    response_time_ms: responseTimeMs,
  });
  return data;
}

export async function skipLevelTestQuestion(testId, puzzleId, responseTimeMs = null) {
  const { data } = await api.post("/api/learning/level-test/skip", {
    test_id: testId,
    puzzle_id: puzzleId,
    response_time_ms: responseTimeMs,
  });
  return data;
}

export async function finishLevelTest(testId, answers = []) {
  const { data } = await api.post("/api/learning/level-test/submit", {
    test_id: testId,
    answers,
  });
  return data;
}

// === Связанные шахматные аккаунты (soft link, без OAuth) ===

export async function fetchLinkedChessAccounts() {
  const { data } = await api.get("/api/chess-profile/linked");
  return data;
}

export async function linkChessAccount(username, platform = "lichess") {
  const { data } = await api.post("/api/chess-profile/link", {
    username,
    platform,
  });
  return data;
}

export async function unlinkChessAccount(platform = "lichess") {
  const { data } = await api.delete(`/api/chess-profile/link/${platform}`);
  return data;
}
