import axios from "axios";

const api = axios.create({ baseURL: "" });

// === Auth: хранение токенов (задача 64) ===

const TOKEN_KEY = "gigachess_access";
const REFRESH_KEY = "gigachess_refresh";
const USER_KEY = "gigachess_user";

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

// Подставляем токен во все запросы автоматически
if (getAccessToken()) {
  api.defaults.headers.common["Authorization"] = `Bearer ${getAccessToken()}`;
}

// При 401 пробуем обновить токен один раз и повторить запрос
let refreshing = null;
api.interceptors.response.use(
  (resp) => resp,
  async (error) => {
    const original = error.config;
    if (
      error.response?.status === 401 &&
      getAccessToken() &&
      !original._retried &&
      !String(original.url).startsWith("/api/auth/")
    ) {
      original._retried = true;
      try {
        refreshing =
          refreshing ||
          axios.post("/api/auth/refresh", {
            refresh_token: localStorage.getItem(REFRESH_KEY),
          });
        const { data } = await refreshing;
        saveAuth(data);
        original.headers["Authorization"] = `Bearer ${data.access_token}`;
        return api(original);
      } catch {
        clearAuth();
      } finally {
        refreshing = null;
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
