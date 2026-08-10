import axios from "axios";

const api = axios.create({ baseURL: "" });

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

export async function fetchPlayerProfile(username, platform = "lichess") {
  const { data } = await api.get("/api/chess-profile", {
    params: { username, platform },
  });
  return data;
}
