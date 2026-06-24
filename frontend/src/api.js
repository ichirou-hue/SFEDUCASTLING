import axios from 'axios'

const api = axios.create({ baseURL: '' })

export async function fetchMaiaMove(fen, elo) {
  const { data } = await api.post('/api/maia-move', { fen, elo })
  return data
}

export async function fetchStockfishAnalysis(fen) {
  const { data } = await api.post('/api/stockfish-analyze', { fen, elo: 1500 })
  return data
}

export async function fetchGigaChatAnalysis(fen) {
  const { data } = await api.post('/api/analyze', { fen, elo: 1500 })
  return data
}

export async function saveMoveToDataset(fen, move, userId, gameId) {
  const { data } = await api.post('/api/save-move-to-dataset', {
    fen, move, user_id: userId, game_id: gameId,
  })
  return data
}

export async function fetchRandomOpening() {
  const { data } = await api.get('/api/knowledge/random-opening')
  return data
}
