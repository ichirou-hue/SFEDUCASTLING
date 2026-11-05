const STORAGE_KEY = "gigachess_local_bot_games_v1";
const MAX_GAMES = 30;

export function getLocalBotGames(userId) {
  try {
    const games = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(games) ? games.filter((game) => game.userId === userId) : [];
  } catch {
    return [];
  }
}

export function saveLocalBotGame(game) {
  try {
    const all = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    const games = Array.isArray(all) ? all : [];
    localStorage.setItem(STORAGE_KEY, JSON.stringify([game, ...games].slice(0, MAX_GAMES)));
  } catch {
    // Private browsing and storage limits must not interrupt the game.
  }
}
