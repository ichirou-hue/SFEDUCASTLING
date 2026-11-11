import { useEffect, useMemo, useState } from "react";
import { Chess } from "chess.js";
import { Link, useParams } from "react-router-dom";
import { fetchLocalGameHistory, fetchStockfishAnalysis, getAccessToken, getStoredUser, reviewGame } from "../api.js";
import { getLocalBotGames } from "../localBotGames.js";
import { playMoveSound } from "../moveSound.js";
import MiniBoard from "./MiniBoard.jsx";
import "./GameReviewPage.css";

const GRADES = {
  best: ["★", "Лучший"], excellent: ["!", "Отличный"],
  good: ["✓", "Хороший"], inaccuracy: ["?!", "Неточность"],
  mistake: ["?", "Ошибка"], blunder: ["??", "Зевок"],
};

function parsePgn(source) {
  const game = new Chess();
  game.loadPgn(source.trim());
  const history = game.history({ verbose: true });
  if (history.length < 1) throw new Error("В партии нет ходов.");
  const replay = new Chess();
  const positions = [replay.fen()];
  const moves = history.map((move) => {
    replay.move(move);
    positions.push(replay.fen());
    return `${move.from}${move.to}${move.promotion || ""}`;
  });
  return { moves, positions, sanMoves: history.map((move) => move.san), headers: game.header() };
}

function fromServerGame(saved) {
  const chess = new Chess();
  try { saved.moves.forEach((move) => chess.move(move)); } catch { return null; }
  if (chess.history().length < 2) return null;
  return {
    id: `server-${saved.id}`, pgn: chess.pgn(), playedAt: saved.played_at,
    elo: saved.elo, result: saved.result, playerColor: saved.player_color,
  };
}

function moveSignature(game) {
  try { return parsePgn(game.pgn).moves.join(" "); } catch { return null; }
}

function finalPosition(game) {
  try { return parsePgn(game.pgn).positions.at(-1); } catch { return new Chess().fen(); }
}

function resultLabel(game) {
  if (game.result === "1/2-1/2") return "Ничья";
  if (game.result === (game.playerColor === "w" ? "1-0" : "0-1")) return "Победа";
  return "Поражение";
}

function gameTimestamp(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatGameDate(value) {
  const date = gameTimestamp(value);
  if (!date) return "—";
  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function EvalChart({ plies, selected, onSelect }) {
  const points = plies.filter((ply) => ply.eval_cp != null).map((ply) => ({
    index: ply.ply - 1,
    x: 18 + ((ply.ply - 1) / Math.max(1, plies.length - 1)) * 764,
    y: 90 - Math.tanh(ply.eval_cp / 450) * 74,
    evalCp: ply.eval_cp,
  }));
  return <svg className="review-chart" viewBox="0 0 800 180" role="img" aria-label="График оценки партии">
    <line x1="18" x2="782" y1="90" y2="90" className="review-chart-zero" />
    {points.length > 1 && <polyline points={points.map((p) => `${p.x},${p.y}`).join(" ")} className="review-chart-line" />}
    {points.map((point) => <circle key={point.index} cx={point.x} cy={point.y} r={point.index === selected ? 7 : 4}
      className="review-chart-point" onClick={() => onSelect(point.index)} tabIndex={0}
      onKeyDown={(event) => { if (event.key === "Enter") onSelect(point.index); }}>
      <title>Ход {point.index + 1}: {point.evalCp / 100}</title>
    </circle>)}
  </svg>;
}

export default function GameReviewPage() {
  const { gameId } = useParams();
  const detail = Boolean(gameId);
  const storedUserId = getStoredUser()?.id;
  const userId = storedUserId == null ? null : Number(storedUserId);
  const [games, setGames] = useState(() => getLocalBotGames(userId));
  const [selectedGame, setSelectedGame] = useState(() =>
    getLocalBotGames(userId).find((game) => game.id === gameId) || null);
  const [parsed, setParsed] = useState(null);
  const [report, setReport] = useState(null);
  const [selected, setSelected] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [side, setSide] = useState("w");
  const [practice, setPractice] = useState(false);
  const [bestMove, setBestMove] = useState(null);
  const [practiceMessage, setPracticeMessage] = useState("");
  const [practiceLoading, setPracticeLoading] = useState(false);

  useEffect(() => {
    if (!getAccessToken()) return;
    let active = true;
    fetchLocalGameHistory().then((saved) => {
      if (!active) return;
      const serverGames = saved.map(fromServerGame).filter(Boolean);
      const localGames = getLocalBotGames(userId);
      const merged = [...localGames];
      const signatures = new Set(localGames.map(moveSignature).filter(Boolean));
      for (const game of serverGames) {
        const signature = moveSignature(game);
        if (!signatures.has(signature)) { merged.push(game); signatures.add(signature); }
      }
      merged.sort((a, b) => new Date(b.playedAt || 0) - new Date(a.playedAt || 0));
      setGames(merged);
      if (gameId) setSelectedGame((current) => current || serverGames.find((game) => game.id === gameId) || null);
    }).catch(() => { if (active && gameId && !selectedGame) setError("Не удалось загрузить партию из архива."); });
    return () => { active = false; };
  }, [gameId, userId]);

  useEffect(() => {
    if (!detail || !selectedGame) return;
    let active = true;
    let game;
    try { game = parsePgn(selectedGame.pgn); }
    catch (cause) { setError(cause.message || "Не удалось прочитать партию."); return; }
    setParsed(game);
    setSelected(-1);
    setPlaying(true);
    setSide(selectedGame.playerColor || "w");
    setBusy(true);
    setReport(null);
    setError("");
    Promise.all([reviewGame(game.moves, "w"), reviewGame(game.moves, "b")])
      .then(([white, black]) => { if (active) setReport({ white, black }); })
      .catch((cause) => { if (active) setError(cause?.response?.data?.detail || "Не удалось оценить ходы Stockfish. Партию можно просмотреть без оценок."); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [detail, selectedGame?.id]);

  useEffect(() => {
    if (!playing || !parsed) return;
    if (selected >= parsed.moves.length - 1) { setPlaying(false); return; }
    const timer = setTimeout(() => setSelected((current) => current + 1), 420);
    return () => clearTimeout(timer);
  }, [playing, parsed, selected]);

  useEffect(() => { setPractice(false); setBestMove(null); setPracticeMessage(""); }, [selected]);

  const plies = useMemo(() => parsed?.moves.map((move, index) => ({
    ply: index + 1,
    move_no: Math.floor(index / 2) + 1,
    san: parsed.sanMoves[index],
    fen: parsed.positions[index],
    analysis: (index % 2 === 0 ? report?.white : report?.black)?.plies[index],
    classification: (index % 2 === 0 ? report?.white : report?.black)?.plies[index]?.classification,
  })) || [], [parsed, report]);
  const current = parsed?.positions[selected + 1] || new Chess().fen();
  const analysisPlies = report?.white?.plies || [];
  const stage = busy
    ? selected < plies.length - 1 ? "Воспроизводим партию и оцениваем ходы…" : "Готовим оценки и классификацию ходов…"
    : report ? "Разбор готов. Выберите ход или запустите повтор." : "Просмотр партии";

  function navigateTo(index) { setPlaying(false); setSelected(Math.max(-1, Math.min(plies.length - 1, index))); }

  async function startPractice() {
    const ply = plies[selected];
    if (!ply) return;
    setPracticeLoading(true);
    setPracticeMessage("");
    try {
      const result = await fetchStockfishAnalysis(ply.fen);
      if (!result.best_move) throw new Error(result.error || "Движок не нашёл ход.");
      setBestMove(result);
      setPractice(true);
    } catch (cause) { setPracticeMessage(cause.message || "Не удалось подготовить позицию."); }
    finally { setPracticeLoading(false); }
  }

  function tryPracticeMove(from, to) {
    if (!practice || !bestMove) return false;
    try {
      const game = new Chess(plies[selected].fen);
      if (!game.move({ from, to, promotion: "q" })) return false;
    } catch {
      return false;
    }
    playMoveSound();
    setPracticeMessage(`${from}${to}` === bestMove.best_move.slice(0, 4)
      ? `Верно! Лучший ход: ${bestMove.san}` : "Попробуйте другой ход.");
    return false;
  }

  if (!detail) return <main className="review-page review-page--archive">
    <header className="review-heading"><span className="review-kicker">МОИ ПАРТИИ</span>
      <h1>Партии против бота</h1><p>Откройте партию, чтобы просмотреть ходы и оценки.</p></header>
    <section className="review-archive">
      {games.length ? games.map((game) => <Link key={game.id} className="review-game-row"
        to={`/review/game/${encodeURIComponent(game.id)}`} target="_blank" rel="noopener noreferrer">
        <div className="review-game-info"><strong>{resultLabel(game)}</strong>
          <span>Maia {game.elo ?? "—"} · {game.playerColor === "b" ? "игра чёрными" : "игра белыми"}</span>
          <span>{formatGameDate(game.playedAt)}</span>
          <span className="review-game-open">Открыть разбор ↗</span></div>
        <div className="review-game-board" aria-label="Финальная позиция партии">
          <MiniBoard fen={finalPosition(game)} width={Math.min(180, Math.max(130, window.innerWidth - 225))} />
        </div>
      </Link>) : <p className="review-empty">Завершённых партий пока нет. Сыграйте партию против Maia на странице «Игра».</p>}
    </section>
  </main>;

  return <main className="review-page review-page--detail">
    <header className="review-heading"><Link className="review-back" to="/review">← Все партии</Link>
      <span className="review-kicker">РАЗБОР ПАРТИИ</span><h1>{selectedGame ? resultLabel(selectedGame) : "Загружаем партию"}</h1>
      {selectedGame && <p className="review-meta">
        <span className="review-meta-elo">Maia {selectedGame.elo ?? "—"}</span>
        <span className="review-meta-divider" aria-hidden="true" />
        <time className="review-meta-date" dateTime={gameTimestamp(selectedGame.playedAt)?.toISOString()}
          title={gameTimestamp(selectedGame.playedAt)?.toLocaleString("ru-RU")}>{formatGameDate(selectedGame.playedAt)}</time>
      </p>}</header>
    {error && <p className="review-error" role="alert">{error}</p>}
    {!parsed && !error && <p className="review-loading" role="status">Загружаем ходы партии…</p>}
    {parsed && <section className="review-report">
      <div className="review-stage" role="status"><strong>{stage}</strong><span>{selected + 1} / {plies.length} ходов</span>
        <div className="review-stage-track"><span style={{ width: `${Math.round(((selected + 1) / Math.max(1, plies.length)) * 100)}%` }} /></div></div>
      {report && <><div className="review-summary">
        {[["w", "white", "Белые"], ["b", "black", "Чёрные"]].map(([color, key, fallback]) =>
          <button type="button" key={key} className={`review-player ${side === color ? "is-selected" : ""}`} onClick={() => setSide(color)}>
            <span>{parsed.headers[color === "w" ? "White" : "Black"] || fallback}</span>
            <strong>{report[key].accuracy == null ? "—" : `${report[key].accuracy}%`}</strong>
            <small>точность · {report[key].moves_scored} оценено</small>
          </button>)}</div>
        <div className="review-grade-summary">{Object.entries(GRADES).map(([key, [symbol, label]]) =>
          <span key={key} className={`review-grade review-grade--${key}`}>{symbol} {label}: {report[side === "w" ? "white" : "black"].counts[key] || 0}</span>)}</div>
        <div className="review-chart-panel"><h2>Оценка партии</h2><EvalChart plies={analysisPlies} selected={selected} onSelect={navigateTo} /></div></>}
      <div className="review-workspace"><div className="review-board-column"><div className="review-board">
        <MiniBoard fen={practice && selected >= 0 ? plies[selected].fen : current}
          width={Math.min(600, Math.max(250, window.innerWidth - 360))}
          orientation={side === "w" ? "white" : "black"} arePiecesDraggable={practice}
          onPieceDrop={tryPracticeMove} /></div>
        {selected >= 0 && ["inaccuracy", "mistake", "blunder"].includes(plies[selected]?.classification) &&
          <div className="review-practice"><button type="button" onClick={startPractice} disabled={practiceLoading}>
            {practiceLoading ? "Ищем лучший ход…" : "Переиграть ошибку"}</button>
            {practice && <button type="button" onClick={() => setPracticeMessage(`Лучший ход: ${bestMove.san}`)}>Показать ответ</button>}
            {practice && <p>Сделайте лучший ход на доске.</p>}{practiceMessage && <p role="status">{practiceMessage}</p>}
          </div>}</div>
        <div className="review-moves-panel"><h2>Ходы партии</h2>
          <div className="review-moves">{plies.map((ply, index) => {
            const grade = GRADES[ply.classification] || (report && ply.analysis?.eval_cp == null ? ["…", "Без оценки Stockfish"] : null);
            return <button type="button" key={index} onClick={() => navigateTo(index)}
              className={`review-move ${index === selected ? "is-selected" : ""}`}>
              <span>{ply.move_no}{index % 2 ? "..." : "."} {ply.san}</span>
              {grade && <span className={`review-move-grade review-grade--${ply.classification || "unscored"}`}>{grade[0]} {grade[1]}</span>}
            </button>;
          })}</div>
          <div className="review-navigation"><button type="button" onClick={() => navigateTo(-1)}>⏮ Начало</button>
            <button type="button" disabled={selected < 0} onClick={() => navigateTo(selected - 1)}>←</button>
            <button type="button" onClick={() => { if (selected >= plies.length - 1) setSelected(-1); setPlaying((value) => !value); }}>
              {playing ? "Пауза" : "▶ Воспроизвести"}</button>
            <button type="button" disabled={selected >= plies.length - 1} onClick={() => navigateTo(selected + 1)}>→</button>
            <span>{selected < 0 ? "Начало партии" : `${selected + 1} / ${plies.length}`}</span></div>
        </div></div>
    </section>}
  </main>;
}
