import { useState, useRef, useCallback, useEffect } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import TopBar from './components/TopBar.jsx'
import MoveHistory from './components/MoveHistory.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import FenBar from './components/FenBar.jsx'
import { fetchMaiaMove, fetchStockfishAnalysis, fetchGigaChatAnalysis, saveMoveToDataset } from './api.js'

const PIECE_VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }

const userId = localStorage.getItem('sfedu_user_id') || 'user_' + Math.random().toString(36).substr(2, 9)
localStorage.setItem('sfedu_user_id', userId)
let gameId = 'game_' + Date.now()

function getMaterial(game) {
  const board = game.board()
  let white = 0, black = 0
  for (const row of board) {
    for (const cell of row) {
      if (!cell) continue
      const val = PIECE_VALUES[cell.type]
      if (cell.color === 'w') white += val
      else black += val
    }
  }
  return { white, black, diff: white - black }
}

function getCapturedPieces(history, game) {
  const tempGame = new Chess()
  const capturedByWhite = []
  const capturedByBlack = []
  for (const san of history) {
    const move = tempGame.move(san)
    if (move) {
      if (move.captured) {
        if (move.color === 'w') capturedByWhite.push(move.captured)
        else capturedByBlack.push(move.captured)
      }
    }
  }
  return { capturedByWhite, capturedByBlack }
}

export default function App() {
  const gameRef = useRef(new Chess())
  const [fen, setFen] = useState(gameRef.current.fen())
  const [moveHistory, setMoveHistory] = useState([])
  const [moveNumber, setMoveNumber] = useState(1)
  const [turn, setTurn] = useState('w')
  const [gameStatus, setGameStatus] = useState('Ваш ход')
  const [boardFlipped, setBoardFlipped] = useState(false)
  const [elo, setElo] = useState(1500)
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [lastMove, setLastMove] = useState(null)
  const [positionSnapshots, setPositionSnapshots] = useState([])
  const [viewIndex, setViewIndex] = useState(-1)
  const [isViewMode, setIsViewMode] = useState(false)
  const [selectedSquare, setSelectedSquare] = useState(null)
  const [legalMovesForSelected, setLegalMovesForSelected] = useState([])
  const [boardWidth, setBoardWidth] = useState(560)

  const game = gameRef.current

  useEffect(() => {
    const updateSize = () => {
      const topH = 58
      const availH = window.innerHeight - topH
      const availW = window.innerWidth * 0.6
      const size = Math.floor(Math.min(availW - 60, availH - 160))
      setBoardWidth(Math.max(320, Math.min(size, 640)))
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  const updateStatus = useCallback(() => {
    const g = gameRef.current
    if (g.isCheckmate()) {
      setGameStatus(g.turn() === 'w' ? 'Мат — чёрные победили' : 'Мат — белые победили!')
    } else if (g.isStalemate()) {
      setGameStatus('Пат — ничья')
    } else if (g.isDraw()) {
      setGameStatus('Ничья')
    } else if (g.isThreefoldRepetition()) {
      setGameStatus('Ничья — тройное повторение')
    } else if (g.isInsufficientMaterial()) {
      setGameStatus('Ничья — недостаточно материала')
    } else if (g.isCheck()) {
      setGameStatus(g.turn() === 'w' ? 'Шах! Ваш ход' : 'Шах! Maia думает...')
    } else {
      setGameStatus(g.turn() === 'w' ? 'Ваш ход' : 'Maia думает...')
    }
    setTurn(g.turn())
  }, [])

  const buildMoveHistory = useCallback(() => {
    const g = gameRef.current
    const hist = g.history()
    const pairs = []
    for (let i = 0; i < hist.length; i += 2) {
      pairs.push({
        num: Math.floor(i / 2) + 1,
        w: hist[i],
        b: hist[i + 1] || null,
      })
    }
    setMoveHistory(pairs)
    setMoveNumber(pairs.length > 0 ? pairs[pairs.length - 1].num : 1)
  }, [])

  const takeSnapshot = useCallback(() => {
    setPositionSnapshots(prev => [...prev, gameRef.current.fen()])
    setViewIndex(-1)
    setIsViewMode(false)
  }, [])

  const onNavigate = useCallback((direction) => {
    const g = gameRef.current
    const snapshots = [...positionSnapshots, g.fen()]

    if (direction === 'first') {
      if (snapshots.length === 0) return
      setIsViewMode(true)
      setViewIndex(0)
      game.load(snapshots[0])
      setFen(snapshots[0])
    } else if (direction === 'prev') {
      if (snapshots.length === 0) return
      const currentIdx = isViewMode ? viewIndex : snapshots.length - 1
      const newIdx = Math.max(0, currentIdx - 1)
      setIsViewMode(true)
      setViewIndex(newIdx)
      game.load(snapshots[newIdx])
      setFen(snapshots[newIdx])
    } else if (direction === 'next') {
      if (!isViewMode) return
      const newIdx = viewIndex + 1
      if (newIdx >= snapshots.length) {
        setIsViewMode(false)
        setViewIndex(-1)
        const last = snapshots[snapshots.length - 1]
        game.load(last)
        setFen(last)
      } else {
        setViewIndex(newIdx)
        game.load(snapshots[newIdx])
        setFen(snapshots[newIdx])
      }
    } else if (direction === 'last') {
      setIsViewMode(false)
      setViewIndex(-1)
      const last = snapshots[snapshots.length - 1]
      game.load(last)
      setFen(last)
    }
  }, [positionSnapshots, isViewMode, viewIndex, game])

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT') return
      if (e.key === 'ArrowLeft') { e.preventDefault(); onNavigate('prev') }
      if (e.key === 'ArrowRight') { e.preventDefault(); onNavigate('next') }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onNavigate])

  const handleNewGame = useCallback(() => {
    game.reset()
    setFen(game.fen())
    setMoveHistory([])
    setMoveNumber(1)
    setTurn('w')
    setLastMove(null)
    setPositionSnapshots([])
    setViewIndex(-1)
    setIsViewMode(false)
    setGameStatus('Ваш ход')
    setSelectedSquare(null)
    setLegalMovesForSelected([])
    gameId = 'game_' + Date.now()
    setIsAiThinking(false)
  }, [game])

  const handleLoadFen = useCallback((fenStr) => {
    try {
      game.load(fenStr)
      setFen(game.fen())
      setMoveHistory([])
      setMoveNumber(1)
      setTurn(game.turn())
      setLastMove(null)
      setPositionSnapshots([game.fen()])
      setViewIndex(-1)
      setIsViewMode(false)
      setSelectedSquare(null)
      setLegalMovesForSelected([])
      updateStatus()
    } catch {
      alert('Неверный FEN')
    }
  }, [game, updateStatus])

  const makeAiMove = useCallback(async () => {
    if (game.isGameOver() || game.isDraw()) return
    setIsAiThinking(true)
    try {
      const data = await fetchMaiaMove(game.fen(), elo)
      if (data.error) {
        setIsAiThinking(false)
        return
      }
      const move = game.move({
        from: data.from,
        to: data.to,
        promotion: 'q',
      })
      if (move) {
        setFen(game.fen())
        setLastMove({ from: data.from, to: data.to })
        buildMoveHistory()
        takeSnapshot()
        updateStatus()
      }
    } catch (err) {
      console.error('Maia2 ошибка:', err)
      const legalMoves = game.moves({ verbose: true })
      if (legalMoves.length > 0) {
        const pick = legalMoves[Math.floor(Math.random() * legalMoves.length)]
        game.move(pick)
        setFen(game.fen())
        setLastMove({ from: pick.from, to: pick.to })
        buildMoveHistory()
        takeSnapshot()
        updateStatus()
      }
    }
    setIsAiThinking(false)
  }, [game, elo, buildMoveHistory, takeSnapshot, updateStatus])

  const onDrop = useCallback((sourceSquare, targetSquare) => {
    if (isViewMode) return false
    if (isAiThinking) return false
    if (game.turn() !== 'w') return false

    const move = game.move({
      from: sourceSquare,
      to: targetSquare,
      promotion: 'q',
    })
    if (move === null) return false

    setFen(game.fen())
    setLastMove({ from: sourceSquare, to: targetSquare })
    setSelectedSquare(null)
    setLegalMovesForSelected([])
    buildMoveHistory()
    takeSnapshot()
    updateStatus()

    saveMoveToDataset(game.fen(), sourceSquare + targetSquare, userId, gameId).catch(() => {})

    if (!game.isGameOver() && !game.isDraw()) {
      setTimeout(makeAiMove, 500)
    }

    return true
  }, [game, isViewMode, isAiThinking, buildMoveHistory, takeSnapshot, updateStatus, makeAiMove])

  const onSquareClick = useCallback((square) => {
    if (isViewMode) return
    if (isAiThinking) return
    if (game.turn() !== 'w') return

    const piece = game.get(square)

    if (selectedSquare) {
      if (piece && piece.color === game.turn()) {
        if (square === selectedSquare) {
          setSelectedSquare(null)
          setLegalMovesForSelected([])
          return
        }
        setSelectedSquare(square)
        const moves = game.moves({ square, verbose: true })
        setLegalMovesForSelected(moves.map(m => m.to))
        return
      }

      const move = game.move({ from: selectedSquare, to: square, promotion: 'q' })
      if (move) {
        setFen(game.fen())
        setLastMove({ from: selectedSquare, to: square })
        setSelectedSquare(null)
        setLegalMovesForSelected([])
        buildMoveHistory()
        takeSnapshot()
        updateStatus()
        saveMoveToDataset(game.fen(), selectedSquare + square, userId, gameId).catch(() => {})
        if (!game.isGameOver() && !game.isDraw()) {
          setTimeout(makeAiMove, 500)
        }
        return
      }

      setSelectedSquare(null)
      setLegalMovesForSelected([])
      return
    }

    if (piece && piece.color === game.turn()) {
      setSelectedSquare(square)
      const moves = game.moves({ square, verbose: true })
      setLegalMovesForSelected(moves.map(m => m.to))
    }
  }, [game, selectedSquare, isViewMode, isAiThinking, buildMoveHistory, takeSnapshot, updateStatus, makeAiMove])

  const handlePromotionPieceSelect = useCallback((piece, promoteFromSquare, promoteToSquare) => {
    return true
  }, [])

  const handleAnalyze = useCallback(async () => {
    return await fetchGigaChatAnalysis(game.fen())
  }, [game])

  const handleStockfishAnalyze = useCallback(async () => {
    const data = await fetchStockfishAnalysis(game.fen())
    return data
  }, [game])

  const material = getMaterial(game)
  const materialDiff = material.diff

  const customSquareStyles = {}
  if (lastMove) {
    customSquareStyles[lastMove.from] = { backgroundColor: 'rgba(34, 90, 115, 0.5)' }
    customSquareStyles[lastMove.to] = { backgroundColor: 'rgba(34, 90, 115, 0.7)' }
  }
  if (selectedSquare) {
    customSquareStyles[selectedSquare] = { backgroundColor: 'rgba(163, 196, 209, 0.6)' }
  }
  for (const sq of legalMovesForSelected) {
    const targetPiece = game.get(sq)
    if (!customSquareStyles[sq]) {
      if (targetPiece) {
        customSquareStyles[sq] = {
          backgroundImage: 'radial-gradient(circle, transparent 50%, rgba(0,0,0,0.18) 52%, rgba(0,0,0,0.18) 80%, transparent 82%)',
          backgroundPosition: 'center',
          backgroundSize: '100% 100%',
          backgroundRepeat: 'no-repeat',
        }
      } else {
        customSquareStyles[sq] = {
          backgroundImage: 'radial-gradient(circle, rgba(0,0,0,0.18) 15%, transparent 16%)',
          backgroundPosition: 'center',
          backgroundSize: '100% 100%',
          backgroundRepeat: 'no-repeat',
        }
      }
    }
  }

  return (
    <>
      <TopBar />
      <div className="main-area">
        <div className="board-section">
          <div className="board-wrapper">
            <div className="player-bar">
              <div className="rating-label">
                Противник (Maia2): <span>{elo}</span> ELO
              </div>
              <input
                type="range"
                min="1100"
                max="1900"
                step="100"
                value={elo}
                onChange={e => setElo(parseInt(e.target.value))}
                style={{ width: 100, accentColor: '#225A73', cursor: 'pointer' }}
              />
              <MaterialDisplay diff={-materialDiff} pieces={getCapturedPieces(game.history(), game).capturedByBlack} side="top" />
            </div>

            <Chessboard
              position={fen}
              onPieceDrop={onDrop}
              onSquareClick={onSquareClick}
              boardWidth={boardWidth}
              animationDuration={200}
              boardOrientation={boardFlipped ? 'black' : 'white'}
              showBoardNotation={true}
              areArrowsAllowed={true}
              onPromotionPieceSelect={handlePromotionPieceSelect}
              customBoardStyle={{
                border: '3px solid #225A73',
                boxShadow: '0 0 0 1px #1a4a5f, 0 10px 50px rgba(0, 0, 0, 0.7)',
                borderRadius: '6px',
              }}
              customDarkSquareStyle={{ backgroundColor: '#225A73' }}
              customLightSquareStyle={{ backgroundColor: '#F8FAFC' }}
              customSquareStyles={customSquareStyles}
            />

            <div className="player-bar">
              <div className="rating-label">Вы — Белые</div>
              <MaterialDisplay diff={materialDiff} pieces={getCapturedPieces(game.history(), game).capturedByWhite} side="bottom" />
            </div>

            <div className="controls-row">
              <div id="game-status" style={{ flex: 1 }}>{isAiThinking ? 'Maia думает...' : gameStatus}</div>
              <button className="ctrl-btn" style={{ background: '#8b5cf6' }} onClick={handleStockfishAnalyze}>🔍 Анализ</button>
              <button className="ctrl-btn" style={{ background: '#0d3550', border: '1px solid #225A73' }} onClick={() => setBoardFlipped(f => !f)}>↺ Доска</button>
              <button className="ctrl-btn" style={{ background: '#225A73' }} onClick={handleNewGame}>Новая игра</button>
            </div>

            <FenBar onLoadFen={handleLoadFen} />
          </div>
        </div>

        <MoveHistory
          moveHistory={moveHistory}
          positionSnapshots={positionSnapshots}
          viewIndex={viewIndex}
          isViewMode={isViewMode}
          onNavigate={onNavigate}
        />

        <ChatPanel
          onAnalyze={handleAnalyze}
          fen={fen}
        />
      </div>
    </>
  )
}

function MaterialDisplay({ diff, pieces, side }) {
  const PIECE_ICONS = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛' }

  let className = 'material-display'
  if (diff > 0) className += ' white-adv'
  else if (diff < 0) className += ' black-adv'
  else className += ' equal'

  const displayPieces = side === 'top' ? (diff < 0 ? pieces : []) : (diff > 0 ? pieces : [])

  return (
    <div className={className}>
      {Math.abs(diff) > 0 && <span style={{ fontWeight: 'bold' }}>+{Math.abs(diff)}</span>}
      {displayPieces.map((p, i) => (
        <span className="mat-icon" key={i}>{PIECE_ICONS[p] || ''}</span>
      ))}
    </div>
  )
}
