import { useState, useRef, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react'
import { Chessboard as ReactChessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import FenBar from './FenBar.jsx'
import { fetchMaiaMove, saveMoveToDataset } from '../api.js'

const userId = localStorage.getItem('sfedu_user_id') || 'user_' + Math.random().toString(36).substr(2, 9)
localStorage.setItem('sfedu_user_id', userId)
let gameId = 'game_' + Date.now()

let audioCtx = null
function playMoveSound() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  const now = audioCtx.currentTime

  const osc = audioCtx.createOscillator()
  const gain = audioCtx.createGain()
  const filter = audioCtx.createBiquadFilter()

  filter.type = 'lowpass'
  filter.frequency.setValueAtTime(1200, now)
  filter.frequency.exponentialRampToValueAtTime(300, now + 0.08)

  osc.type = 'triangle'
  osc.frequency.setValueAtTime(180, now)
  osc.frequency.exponentialRampToValueAtTime(60, now + 0.06)

  gain.gain.setValueAtTime(0.5, now)
  gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1)

  osc.connect(filter)
  filter.connect(gain)
  gain.connect(audioCtx.destination)

  osc.start(now)
  osc.stop(now + 0.12)
}

const ChessboardComponent = forwardRef(function ChessboardComponent({ onStateChange }, ref) {
  const gameRef = useRef(new Chess())
  const [fen, setFen] = useState(gameRef.current.fen())
  const [moveHistory, setMoveHistory] = useState([])
  const [turn, setTurn] = useState('w')
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
      g.load(snapshots[0])
      setFen(snapshots[0])
    } else if (direction === 'prev') {
      if (snapshots.length === 0) return
      const currentIdx = isViewMode ? viewIndex : snapshots.length - 1
      const newIdx = Math.max(0, currentIdx - 1)
      setIsViewMode(true)
      setViewIndex(newIdx)
      g.load(snapshots[newIdx])
      setFen(snapshots[newIdx])
    } else if (direction === 'next') {
      if (!isViewMode) return
      const newIdx = viewIndex + 1
      if (newIdx >= snapshots.length) {
        setIsViewMode(false)
        setViewIndex(-1)
        const last = snapshots[snapshots.length - 1]
        g.load(last)
        setFen(last)
      } else {
        setViewIndex(newIdx)
        g.load(snapshots[newIdx])
        setFen(snapshots[newIdx])
      }
    } else if (direction === 'last') {
      setIsViewMode(false)
      setViewIndex(-1)
      const last = snapshots[snapshots.length - 1]
      g.load(last)
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
        playMoveSound()
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
        playMoveSound()
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
    playMoveSound()

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
        playMoveSound()
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

  const handlePromotionPieceSelect = useCallback(() => {
    return true
  }, [])

  useImperativeHandle(ref, () => ({
    onNavigate,
    handleNewGame,
    getFen: () => gameRef.current.fen(),
  }))

  useEffect(() => {
    onStateChange({
      fen,
      moveHistory,
      turn,
      positionSnapshots,
      viewIndex,
      isViewMode,
    })
  }, [fen, moveHistory, turn, positionSnapshots, viewIndex, isViewMode, onStateChange])

  const cornerStyle = (radius) => ({ borderRadius: radius })
  const customSquareStyles = {}
  if (!boardFlipped) {
    customSquareStyles.a1 = cornerStyle('15px 0 0 0')
    customSquareStyles.h1 = cornerStyle('0 15px 0 0')
    customSquareStyles.a8 = cornerStyle('0 0 0 15px')
    customSquareStyles.h8 = cornerStyle('0 0 15px 0')
  } else {
    customSquareStyles.a8 = cornerStyle('15px 0 0 0')
    customSquareStyles.h8 = cornerStyle('0 15px 0 0')
    customSquareStyles.a1 = cornerStyle('0 0 0 15px')
    customSquareStyles.h1 = cornerStyle('0 0 15px 0')
  }
  if (lastMove) {
    customSquareStyles[lastMove.from] = { backgroundColor: 'rgba(201, 169, 110, 0.45)' }
    customSquareStyles[lastMove.to] = { backgroundColor: 'rgba(201, 169, 110, 0.6)' }
  }
  if (selectedSquare) {
    customSquareStyles[selectedSquare] = { backgroundColor: 'rgba(201, 169, 110, 0.55)' }
  }
  for (const sq of legalMovesForSelected) {
    const targetPiece = game.get(sq)
    if (!customSquareStyles[sq]) {
      if (targetPiece) {
        customSquareStyles[sq] = {
          backgroundImage: 'radial-gradient(circle, transparent 48%, rgba(100,80,40,0.25) 50%, rgba(100,80,40,0.25) 80%, transparent 82%)',
          backgroundPosition: 'center',
          backgroundSize: '100% 100%',
          backgroundRepeat: 'no-repeat',
        }
      } else {
        customSquareStyles[sq] = {
          backgroundImage: 'radial-gradient(circle, rgba(100,80,40,0.2) 15%, transparent 16%)',
          backgroundPosition: 'center',
          backgroundSize: '100% 100%',
          backgroundRepeat: 'no-repeat',
        }
      }
    }
  }

  return (
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
            style={{ width: 100, accentColor: '#C9A96E', cursor: 'pointer' }}
          />
        </div>

        <ReactChessboard
          position={fen}
          onPieceDrop={onDrop}
          onSquareClick={onSquareClick}
          boardWidth={boardWidth}
          animationDuration={200}
          boardOrientation={boardFlipped ? 'black' : 'white'}
          showBoardNotation={true}
          areArrowsAllowed={true}
          onPromotionPieceSelect={handlePromotionPieceSelect}
          customPieces={{
            wK: () => <img src="/pieces/white_king.svg" style={{ width: '100%', height: '100%' }} />,
            wQ: () => <img src="/pieces/white_queen.svg" style={{ width: '100%', height: '100%' }} />,
            wR: () => <img src="/pieces/white_rook.svg" style={{ width: '100%', height: '100%' }} />,
            wB: () => <img src="/pieces/white_bishop.svg" style={{ width: '100%', height: '100%' }} />,
            wN: () => <img src="/pieces/white_knight.svg" style={{ width: '100%', height: '100%' }} />,
            wP: () => <img src="/pieces/white_pawn.svg" style={{ width: '100%', height: '100%' }} />,
            bK: () => <img src="/pieces/black_king.svg" style={{ width: '100%', height: '100%' }} />,
            bQ: () => <img src="/pieces/black_queen.svg" style={{ width: '100%', height: '100%' }} />,
            bR: () => <img src="/pieces/black_rook.svg" style={{ width: '100%', height: '100%' }} />,
            bB: () => <img src="/pieces/black_bishop.svg" style={{ width: '100%', height: '100%' }} />,
            bN: () => <img src="/pieces/black_knight.svg" style={{ width: '100%', height: '100%' }} />,
            bP: () => <img src="/pieces/black_pawn.svg" style={{ width: '100%', height: '100%' }} />,
          }}
          customBoardStyle={{
            borderRadius: '15px',
            backgroundImage: 'linear-gradient(0deg, rgba(74, 178, 45, 0.43) 0%, rgba(74, 178, 45, 0.43) 100%), url(/textures/green-marble.png)',
            backgroundPosition: '-0.111px 0px',
            backgroundSize: '100.027% 100%',
            backgroundRepeat: 'no-repeat',
            backgroundColor: 'lightgray',
          }}
          customDarkSquareStyle={{
            borderLeft: '1px solid rgba(226, 213, 124, 0.85)',
            backgroundColor: 'rgba(223, 239, 252, 0.20)',
          }}
          customLightSquareStyle={{
            borderTop: '1px solid rgba(226, 213, 124, 0.85)',
            borderLeft: '1px solid rgba(226, 213, 124, 0.85)',
            backgroundImage: 'url(/textures/white-marble.png)',
            backgroundPosition: '50%',
            backgroundSize: 'cover',
            backgroundRepeat: 'no-repeat',
            backgroundColor: 'rgba(255, 255, 255, 0.75)',
          }}
          customSquareStyles={customSquareStyles}
        />

        <div className="controls-row">
          <button className="ctrl-btn" style={{ background: '#8B7340', border: '1px solid #C9A96E' }} onClick={() => setBoardFlipped(f => !f)}>↺ Доска</button>
          <button className="ctrl-btn" style={{ background: '#C9A96E' }} onClick={handleNewGame}>Новая игра</button>
        </div>

        <FenBar onLoadFen={handleLoadFen} />
      </div>
    </div>
  )
})

export default ChessboardComponent
