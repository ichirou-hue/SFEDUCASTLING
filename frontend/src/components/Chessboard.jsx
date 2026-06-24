import { useState, useRef, useCallback, useEffect, forwardRef, useImperativeHandle } from 'react'
import { Chessboard as ReactChessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import FenBar from './FenBar.jsx'
import { fetchMaiaMove, saveMoveToDataset } from '../api.js'

const userId = localStorage.getItem('sfedu_user_id') || 'user_' + Math.random().toString(36).substr(2, 9)
localStorage.setItem('sfedu_user_id', userId)
let gameId = 'game_' + Date.now()

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
          customBoardStyle={{
            border: '3px solid #225A73',
            boxShadow: '0 0 0 1px #1a4a5f, 0 10px 50px rgba(0, 0, 0, 0.7)',
            borderRadius: '6px',
          }}
          customDarkSquareStyle={{ backgroundColor: '#225A73' }}
          customLightSquareStyle={{ backgroundColor: '#F8FAFC' }}
          customSquareStyles={customSquareStyles}
        />

        <div className="controls-row">
          <button className="ctrl-btn" style={{ background: '#0d3550', border: '1px solid #225A73' }} onClick={() => setBoardFlipped(f => !f)}>↺ Доска</button>
          <button className="ctrl-btn" style={{ background: '#225A73' }} onClick={handleNewGame}>Новая игра</button>
        </div>

        <FenBar onLoadFen={handleLoadFen} />
      </div>
    </div>
  )
})

export default ChessboardComponent
