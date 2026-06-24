import { useState, useRef } from 'react'

export default function FenBar({ onLoadFen }) {
  const [fenInput, setFenInput] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
  const fileRef = useRef(null)

  const handleLoad = () => {
    if (fenInput.trim()) onLoadFen(fenInput.trim())
  }

  const handleFile = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const line = ev.target.result.trim().split('\n')[0].trim()
      if (line) {
        setFenInput(line)
        onLoadFen(line)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  return (
    <div className="fen-bar">
      <input
        type="text"
        value={fenInput}
        onChange={e => setFenInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && handleLoad()}
        placeholder="Вставьте FEN..."
      />
      <button onClick={handleLoad}>Загрузить</button>
      <button className="fen-file-btn" onClick={() => fileRef.current?.click()} title="Загрузить из файла">📂</button>
      <input type="file" ref={fileRef} accept=".fen,.txt" style={{ display: 'none' }} onChange={handleFile} />
    </div>
  )
}
