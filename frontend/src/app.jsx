import { useState, useRef, useCallback } from "react";
import TopBar from "./components/TopBar.jsx";
import ChessboardComponent from "./components/Chessboard.jsx";
import MoveHistory from "./components/MoveHistory.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import { fetchGigaChatAnalysis } from "./api.js";

export default function App() {
  const boardRef = useRef(null);
  const [boardState, setBoardState] = useState({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    moveHistory: [],
    turn: "w",
    positionSnapshots: [],
    viewIndex: -1,
    isViewMode: false,
  });

  const handleAnalyze = useCallback(async () => {
    return await fetchGigaChatAnalysis(boardState.fen);
  }, [boardState.fen]);

  // === НОВАЯ ФУНКЦИЯ ДЛЯ ОБРАБОТКИ ФАЙЛОВ ===
  const handleFileUpload = (event) => {
    const file = event.target.files;
    if (!file) return;

    const extension = file.name.split(".").pop().toLowerCase();

    if (extension === "png") {
      // Логика для изображений (передача в LLaVA / backend)
      console.log("Загружено изображение:", file.name);
      alert(
        "Картинка загружена! Здесь будет вызов API для распознавания доски.",
      );
      // TODO: Добавить отправку FormData на ваш эндпоинт распознавания
    } else if (extension === "fen" || extension === "pgn") {
      // Логика для текстовых шахматных форматов
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result.trim();
        if (boardRef.current && boardRef.current.handleLoadFen) {
          // Если это PGN, парсинг лучше делать на бэкенде, но FEN можно сразу отдать доске
          boardRef.current.handleLoadFen(content);
        }
      };
      reader.readAsText(file);
    }

    // Очищаем значение, чтобы можно было загрузить тот же файл повторно
    event.target.value = null;
  };

  return (
    <>
      <TopBar />
      <div className="main-area">
        {/* Передаем функцию onImport в левую панель */}
        <MoveHistory
          moveHistory={boardState.moveHistory}
          positionSnapshots={boardState.positionSnapshots}
          viewIndex={boardState.viewIndex}
          isViewMode={boardState.isViewMode}
          onNavigate={(dir) => boardRef.current?.onNavigate(dir)}
          onImport={handleFileUpload}
        />

        <ChessboardComponent ref={boardRef} onStateChange={setBoardState} />
      </div>
    </>
  );
}
