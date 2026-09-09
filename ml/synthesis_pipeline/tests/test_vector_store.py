import os
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_root = os.path.abspath(os.path.join(current_dir, ".."))
rag_dir = os.path.join(pipeline_root, "rag_module")

if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from vector_store import ChessVectorStore
from clip_encoder import ChessPositionEncoder


def run_vector_store_test():
    test_db_dir = os.path.join(current_dir, "test_chroma_db")
    
    # Очищаем тестовую БД перед запуском, если осталась от прошлых тестов
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

    print("[*] Инициализация энкодера и тестовой БД...")
    weights_path = os.path.expanduser("~/SFEDUCASTLING/ml/models/chessclip/chessclip.pt")
    encoder = ChessPositionEncoder(weights_path=weights_path)
    store = ChessVectorStore(
        persist_directory=test_db_dir,
        collection_name="test_positions",
        encoder=encoder
    )

    test_positions = [
        {
            "fen": "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "move": "d4",
            "eval": 0.35,
            "game_id": "game_sicilian_open",
            "move_number": 3,
            "comment": "Белые вскрывают центр ходом d4."
        },
        {
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "move": "Nf3",
            "eval": 0.28,
            "game_id": "game_sicilian_start",
            "move_number": 2,
            "comment": "Развитие коня с подготовкой d4."
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "move": "c5",
            "eval": 0.15,
            "game_id": "game_french",
            "move_number": 1,
            "comment": "Сицилианская защита c5."
        }
    ]

    print("\n--- Добавление позиций в тестовую ChromaDB ---")
    store.add_positions_batch(test_positions)
    assert store.collection.count() == 3, f"Ожидалось 3 записи, получено: {store.collection.count()}"

    query_fen = "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"
    print(f"\n--- Поиск похожих позиций для FEN: {query_fen} ---")
    matches = store.query_similar(query_fen, n_results=2)

    assert len(matches) > 0, "Поиск не вернул результатов!"
    for i, match in enumerate(matches, 1):
        print(f"\nСовпадение #{i}:")
        print(f"  Дистанция: {match['distance']}")
        print(f"  Рекомендуемый ход: {match['move']}")
        print(f"  Оценка: {match['eval']}")
        print(f"  Комментарий: {match['comment']}")

    print("\n[✓] Тест успешно пройден!")

    # Удаляем временную тестовую базу
    shutil.rmtree(test_db_dir)


if __name__ == "__main__":
    run_vector_store_test()