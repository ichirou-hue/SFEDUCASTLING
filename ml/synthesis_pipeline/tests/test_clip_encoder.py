import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_root = os.path.abspath(os.path.join(current_dir, ".."))
rag_dir = os.path.join(pipeline_root, "rag_module")

if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from clip_encoder import ChessPositionEncoder


def test_encoder():
    print("[*] Запуск теста ChessPositionEncoder...")
    weights_path = os.path.expanduser("~/SFEDUCASTLING/ml/models/chessclip/chessclip.pt")
    encoder = ChessPositionEncoder(weights_path=weights_path)

    # 1. Проверка стартовой позиции
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    vec_start = encoder.get_embedding(start_fen)

    assert isinstance(vec_start, list), "Эмбеддинг должен быть списком"
    assert len(vec_start) == 512, f"Ожидалась размерность 512, получено {len(vec_start)}"

    # 2. Проверка L2-нормализации (|v| = 1.0)
    norm = math.sqrt(sum(x * x for x in vec_start))
    assert abs(norm - 1.0) < 1e-4, f"Вектор должен быть L2-нормализован. Норма: {norm}"

    # 3. Проверка другой позиции (Сицилианка)
    sicilian_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
    vec_sicilian = encoder.get_embedding(sicilian_fen)

    # Вычисление косинусного сходства (для нормализованных векторов = скалярное произведение)
    similarity = sum(a * b for a, b in zip(vec_start, vec_sicilian))
    print(f"[+] Размерность вектора: {len(vec_start)}")
    print(f"[+] Длина вектора (L2-норма): {norm:.4f}")
    print(f"[+] Косинусное сходство между 1.e4 и стартовой: {similarity:.4f}")

    assert similarity < 0.999, "Разные позиции не должны возвращать одинаковый вектор"
    assert similarity > 0.500, "Смежные начальные позиции должны иметь высокое сходство"

    print("[✓] Тест ChessPositionEncoder успешно пройден!\n")


if __name__ == "__main__":
    test_encoder()