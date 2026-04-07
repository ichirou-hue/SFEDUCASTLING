import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Optional
import sys

import cv2
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

BASE_DIR = Path(__file__).resolve().parent

# чтобы работали импорты вида "from src...."
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from perception.converters.BoardToFEN.consts import (
    INFERENCE_CONFIG_PATH,
    INFERENCE_CONFIG_FILE_NAME,
)

from src.board.board import parse_board
from src.fen_converter.fen_converter import convert_pieces_to_fen
from src.model.dataset import PiecesDataset
from src.model.model import PieceClassifier
from src.utils.transforms import parse_config_transforms


PIECE_MAP_HUMAN = {
    "p": "bp",
    "r": "br",
    "n": "bn",
    "b": "bb",
    "q": "bq",
    "k": "bk",
    "P": "wP",
    "R": "wR",
    "N": "wN",
    "B": "wB",
    "Q": "wQ",
    "K": "wK",
}


def load_config() -> DictConfig:
    config_dir = (BASE_DIR / INFERENCE_CONFIG_PATH).resolve()

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        config = compose(config_name=INFERENCE_CONFIG_FILE_NAME)

    return config


def fen_rows_to_board(fen_rows: list[str]) -> list[list[str]]:
    board = []

    for row in fen_rows:
        parsed_row = []
        for ch in row:
            if ch.isdigit():
                parsed_row.extend([".."] * int(ch))
            else:
                parsed_row.append(PIECE_MAP_HUMAN[ch])
        board.append(parsed_row)

    return board


def print_human_board(fen_rows: list[str]) -> None:
    board = fen_rows_to_board(fen_rows)

    print("\nFEN:")
    print("/".join(fen_rows))

    print("\nПозиция:\n")

    rank = 8
    for row in board:
        print(f"{rank}  " + " ".join(f"{cell:>2}" for cell in row))
        rank -= 1

    print("\n   a  b  c  d  e  f  g  h\n")


def run_png_pipeline(image_path: str, config: DictConfig) -> Optional[list[str]]:
    image_path = Path(image_path).expanduser().resolve()

    print(f"[0] Старт. Аргумент: {image_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    print(f"[1] Путь к файлу: {image_path}")
    print("[2] Читаю изображение...")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    print("[3] Создаю модель...")
    model = PieceClassifier(**config.model_params)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[4] Устройство: {device}")

    model_path = (BASE_DIR / "final_model" / "model.pt").resolve()
    print(f"[5] Загружаю веса из: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    model.load_state_dict(
        torch.load(str(model_path), map_location=torch.device(device))
    )
    model.eval()

    print("[6] Готовлю transforms...")
    transforms = parse_config_transforms(config.transforms)

    print("[7] Ищу доску на изображении...")
    board_squares = parse_board(image)
    if board_squares is None:
        print("Доска не найдена")
        return None

    print(f"[8] Найдено клеток: {len(board_squares)}")
    print("[9] Готовлю батч...")
    pieces_batch = PiecesDataset.board_squares_to_pieces_dataset(
        board_squares, transforms
    )

    print("[10] Делаю inference...")
    predicted_labels = model.inference(pieces_batch)

    print("[11] Собираю результат...")
    board_rows_as_fen = convert_pieces_to_fen(predicted_labels)

    return board_rows_as_fen


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("python -m perception.converters.BoardToFEN.png_cli /путь/до/image.png")
        return

    image_path = sys.argv[1]
    config = load_config()
    result = run_png_pipeline(image_path, config)

    if result is None:
        return

    print_human_board(result)


if __name__ == "__main__":
    main()
