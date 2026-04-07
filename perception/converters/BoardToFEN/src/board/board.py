from __future__ import annotations

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional

from perception.converters.BoardToFEN.src.board.canny import canny_edge_detector
from perception.converters.BoardToFEN.src.board.consts import Canny
from perception.converters.BoardToFEN.src.board.board_utils import (
    split_board_image_to_squares,
    crop_image,
)

logger = logging.getLogger(__name__)


class Board:
    """
    Board class that represents a board detected in a given image
    """

    N_LINES = 9

    def __init__(self, board: np.ndarray):
        self.board = board

    @property
    def area(self) -> int:
        return self.board.shape[0] * self.board.shape[1]

    def __gt__(self, other: Board) -> bool:
        return self.area > other.area

    def split_board_into_squares(self):
        print("[7.4.1] Запускаю canny")
        edges = canny_edge_detector(
            self.board,
            low_thresh_ratio=Canny.low_thresh_ratio.value,
            high_thresh_ratio=Canny.high_thresh_ratio.value,
        )

        print("[7.4.2] Запускаю crop_image")
        cropped_image, cropped_edges, cropped_row_seq, cropped_col_seq = crop_image(
            self.board, edges, verbose=False
        )

        print("[7.4.3] После crop_image")
        print(f"[7.4.3] cropped_image shape: {cropped_image.shape}")
        print(f"[7.4.3] row lines: {len(cropped_row_seq)}")
        print(f"[7.4.3] col lines: {len(cropped_col_seq)}")

        print("[7.4.4] Режу доску на 64 клетки")
        board_squares = split_board_image_to_squares(
            cropped_image, cropped_row_seq, cropped_col_seq
        )

        print("[7.4.5] Нарезка завершена")
        return board_squares


def parse_board(image: np.ndarray) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    print("[7.1] Конвертирую BGR -> RGB")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[7.2] Ищу доску")
    board = detect_board(rgb_image)
    if board is None:
        logger.info("Couldn't detect a board")
        print("[7.3] Доска не найдена")
        return

    print("[7.4] Доска найдена, режу на клетки")
    board_squares = board.split_board_into_squares()

    print("[7.5] Клетки готовы")
    return board_squares


def detect_board(image: np.ndarray) -> Optional[Board]:
    """Detects the board inside the given image. The process is as follows:
        1. Convert the image from RGB to grayscale
        2. Sharpen the image using an enhanced sharpening kernel
        3. Convert the image to binary using thresholding
        4. Find all contours in the image
        5. Filter contours by area, height/width ratio. The board has to be of certain size, and has to be a square

    Args:
        image: the image in which we detect the board

    Returns:
        board: Board object of the detected board

    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, sharpen_kernel)

    threshold_image = cv2.threshold(sharpened_image, 10, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(
        threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    min_area_board = 6400
    height_width_error_ratio = 0.05
    board = None
    board_count = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area_board:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        height_width_ratio = h / w
        if (
            height_width_ratio <= 1 - height_width_error_ratio
            or height_width_ratio >= 1 + height_width_error_ratio
        ):
            continue

        roi = image[y : y + h, x : x + w]

        board_count += 1
        candidate_board = Board(roi)
        if board is None:
            board = candidate_board
        elif candidate_board > board:
            board = candidate_board

    logger.info(
        f"Detected {board_count} boards. Proceeding with the largest board detected"
    )

    return board


if __name__ == "__main__":
    _image = cv2.imread(
        r"C:\Users\GoMaN\Desktop\GoMaN\Projects\BoardToFEN\src\data\temp_data\lichess.png"
    )
    squares = parse_board(_image)

    fig, axs = plt.subplots(8, 8)
    for (i, j), square in squares.items():
        axs[i, j].imshow(square)

    plt.show()
