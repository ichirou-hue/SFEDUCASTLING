"""Endpoint'ы компьютерного зрения: распознавание доски через LLaVA."""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File

from backend.api_gateway.state import llava_model, load_llava, extract_fen_from_image

router = APIRouter(tags=["vision"])


@router.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Распознаёт шахматную позицию с загруженного изображения доски."""
    global llava_model
    if llava_model is None:
        if not load_llava():
            return {
                "error": "LLaVA не загружена. Проверьте что transformers и torch установлены.",
                "hint": "pip install transformers torch pillow",
            }

    tmp_path = None
    try:
        contents = await file.read()

        filename = file.filename or "image.png"
        ext = filename.split(".")[-1].lower()
        if ext not in ["png", "jpg", "jpeg", "bmp", "webp"]:
            ext = "png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        fen = extract_fen_from_image(tmp_path)

        if fen.startswith("ERROR"):
            return {"error": fen.replace("ERROR: ", ""), "fen": None}

        return {"fen": fen, "message": "Позиция распознана!"}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
