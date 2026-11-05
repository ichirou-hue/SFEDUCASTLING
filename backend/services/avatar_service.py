"""Безопасная обработка пользовательских аватаров.

Исходный файл никогда не сохраняется. Сервер:
1) ограничивает размер до чтения всего файла;
2) сверяет расширение, заявленный MIME и magic bytes;
3) открывает только JPEG/PNG/WebP;
4) отклоняет decompression bombs, анимацию и чрезмерные размеры;
5) декодирует пиксели и создаёт новое изображение без метаданных;
6) уменьшает до 512px и сохраняет только как WebP под серверным UUID.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import warnings
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
AVATAR_DIR = REPO_ROOT / "data" / "avatars"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_PIXELS = 4096 * 4096
OUTPUT_MAX_SIDE = 512

ALLOWED_EXTENSIONS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}
ALLOWED_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
ALLOWED_PIL_FORMATS = ["JPEG", "PNG", "WEBP"]
_SAFE_FILENAME_RE = re.compile(r"^[1-9][0-9]*-[0-9a-f]{32}\.webp$")


def _detected_magic_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Аватар слишком большой. Максимальный размер — 5 МБ.",
            )
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(status_code=422, detail="Файл аватара пуст.")
    return b"".join(chunks)


def _validate_declared_metadata(upload: UploadFile) -> tuple[str, str | None]:
    original_name = upload.filename or ""
    if not original_name or len(original_name) > 255:
        raise HTTPException(status_code=400, detail="Некорректное имя файла.")

    suffix = Path(original_name).suffix.lower()
    expected = ALLOWED_EXTENSIONS.get(suffix)
    if expected is None:
        raise HTTPException(
            status_code=415,
            detail="Разрешены только JPEG, PNG и WebP.",
        )

    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if content_type:
        mime_format = ALLOWED_MIME_TYPES.get(content_type)
        if mime_format is None:
            raise HTTPException(
                status_code=415,
                detail="Недопустимый MIME-тип изображения.",
            )
        if mime_format != expected:
            raise HTTPException(
                status_code=415,
                detail="Расширение файла не соответствует MIME-типу.",
            )
    return expected, content_type or None


async def sanitize_avatar(upload: UploadFile) -> tuple[bytes, dict]:
    expected_format, declared_mime = _validate_declared_metadata(upload)
    raw = await _read_limited(upload)

    magic_format = _detected_magic_format(raw)
    if magic_format is None or magic_format != expected_format:
        raise HTTPException(
            status_code=415,
            detail="Содержимое файла не соответствует разрешённому формату изображения.",
        )

    digest = hashlib.sha256(raw).hexdigest()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(BytesIO(raw), formats=ALLOWED_PIL_FORMATS) as probe:
                if probe.format != expected_format:
                    raise HTTPException(
                        status_code=415,
                        detail="Фактический формат изображения не совпадает с расширением.",
                    )

                width, height = probe.size
                if width < 1 or height < 1:
                    raise HTTPException(status_code=422, detail="Некорректные размеры изображения.")
                if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
                    raise HTTPException(
                        status_code=422,
                        detail="Разрешение аватара не должно превышать 4096×4096.",
                    )
                if int(getattr(probe, "n_frames", 1) or 1) != 1:
                    raise HTTPException(
                        status_code=422,
                        detail="Анимированные изображения для аватара не поддерживаются.",
                    )
                probe.verify()

            # После verify файл открывается заново. Разрешённые декодеры
            # задаются ещё раз, чтобы никакой другой Pillow plugin не участвовал.
            with Image.open(BytesIO(raw), formats=ALLOWED_PIL_FORMATS) as image:
                oriented = ImageOps.exif_transpose(image)
                oriented.load()

                target_mode = "RGBA" if "A" in oriented.getbands() else "RGB"
                pixels = oriented.convert(target_mode)

                # Новый объект гарантирует, что EXIF/XMP/ICC/comments из upload
                # не наследуются в итоговый публичный файл.
                clean = Image.new(target_mode, pixels.size)
                clean.paste(pixels)
                clean.info.clear()
                clean.thumbnail(
                    (OUTPUT_MAX_SIDE, OUTPUT_MAX_SIDE),
                    Image.Resampling.LANCZOS,
                )

                out = BytesIO()
                clean.save(
                    out,
                    format="WEBP",
                    quality=88,
                    method=6,
                )
                result = out.getvalue()

    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        logger.warning("avatar_validation decompression_bomb sha256=%s", digest)
        raise HTTPException(
            status_code=422,
            detail="Изображение отклонено из-за небезопасного размера.",
        ) from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        logger.warning("avatar_validation invalid_image sha256=%s", digest)
        raise HTTPException(
            status_code=422,
            detail="Не удалось безопасно декодировать изображение.",
        ) from None

    logger.info(
        "avatar_validation passed sha256=%s format=%s declared_mime=%s dimensions=%sx%s output_bytes=%s",
        digest,
        expected_format,
        declared_mime or "-",
        width,
        height,
        len(result),
    )
    return result, {
        "input_format": expected_format,
        "width": width,
        "height": height,
        "sha256": digest,
        "output_bytes": len(result),
    }


def store_avatar(user_id: int, payload: bytes) -> str:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{int(user_id)}-{secrets.token_hex(16)}.webp"
    path = AVATAR_DIR / filename
    with path.open("xb") as stream:
        stream.write(payload)
    return filename


def avatar_path(filename: str | None) -> Path | None:
    if not filename or not _SAFE_FILENAME_RE.fullmatch(filename):
        return None
    path = AVATAR_DIR / filename
    return path if path.is_file() else None


def delete_avatar_file(filename: str | None) -> None:
    if not filename or not _SAFE_FILENAME_RE.fullmatch(filename):
        return
    path = AVATAR_DIR / filename
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("avatar_cleanup failed filename=%s", filename)
