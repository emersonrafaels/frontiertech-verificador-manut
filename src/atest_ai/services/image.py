"""Pré-processamento e proveniência da imagem enviada ao modelo."""

from __future__ import annotations

import hashlib
import io
from PIL import Image, ImageOps

from atest_ai.config import ImageConfig


def prepare_image(original: bytes, cfg: ImageConfig, file_name: str = "") -> tuple[bytes, dict]:
    """Valida, normaliza e reduz a imagem antes do envio ao modelo.

    Aplica orientação EXIF, converte para RGB, limita o maior lado e
    recodifica em JPEG, registrando hashes para proveniência.

    Args:
        original: Bytes da imagem original (arquivo local ou upload da API).
        cfg: Limites e parâmetros de pré-processamento.
        file_name: Nome do arquivo de origem, apenas para proveniência.

    Returns:
        Tupla (bytes JPEG preparados, metadados de proveniência).

    Raises:
        ValueError: Se a imagem violar formato, tamanho ou limites configurados.
    """
    if len(original) > cfg.max_original_bytes:
        raise ValueError(f"File exceeds the local limit of {cfg.max_original_bytes} bytes.")
    with Image.open(io.BytesIO(original)) as img:
        if img.format not in cfg.accepted_formats:
            raise ValueError(f"Use {', '.join(sorted(cfg.accepted_formats))}. PDFs must be rendered first.")
        if getattr(img, "n_frames", 1) != 1:
            raise ValueError("Animated images are not accepted.")
        width, height = img.size
        if width * height > cfg.max_pixels:
            raise ValueError(f"Image exceeds {cfg.max_pixels} pixels.")
        image = ImageOps.exif_transpose(img).convert("RGB")
        image.thumbnail((cfg.max_side, cfg.max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=cfg.jpeg_quality)
        converted = buffer.getvalue()
        if len(converted) > cfg.max_prepared_bytes:
            raise ValueError("Prepared image exceeds the configured limit; adjust processing.")
        return converted, {
            "file_name": file_name,
            "sha256_original": hashlib.sha256(original).hexdigest(),
            "sha256_analyzed_image": hashlib.sha256(converted).hexdigest(),
            "original_dimensions": [width, height],
            "analyzed_dimensions": list(image.size),
            "preprocessing": f"EXIF_orientation_RGB_thumbnail_{cfg.max_side}_JPEG{cfg.jpeg_quality}",
        }
