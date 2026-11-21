"""Render text pages to real PDFs for the PDF-pipeline tests: digital (text layer) or scanned (image only, needs OCR)."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

LETTER = (612, 792)
_FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"]


def _wrap(text: str, width: int) -> list[str]:
    out = []
    for line in text.split("\n"):
        out += textwrap.wrap(line, width) or [""]
    return out


def _safe(s: str) -> str:
    return s.replace("☒", "[x]").replace("☐", "[ ]")


def degrade_image(img, level: int, rng):
    """Make a clean page image look like a bad scan. level 0 = untouched; 1 mild fax/copier; 2 poor; 3 very poor.
    Rotation (skew), lost resolution, blur, sensor noise, uneven shading, speckle, JPEG artefacts."""
    if level <= 0:
        return img
    import io

    import numpy as np
    from PIL import Image, ImageFilter

    p = {1: dict(rot=1.0, scale=0.75, blur=0.5, noise=8, jpeg=70, speck=0.0005, shade=20),
         2: dict(rot=2.0, scale=0.55, blur=0.9, noise=18, jpeg=50, speck=0.002, shade=45),
         3: dict(rot=3.0, scale=0.42, blur=1.3, noise=30, jpeg=35, speck=0.006, shade=70)}[min(level, 3)]
    W, H = img.size
    img = img.rotate(rng.uniform(-p["rot"], p["rot"]), resample=Image.BICUBIC, fillcolor=255)
    img = img.resize((int(W * p["scale"]), int(H * p["scale"])), Image.BILINEAR).resize((W, H), Image.BILINEAR)   # lost resolution
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    a = np.asarray(img, dtype=np.float32)
    gy = np.linspace(0, 1, H, dtype=np.float32)[:, None] * rng.uniform(0.3, 1.0)
    gx = np.linspace(0, 1, W, dtype=np.float32)[None, :] * rng.uniform(0.0, 1.0)
    a = a - p["shade"] * (gy + gx) / 2                                        # uneven illumination
    nrng = np.random.default_rng(rng.randrange(2**31))
    a = a + nrng.normal(0, p["noise"], a.shape)
    mask = nrng.random(a.shape) < p["speck"]
    a[mask] = nrng.choice([0, 255], size=int(mask.sum()))                     # salt & pepper
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=p["jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("L")


def render_pdf(pages: list[str], path, scanned: bool = False, degrade: int = 0, seed: int = 0) -> None:
    import random

    from reportlab.pdfgen import canvas

    rng = random.Random(seed)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=LETTER)
    for text in pages:
        text = _safe(text)
        if not scanned:
            c.setFont("Courier", 9)
            y = LETTER[1] - 50
            for line in _wrap(text, 98)[:66]:
                c.drawString(40, y, line)
                y -= 11
        else:
            from PIL import Image, ImageDraw, ImageFont

            dpi = 200
            W, H = int(8.5 * dpi), int(11 * dpi)
            img = Image.new("L", (W, H), 255)
            dr = ImageDraw.Draw(img)
            font = None
            for f in _FONT_CANDIDATES:
                if os.path.exists(f):
                    font = ImageFont.truetype(f, 26)
                    break
            if font is None:
                font = ImageFont.load_default(size=26)
            y = 90
            for line in _wrap(text, 80)[:60]:
                dr.text((80, y), line, fill=0, font=font)
                y += 34
            from reportlab.lib.utils import ImageReader

            img = degrade_image(img, degrade, rng)
            c.drawImage(ImageReader(img), 0, 0, width=LETTER[0], height=LETTER[1])  # in-memory: one distinct image per page
        c.showPage()
    c.save()
