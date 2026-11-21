"""PDF -> per-page text. poppler (pdftotext/pdftoppm) + tesseract via subprocess; pypdf as a last-resort fallback.

Per page: use the digital text layer; if a page has (almost) no text or garbled text, OCR that page.
Only the first `max_pages` pages are read: the decision letter, indications form and 510(k) summary come first.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd, timeout, env=None):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)


def num_pages(path) -> int:
    if have("pdfinfo"):
        try:
            r = _run(["pdfinfo", str(path)], 60)
            m = re.search(rb"Pages:\s+(\d+)", r.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0


def digital_pages(path, first: int, last: int) -> list[str]:
    if have("pdftotext"):
        r = _run(["pdftotext", "-f", str(first), "-l", str(last), "-layout", "-enc", "UTF-8", str(path), "-"], 180)
        if r.returncode == 0:
            pages = r.stdout.decode("utf-8", "replace").split("\f")
            if pages and pages[-1] == "":
                pages = pages[:-1]
            return pages
    from pypdf import PdfReader

    rd = PdfReader(str(path))
    return [(rd.pages[i].extract_text() or "") for i in range(first - 1, min(last, len(rd.pages)))]


def looks_garbled(text: str) -> bool:
    if text.count("(cid:") > 20:
        return True
    visible = [c for c in text if not c.isspace()]
    if len(visible) < 200:
        return False
    alpha = sum(c.isalpha() for c in visible)
    return alpha / len(visible) < 0.45


def _otsu(a) -> int:
    import numpy as np

    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / max(hist.sum(), 1.0)
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(between))


def preprocess_png(png, mode: str | None) -> None:
    """Scan clean-up before tesseract.
      median_gray (default): 3x3 median filter + autocontrast, NO explicit binarisation. Measured on synthetic degraded scans
          (6 docs, 12 K-numbers per level): mild degradation word accuracy 0.92 -> 0.995 with every K-number still exact;
          poor scans 0.73 -> 0.94 (K-number recall unchanged at 9/12). Clean scans: no change.
      median_otsu: the same plus global Otsu binarisation. Cleans prose almost as well but it closes the loops of 0/6/9 digits
          (K250001 -> K250061, K117091 -> K117691), so it LOSES K-numbers; kept only for comparison.
      none: raw pdftoppm output.
    Tesseract's mean word confidence does NOT detect the digit errors (it went UP with the binarised variant), so
    validate on real scans by comparing K-numbers against the printed originals (`label-eval`), not by confidence."""
    if mode in (None, "", "none", False):
        return
    try:
        import numpy as np
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return
    im = Image.open(png).convert("L").filter(ImageFilter.MedianFilter(3))
    im = ImageOps.autocontrast(im, cutoff=1)
    if mode == "median_otsu":
        a = np.asarray(im)
        im = Image.fromarray(((a > _otsu(a)) * 255).astype("uint8"))
    im.save(png)


def _tsv_words(tsv: str) -> list[dict]:
    out = []
    for line in tsv.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        try:
            out.append({"blk": int(f[2]), "par": int(f[3]), "line": int(f[4]), "left": int(f[6]), "top": int(f[7]), "w": int(f[8]),
                        "h": int(f[9]), "conf": float(f[10]), "text": f[11]})
        except ValueError:
            continue
    return out


def _mean_conf(words: list[dict]) -> float | None:
    c = [w["conf"] for w in words if w["conf"] >= 0]
    return sum(c) / len(c) if c else None


def _words_to_text(words: list[dict]) -> str:
    """Rebuild page text from tesseract's word table: one line per OCR line, blank line between paragraphs/blocks."""
    lines, cur, key, prev_par = [], [], None, None
    for w in words:
        k = (w["blk"], w["par"], w["line"])
        if k != key:
            if cur:
                lines.append(" ".join(cur))
            if prev_par is not None and (w["blk"], w["par"]) != prev_par:
                lines.append("")
            cur, key, prev_par = [], k, (w["blk"], w["par"])
        cur.append(w["text"])
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines) + "\n"


_KTOK = re.compile(r"^(?P<lead>[\(\[]?)(?P<pre>BK|K)(?P<body>[0-9OoIlSsBZ]{5,7})(?P<trail>[\)\]\.,;:]*)$", re.I)
_KFULL = re.compile(r"^(?:BK|K)\d{6}$")


def reread_knumbers(png, words: list[dict], ocfg: dict) -> int:
    """K-numbers are the labels, and OCR digit confusions (0/6/9, 1/7, 5/6 ...) corrupt them silently. Re-read every K-number-looking
    word from a tight crop, upscaled 2x/3x/4x, with a digits-only whitelist and single-line page segmentation, and take a majority
    vote with the original reading. Returns the number of tokens whose text changed."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return 0
    from collections import Counter

    scales = ocfg.get("k_reread_scales", [2, 3, 4])
    psm = str(ocfg.get("k_reread_psm", 7))
    im = Image.open(png).convert("L")
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    changed = 0
    for w in words:
        m = _KTOK.match(w["text"])
        if not m:
            continue
        orig = (m.group("pre") + m.group("body")).upper().translate(_REPAIR_DIGITS)
        votes = [orig] if _KFULL.match(orig) else []
        pad = max(4, int(0.35 * w["h"]))
        box = (max(0, w["left"] - pad), max(0, w["top"] - pad), w["left"] + w["w"] + pad, w["top"] + w["h"] + pad)
        crop = im.crop(box)
        for sc in scales:
            c2 = ImageOps.expand(crop.resize((crop.width * sc, crop.height * sc), Image.LANCZOS), border=24, fill=255)
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "k.png")
                c2.save(f)
                r = _run(["tesseract", f, "stdout", "-l", ocfg["lang"], "--psm", psm, "-c", "tessedit_char_whitelist=BK0123456789"], 60, env)
            t = r.stdout.decode("utf-8", "replace").strip().replace(" ", "").upper()
            if _KFULL.match(t):
                votes.append(t)
        if not votes:
            continue
        cnt = Counter(votes)
        best = max(cnt.items(), key=lambda kv: (kv[1], kv[0] == votes[-1] if len(votes) > 1 else False))[0]
        new_text = m.group("lead") + best + m.group("trail")
        if new_text != w["text"]:
            if _KFULL.match(orig) and orig == best:
                continue
            w["text"] = new_text
            changed += 1
    return changed


_REPAIR_DIGITS = str.maketrans("OoIlSsBZ", "00115582")


def ocr_page_conf(path, page: int, ocfg: dict):
    """-> (text, mean word confidence 0-100 or None, number of K-number tokens changed by the digit re-read)"""
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "p")
        _run(["pdftoppm", "-r", str(ocfg["dpi"]), "-f", str(page), "-l", str(page), "-gray", "-png", str(path), prefix], ocfg["timeout_s"])
        imgs = sorted(Path(td).glob("p*.png"))
        if not imgs:
            return "", None, 0
        preprocess_png(imgs[0], ocfg.get("preprocess"))
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        base = os.path.join(td, "o")
        _run(["tesseract", str(imgs[0]), base, "-l", ocfg["lang"], "--psm", str(ocfg["psm"]), "tsv"], ocfg["timeout_s"], env)
        try:
            words = _tsv_words(Path(base + ".tsv").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return "", None, 0
        changed = reread_knumbers(imgs[0], words, ocfg) if ocfg.get("k_reread", False) else 0
        return _words_to_text(words), _mean_conf(words), changed


def ocr_page(path, page: int, ocfg: dict) -> str:
    return ocr_page_conf(path, page, ocfg)[0]


def _alnum(s: str) -> int:
    return sum(c.isalnum() for c in s)


def extract_pdf(path, tcfg: dict) -> dict:
    t0 = time.time()
    n = num_pages(path)
    if n <= 0:
        raise ValueError("unreadable PDF (0 pages)")
    n_read = min(n, tcfg["max_pages"])
    pages = digital_pages(path, 1, n_read)
    ocr_pages: list[int] = []
    ocr_conf: list[float] = []
    k_changed = 0
    ocfg = tcfg["ocr"]
    if ocfg["enabled"] and have("pdftoppm") and have("tesseract"):
        for i, t in enumerate(pages):
            if len(ocr_pages) >= ocfg["max_pages"]:
                break
            if _alnum(t) < ocfg["min_alnum_chars"] or looks_garbled(t):
                txt, conf, nk = ocr_page_conf(path, i + 1, ocfg)
                k_changed += nk
                if _alnum(txt) > _alnum(t):
                    pages[i] = txt
                    ocr_pages.append(i)
                    if conf is not None:
                        ocr_conf.append(round(conf, 1))
    return {
        "pages": pages,
        "n_pages": n,
        "n_pages_read": len(pages),
        "ocr_pages": ocr_pages,
        "k_reread_changed": k_changed,   # K-number tokens whose reading was changed by the digit re-read vote
        "ocr_conf": ocr_conf,          # mean tesseract word confidence (0-100) of each OCR'd page
        "n_chars": sum(len(p) for p in pages),
        "seconds": round(time.time() - t0, 3),
    }
