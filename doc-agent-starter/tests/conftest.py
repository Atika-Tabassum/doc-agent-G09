"""Shared fixtures for A2 tests: a tiny synthetic page-image corpus, generated on the fly so tests
never depend on the real (large, gitignored) Rachanabali corpus or network access."""
from __future__ import annotations

import pytest

# Single source of truth for the fail-fast ink-ratio guard below — imported rather than duplicated,
# so this fixture can never silently drift out of sync with preprocess.py's actual blank-page cutoff.
from doc_agent.ingest.preprocess import _BLANK_INK_PIXEL_RATIO

# Common TrueType font locations across Linux distros / CI images / macOS. PIL's ImageFont.load_default()
# (no args) fallback renders an ~6x11px bitmap font that is FAR too small to clear preprocess.py's
# blank-page threshold (measured ink ratio ~0.0016 vs. the 0.002 cutoff) — every synthetic page below
# silently gets classified as blank and dropped, which is exactly what caused every downstream
# ingest/OCR test that depends on this fixture to fail with "preprocessed 0 pages". Never use that
# zero-arg fallback here.
_CANDIDATE_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",   # macOS
    "C:\\Windows\\Fonts\\arial.ttf",                   # Windows
]


def _load_test_font(size: int):
    """Try real system TrueType fonts first (best OCR-recognisable glyph shapes). If none are
    installed, fall back to Pillow's own bundled font via the *scalable* `load_default(size=...)`
    API (Pillow >= 9.2) — still real character glyphs, just not a system font. Only the legacy
    zero-arg `load_default()` (tiny, ~6x11px, effectively unreadable) is intentionally never used."""
    from PIL import ImageFont

    for path in _CANDIDATE_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow too old to support a scalable default font — nothing usable left to try.
        return None


@pytest.fixture
def synthetic_corpus(tmp_path):
    """Writes a 2-page single-document corpus of rendered text-line images under tmp_path/data/raw,
    and returns a cfg dict pointed at tmp_path. Uses Latin text as a language-agnostic stand-in for
    plumbing tests — this checks the pipeline mechanics (box detection, cropping, chunk merging,
    metadata sidecars), not Bengali OCR quality, which belongs in notebooks/kb_demo.ipynb against the
    real corpus."""
    import numpy as np
    from PIL import Image, ImageDraw

    raw_dir = tmp_path / "data" / "raw" / "testwork"
    raw_dir.mkdir(parents=True)

    font = _load_test_font(40)
    if font is None:
        pytest.skip(
            "no usable TrueType font found (checked system paths and Pillow's own bundled "
            "load_default(size=...)) — install a font (e.g. fonts-dejavu-core) to run tests that "
            "depend on the synthetic_corpus fixture."
        )

    pages = [
        ["The quick brown fox jumps.", "Over the lazy dog again.", "Third line of test text."],
        ["Second page first line here.", "Second page second line too."],
    ]
    for pnum, lines in enumerate(pages, start=1):
        img = Image.new("L", (800, 600), color=255)
        d = ImageDraw.Draw(img)
        y = 40
        for line in lines:
            d.text((40, y), line, fill=0, font=font)
            y += 70
        img.save(raw_dir / f"{pnum}.png")

        # Fail fast and loud, right here, if this environment's rendering ever again produces a
        # near-blank page — instead of that surfacing three call-frames downstream as a confusing
        # "preprocessed 0 pages" failure in the ingest/OCR tests that consume this fixture.
        ink_ratio = float(np.count_nonzero(np.array(img) < 128)) / np.array(img).size
        assert ink_ratio > _BLANK_INK_PIXEL_RATIO * 5, (
            f"synthetic_corpus fixture page {pnum} has ink ratio {ink_ratio:.5f}, too close to "
            f"preprocess.py's blank-page threshold ({_BLANK_INK_PIXEL_RATIO}) in this environment — "
            f"the fixture's synthetic rendering isn't producing legible pages here."
        )

    cfg = {
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "index_dir": str(tmp_path / "data" / "processed" / "index"),
        },
        "layout": {"score_thr": 0.0},
        "ocr": {"model": "tesseract-ben", "lang": "eng", "oem": 1, "psm": 3},
        "index": {"chunk_tokens": 12, "overlap": 3},
        "grading_kit": {"labels_path": str(tmp_path / "grading_kit" / "labels.jsonl")},
    }
    return cfg