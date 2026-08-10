# File: src/doc_agent/ingest/preprocess.py
"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# A1 Section 2 "Preparing the scans" table — these thresholds are the ones declared there.
_SKEW_ROTATE_THRESHOLD_DEG = 0.5
_BLANK_INK_PIXEL_RATIO = 0.002  # below this share of foreground pixels, treat the page as blank/separator


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Projection-profile skew estimate: search a bounded range of realistic scanner-skew angles
    (+-10deg) and pick the one that makes the horizontal ink-row histogram most 'peaky' (text lines
    stacked at sharp, well-separated rows) -- what real scanner skew actually produces. This replaces
    an earlier cv2.minAreaRect()-over-the-whole-page estimate, which was empirically found (by
    comparing processed output against the original scans on the real corpus) to occasionally report
    angles of several degrees, or even close to 90deg, on pages that were already straight -- a known
    failure mode of minAreaRect's angle convention on an aggregate ink-pixel cloud whose orientation
    is ambiguous near the ~45deg/~90deg boundaries. Restricting the search to +-10deg makes that
    catastrophic misrotation structurally impossible: 90deg is simply never a candidate."""
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    small = cv2.resize(fg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # speed; skew doesn't need full-res
    h, w = small.shape
    center = (w / 2, h / 2)

    best_angle, best_score = 0.0, -1.0
    for angle_tenths in range(-100, 101, 2):  # -10.0 .. +10.0 degrees, 0.2-degree steps
        angle = angle_tenths / 10.0
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        score = row_sums.var()  # sharp, well-separated text-line rows -> high row-sum variance
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Projection-profile skew estimate (see _estimate_skew_angle), rotate if > 0.5deg.
    A1: 'Border/margin cropping followed by Hough-transform or CCA skew detection ... automatic rotation
    exceeding 0.5deg.'
    """
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(fg)
    if coords is None or len(coords) < 50:
        return gray
    angle = _estimate_skew_angle(gray)
    if abs(angle) <= _SKEW_ROTATE_THRESHOLD_DEG:
        return gray
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _is_noisy(gray: np.ndarray) -> bool:
    """Cheap noise proxy: high-frequency energy via Laplacian variance, normalised by image size."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) > 500.0


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving filter, applied only to pages flagged as noisy — A1: 'no aggressive smoothing
    by default' to preserve thin conjuncts (যুক্তাক্ষর) and matras."""
    if _is_noisy(gray):
        return cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    return gray


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding (Sauvola-style local binarisation) as primary, Otsu as fallback
    for near-uniform pages (A1: 'Adaptive Gaussian as primary, with Otsu as fallback for uniform pages')."""
    std = float(gray.std())
    if std < 20.0:  # near-uniform illumination -> a single global threshold is stable enough
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=15
    )


def _ink_pixel_ratio(binarized: np.ndarray) -> float:
    ink = np.count_nonzero(binarized == 0)  # ink is dark -> 0 after THRESH_BINARY
    return ink / binarized.size


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing: deskew -> denoise (conditional) -> binarize -> blank-page filter.
    Writes cleaned images to paths.processed_dir, preserving each page's doc_id subfolder, and
    returns an updated Page list (blank/separator pages are dropped, per A1 Section 2)."""
    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    out_pages: list[Page] = []
    dropped = 0

    for page in pages:
        img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image for page {page.id} at {page.image_path}")

        img = _deskew(img)
        img = _denoise(img)
        binarized = _binarize(img)

        if _ink_pixel_ratio(binarized) < _BLANK_INK_PIXEL_RATIO:
            dropped += 1
            continue

        out_dir = processed_dir / page.doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{page.id}.png"
        cv2.imwrite(str(out_path), binarized)
        out_pages.append(Page(id=page.id, image_path=str(out_path), doc_id=page.doc_id))

    logger.info(f"preprocessed {len(out_pages)} pages, dropped {dropped} blank/separator pages")
    return out_pages