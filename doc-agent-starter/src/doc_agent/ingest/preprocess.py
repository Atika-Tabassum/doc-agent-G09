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


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Hough-transform / connected-component skew estimate restricted to text regions, rotate if > 0.5deg.
    A1: 'Border/margin cropping followed by Hough-transform or CCA skew detection ... automatic rotation
    exceeding 0.5deg.'
    """
    # foreground = ink (works on the raw grayscale scan, before binarization)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(fg)
    if coords is None or len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angle convention: normalise to [-45, 45]
    if angle < -45:
        angle = 90 + angle
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
