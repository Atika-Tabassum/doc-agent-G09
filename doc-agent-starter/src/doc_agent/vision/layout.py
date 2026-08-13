"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa

import json
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output 
from ..logging_conf import get_logger

logger = get_logger(__name__)

_LAYOUT_META = "layout_meta.jsonl"
_MIN_Y_TOLERANCE_PX = 5
# Noise-region floor: a genuine text line at typical scan DPI (see A1's 300 DPI corpus) is never
# this small -- even a lone short glyph (a page number digit, a punctuation mark) clears both of
# these comfortably. Real specks/dust/print defects that Tesseract occasionally boxes as a "word"
# sit well below them. Conservative by design: tuned to drop only unambiguous noise, not to force
# detected-line counts to match a particular gold transcription's line count (see vision/ocr.py's
# _align_lines for the part of this fix that handles genuine near-miss count differences).
_MIN_LINE_HEIGHT_PX = 8
_MIN_LINE_AREA_PX = 40


def _tess_config(cfg: dict) -> tuple[str, str]:
    ocr_cfg = cfg.get("ocr", {})
    lang = ocr_cfg.get("lang", "ben")
    oem = ocr_cfg.get("oem", 1)
    psm = ocr_cfg.get("psm", 4)
    return lang, f"--oem {oem} --psm {psm}"

def _group_words_into_lines(
    data: dict,
    score_thr: float,
    min_height_px: float = _MIN_LINE_HEIGHT_PX,
    min_area_px: float = _MIN_LINE_AREA_PX,
) -> list[dict]:
    """Group Tesseract's word-level boxes into lines by (block_num, par_num, line_num),
    preserving that provenance triple on every grouped line for the traceability sidecar.

    Drops two kinds of spurious "lines" before they ever reach layout_meta.jsonl / a Region:
    low mean-confidence groups (score_thr, unchanged) and groups whose merged bbox is smaller
    than min_height_px/min_area_px -- scan noise (ink specks, print defects) that Tesseract
    occasionally boxes as a low-word-count "line" distinct from any real text line, which would
    otherwise inflate the detected line count above what a human (or a gold transcription) would
    count."""
    lines: dict[tuple[int, int, int], dict] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        conf = float(data["conf"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in lines:
            lines[key] = {
                "block_num": key[0], "par_num": key[1], "line_num": key[2],
                "x0": x, "y0": y, "x1": x + w, "y1": y + h, "confs": [],
            }
        else:
            lines[key]["x0"] = min(lines[key]["x0"], x)
            lines[key]["y0"] = min(lines[key]["y0"], y)
            lines[key]["x1"] = max(lines[key]["x1"], x + w)
            lines[key]["y1"] = max(lines[key]["y1"], y + h)
        if conf >= 0:  # tesseract reports -1 for non-text detections
            lines[key]["confs"].append(conf)

    out = []
    for line in lines.values():
        mean_conf = (sum(line["confs"]) / len(line["confs"]) / 100.0) if line["confs"] else 0.0
        if mean_conf < score_thr:
            continue
        height = line["y1"] - line["y0"]
        width = line["x1"] - line["x0"]
        if height < min_height_px or (width * height) < min_area_px:
            continue
        line["bbox"] = (line["x0"], line["y0"], line["x1"] - line["x0"], line["y1"] - line["y0"])
        line["center_y"] = line["y0"] + (line["y1"] - line["y0"]) / 2.0
        out.append(line)
    return out


def _y_tolerance(lines: list[dict], cfg: dict) -> float:
    """Same-row tolerance in pixels, derived from median line height."""
    override = cfg.get("layout", {}).get("y_tolerance_px")
    if override is not None:
        return float(override)

    heights = [line["y1"] - line["y0"] for line in lines if line["y1"] > line["y0"]]
    if not heights:
        return float(_MIN_Y_TOLERANCE_PX)
    heights.sort()
    median_height = heights[len(heights) // 2]
    return max(_MIN_Y_TOLERANCE_PX, 0.4 * median_height)


def _spatial_reading_order(lines: list[dict], y_tolerance: float) -> list[dict]:
    """Sort grouped lines into canonical reading order (Y primary, X secondary) using a fixed band_ref_y."""
    if not lines:
        return []

    by_y = sorted(lines, key=lambda ln: ln["center_y"])

    bands: list[list[dict]] = []
    band_ref_y = None
    for ln in by_y:
        if band_ref_y is None or abs(ln["center_y"] - band_ref_y) > y_tolerance:
            bands.append([ln])
            band_ref_y = ln["center_y"]
        else:
            bands[-1].append(ln)

    ordered: list[dict] = []
    for band in bands:
        band.sort(key=lambda ln: ln["x0"])
        ordered.extend(band)
    return ordered


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text-line regions per page in correct reading order and save layout_meta.jsonl."""
    lang, config = _tess_config(cfg)
    layout_cfg = cfg.get("layout", {})
    score_thr = layout_cfg.get("score_thr", 0.0)
    min_line_height_px = layout_cfg.get("min_line_height_px", _MIN_LINE_HEIGHT_PX)
    min_line_area_px = layout_cfg.get("min_line_area_px", _MIN_LINE_AREA_PX)
    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))

    regions: list[Region] = []
    meta_rows: list[dict] = []

    for page in pages:
        img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image for page {page.id} at {page.image_path}")

        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)
        lines = _group_words_into_lines(data, score_thr, min_line_height_px, min_line_area_px)

        y_tolerance = _y_tolerance(lines, cfg)
        ordered_lines = _spatial_reading_order(lines, y_tolerance)

        for idx, line in enumerate(ordered_lines):
            region_id = f"{page.id}_r{idx:03d}"
            regions.append(Region(page_id=page.id, bbox=line["bbox"], kind="text"))
            meta_rows.append({
                "region_id": region_id,
                "page_id": page.id,
                "bbox": list(line["bbox"]),
                "block_num": line["block_num"],
                "par_num": line["par_num"],
                "line_num": line["line_num"],
            })

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / _LAYOUT_META, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"detected {len(regions)} line regions across {len(pages)} pages -> {processed_dir / _LAYOUT_META}")
    return regions