"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa

import cv2
import pytesseract
from pytesseract import Output 
from ..logging_conf import get_logger

logger = get_logger(__name__)


def _tess_config(cfg: dict) -> tuple[str, str]:
    ocr_cfg = cfg.get("ocr", {})
    lang = ocr_cfg.get("lang", "ben")
    oem = ocr_cfg.get("oem", 1)
    psm = ocr_cfg.get("psm", 3)
    return lang, f"--oem {oem} --psm {psm}"

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions. IMPLEMENT."""
    lang, config = _tess_config(cfg)
    score_thr = cfg.get("layout", {}).get("score_thr", 0.0)

    regions: list[Region] = []
    for page in pages:
        img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image for page {page.id} at {page.image_path}")

        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)

        # group word-level boxes into lines by (block_num, par_num, line_num)
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
                lines[key] = {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "confs": []}
            else:
                lines[key]["x0"] = min(lines[key]["x0"], x)
                lines[key]["y0"] = min(lines[key]["y0"], y)
                lines[key]["x1"] = max(lines[key]["x1"], x + w)
                lines[key]["y1"] = max(lines[key]["y1"], y + h)
            if conf >= 0:  # tesseract reports -1 for non-text detections
                lines[key]["confs"].append(conf)

        for line in lines.values():
            mean_conf = (sum(line["confs"]) / len(line["confs"]) / 100.0) if line["confs"] else 0.0
            if mean_conf < score_thr:
                continue
            bbox = (line["x0"], line["y0"], line["x1"] - line["x0"], line["y1"] - line["y0"])
            regions.append(Region(page_id=page.id, bbox=bbox, kind="text"))

    logger.info(f"detected {len(regions)} line regions across {len(pages)} pages")
    return regions

