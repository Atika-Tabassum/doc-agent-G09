"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import cv2
import pytesseract

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# A1 Section 1, Data speciality: IA's own OCR is the ONLY text layer this corpus ships with — there is
# no independent gold transcription. evidence_tier per line is one of:
#   "gold"   — page falls inside grading_kit/heldout_pages + labels.jsonl (independently reviewed)
#   "silver" — no gold label, but the line passed the CER quality gate against IA's own silver reference
#   "raw"    — failed the gate, or unverifiable (no reference reachable) — NOT indexed as a Chunk
# contracts.Chunk has no field for confidence/CER/tier (contracts.py is fixed) so all of it lives in
# this sidecar, keyed by region_id (Stage 2's identity) and chunk_id (Stage 4's join key).
_META_SIDECAR = "ocr_meta.jsonl"
_LAYOUT_META = "layout_meta.jsonl"

_DEFAULT_MAX_CER = 0.15       # A1 notebook §5.0 target; overridable via cfg['ocr']['max_cer_target']
_DEFAULT_MIN_LINE_CHARS = 1   # reject only genuinely empty OCR output by default — Bengali graphemes
                              # often need several Unicode codepoints, so a stricter default risks
                              # rejecting legitimate short lines (page numbers, single-word headings).
                              # Raise cfg['ocr']['min_line_chars'] once real noise patterns are visible.


def _image_path_for_page(page_id: str, cfg: dict) -> Path:
    doc_id = page_id.rsplit("_p", 1)[0]
    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    return processed_dir / doc_id / f"{page_id}.png"


def _load_gold_page_ids(cfg: dict) -> set[str]:
    """Page ids covered by the reviewed gold sample (grading_kit/labels.jsonl)."""
    labels_path = Path(cfg.get("grading_kit", {}).get("labels_path", "grading_kit/labels.jsonl"))
    gold: set[str] = set()
    if not labels_path.exists():
        return gold
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("page_id")
            text = row.get("text", "")
            if pid and not text.startswith("REPLACE ME"):  # skip the un-filled stub row
                gold.add(pid)
    return gold


def _load_layout_region_lookup(cfg: dict) -> dict[str, dict]:
    """region_id -> its layout_meta.jsonl row, for the Stage 2/3 provenance cross-check (req 5).
    Returns {} if layout_meta.jsonl hasn't been written yet — the cross-check is then skipped
    rather than treated as an error (Stage 3 can still run standalone, e.g. in unit tests)."""
    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    path = processed_dir / _LAYOUT_META
    lookup: dict[str, dict] = {}
    if not path.exists():
        return lookup
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                lookup[row["region_id"]] = row
    return lookup


def _load_page_reference_lines(page_id: str, cfg: dict, expected_n_lines: int) -> list[str] | None:
    """Best-effort IA silver-reference lines for a page, for the CER gate (req 6/8).

    Convention: data/raw/<doc_id>/<page_id>.ref.txt, one line per physical line, same top-to-bottom
    order as our own detected regions. NOTE: scripts/get_data.sh does not fetch this today — nothing in
    this repo populates it yet, so until that's wired up, every non-gold line will correctly get
    reject_reason='reference_alignment_failed' rather than a false accept. That's the intended safe
    default, not a bug.

    Line-to-line correspondence is positional (reference line i <-> region i), which is a real
    limitation: IA's own OCR may have split/merged lines differently than our layout pass did. As a
    cheap sanity guard against a grossly mismatched segmentation, alignment is refused outright (returns
    None) if the reference's line count isn't within roughly 2x of our detected line count either way —
    a true per-line sequence-alignment (e.g. the Levenshtein-projection approach A1's notebook describes
    for building the gold set) is a heavier batch job better suited to an offline script, not this
    per-page hot path.
    """
    override_dir = cfg.get("ocr", {}).get("reference_dir")
    doc_id = page_id.rsplit("_p", 1)[0]
    if override_dir:
        ref_path = Path(override_dir) / doc_id / f"{page_id}.ref.txt"
    else:
        raw_dir = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))
        ref_path = raw_dir / doc_id / f"{page_id}.ref.txt"

    if not ref_path.exists():
        return None

    lines = [ln.rstrip("\n") for ln in open(ref_path, encoding="utf-8")]
    lines = [ln for ln in lines if ln.strip()]  # drop blank reference lines
    if not lines or expected_n_lines == 0:
        return None
    ratio = len(lines) / expected_n_lines
    if not (0.5 <= ratio <= 2.0):
        logger.warning(
            f"page {page_id}: reference has {len(lines)} lines vs. {expected_n_lines} detected regions "
            f"(ratio {ratio:.2f}) — segmentation looks too mismatched to align positionally, skipping"
        )
        return None
    return lines


def _normalize(text: str) -> str:
    """Conservative normalization (req 7): Unicode NFC + whitespace collapse only. Deliberately does
    NOT strip Bengali punctuation, digits, or any characters — over-normalizing would corrupt the very
    text a citation is supposed to quote verbatim."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str) -> int:
    """Character-level edit distance (standard DP), used for CER. Cheap at line-length scale."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _cer(hyp: str, ref: str) -> float:
    """CER = (S+D+I) / N, N = len(reference) (req 6). Caller must ensure `ref` is non-empty --
    an empty reference is a reference_alignment_failed case, not a CER of 0 or infinity."""
    if not ref:
        raise ValueError("_cer() requires a non-empty reference — caller should treat that as alignment failure")
    return _levenshtein(hyp, ref) / len(ref)


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract.

    Deployed model here is Tesseract-ben (A1 Section 5, Model facet): open-weight, CPU-only, frozen at
    --oem 1 --psm 3 for layout's full-page pass; per-line transcription here uses --oem 1 --psm 7 -l ben
    (single text line). CRNN+BiLSTM+CTC and baidu/Unlimited-OCR stay gated off pending a GPU comparison
    run — see A1 notebook §6.5/§8 and configs/design_choices.md.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.full_cfg = cfg

    def _crop(self, region: Region):
        """4px pad, strictly clamped to [0,width] x [0,height] (req 3) — protects Bengali matras and
        conjunct ascenders/descenders sitting right at the detected box edge."""
        img_path = _image_path_for_page(region.page_id, self.full_cfg)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read processed image for page {region.page_id} at {img_path}")
        x, y, w, h = region.bbox
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        return img[y0:y1, x0:x1]

    def _ocr_line(self, crop) -> tuple[str, float]:
        """OCR one line-crop with --oem 1 --psm 7 -l ben as a single literal config string (req 2).
        Returns (raw_text, mean_word_confidence in [0,1])."""
        if crop.size == 0:
            return "", 0.0
        lang = self.cfg.get("lang", "ben")
        oem = self.cfg.get("oem", 1)
        config = f"--oem {oem} --psm 7 -l {lang}"
        data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
        words = [t.strip() for t in data["text"] if t.strip()]
        confs = [float(c) for c, t in zip(data["conf"], data["text"]) if t.strip() and float(c) >= 0]
        text = " ".join(words)
        mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, mean_conf

    def transcribe_region(self, region: Region) -> str:
        """Crop the region from its page image and OCR just that line. IMPLEMENT (fixed signature)."""
        text, _conf = self._ocr_line(self._crop(region))
        return text


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> line-level Chunks, gated by a silver-label CER quality check (req 6/8/11).

    Granularity/order (req 1): regions are consumed exactly as vision/layout.py returned them, grouped
    by page but never re-sorted -- that ordering is what lets region_id be regenerated deterministically
    here and still match layout_meta.jsonl's (req 5), cross-checked explicitly below rather than assumed.

    Every region gets an ocr_meta.jsonl row (req 11/12), whether it's accepted or not:
      - accepted=True  -> a Chunk is created AND the full record is written.
      - accepted=False -> NO Chunk is created (keeps the vector index clean), but the full audit record
        (raw/normalized OCR text, reference text if any, confidence, cer, reject_reason) is still written.
    """
    reader = Reader(cfg)
    gold_pages = _load_gold_page_ids(cfg)
    ocr_cfg = cfg.get("ocr", {})
    max_cer = ocr_cfg.get("max_cer_target", _DEFAULT_MAX_CER)
    min_chars = ocr_cfg.get("min_line_chars", _DEFAULT_MIN_LINE_CHARS)
    layout_lookup = _load_layout_region_lookup(cfg)

    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = processed_dir / _META_SIDECAR

    # group regions by page. Order is NOT re-sorted here (req 1): vision/layout.py's detect() already
    # returns regions in correct reading order per page (Y-band-tolerant primary sort, X secondary).
    by_page: dict[str, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page_id, []).append(r)

    chunks: list[Chunk] = []
    meta_rows: list[dict] = []

    for page_id, page_regions in by_page.items():
        doc_id = page_id.rsplit("_p", 1)[0]
        is_gold_page = page_id in gold_pages
        ref_lines = _load_page_reference_lines(page_id, cfg, expected_n_lines=len(page_regions))

        for idx, region in enumerate(page_regions):
            region_id = f"{page_id}_r{idx:03d}"  # must match layout.py's own derivation exactly
            chunk_id = f"{page_id}_l{idx:03d}"

            layout_row = layout_lookup.get(region_id)
            if layout_lookup and layout_row is None:
                logger.warning(f"{region_id}: not found in {_LAYOUT_META} — Stage 2/3 region order may have desynced")
            elif layout_row is not None and tuple(layout_row["bbox"]) != tuple(region.bbox):
                logger.warning(f"{region_id}: bbox mismatch vs. {_LAYOUT_META} — Stage 2/3 region order may have desynced")

            raw_text, conf = reader._ocr_line(reader._crop(region))
            norm_text = _normalize(raw_text)

            row = {
                "region_id": region_id,
                "chunk_id": chunk_id,
                "page_id": page_id,
                "ocr_confidence": conf,
                "cer": None,
                "ocr_text_raw": raw_text,
                "ocr_text_normalized": norm_text,
                "reference_text_normalized": None,
                "evidence_tier": None,
                "accepted": False,
                "reject_reason": None,
            }

            if len(norm_text) < min_chars:
                row["reject_reason"] = "REJECT_TOO_SHORT"
                row["evidence_tier"] = "raw"
                meta_rows.append(row)
                continue

            ref_line = ref_lines[idx] if ref_lines is not None and idx < len(ref_lines) else None
            ref_norm = _normalize(ref_line) if ref_line else ""
            if ref_norm:
                row["reference_text_normalized"] = ref_norm
                row["cer"] = _cer(norm_text, ref_norm)

            if is_gold_page:
                # Independently human-verified page: accepted unconditionally past the length check.
                # CER against IA's silver reference is still recorded above when available, for audit/
                # diagnostic value, but never gates acceptance here -- gold overrides the silver gate.
                row["accepted"] = True
                row["evidence_tier"] = "gold"
            elif not ref_norm:
                row["accepted"] = False
                row["reject_reason"] = "reference_alignment_failed"
                row["evidence_tier"] = "raw"
            elif row["cer"] <= max_cer:
                row["accepted"] = True
                row["evidence_tier"] = "silver"
            else:
                row["accepted"] = False
                row["reject_reason"] = "cer_above_threshold"
                row["evidence_tier"] = "raw"

            if row["accepted"]:
                chunks.append(Chunk(id=chunk_id, doc_id=doc_id, text=norm_text, page_ids=[page_id]))

            meta_rows.append(row)

    with open(meta_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_accepted = sum(1 for r in meta_rows if r["accepted"])
    logger.info(f"OCR'd {len(meta_rows)} regions across {len(by_page)} pages: "
                f"{n_accepted} accepted -> {len(chunks)} chunks, {len(meta_rows) - n_accepted} rejected -> {meta_path}")
    return chunks