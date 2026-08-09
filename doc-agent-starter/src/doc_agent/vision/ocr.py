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
#   "gold"   — our OCR of this exact line matches (CER==0.0) grading_kit/labels.jsonl's human-verified
#              page text, positionally aligned. A page being IN labels.jsonl is NOT sufficient by
#              itself — that only proves a human verified the page's text, not that our OCR engine's
#              output for this specific line matches it.
#   "silver" — didn't earn gold, but passed the CER quality gate against IA's own silver reference
#   "raw"    — failed every check above, or unverifiable — NOT indexed as a Chunk
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


def _load_gold_page_texts(cfg: dict) -> dict[str, str]:
    """page_id -> its human-verified page text from grading_kit/labels.jsonl (raw, not yet split into
    lines or normalized — that happens per-page in _align_reference_lines, same as the IA reference)."""
    labels_path = Path(cfg.get("grading_kit", {}).get("labels_path", "grading_kit/labels.jsonl"))
    texts: dict[str, str] = {}
    if not labels_path.exists():
        return texts
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
            if pid and text and not text.startswith("REPLACE ME"):  # skip the un-filled stub row
                texts[pid] = text
    return texts


def _load_layout_region_lookup(cfg: dict) -> dict[str, dict]:
    """region_id -> its layout_meta.jsonl row, for the Stage 2/3 provenance check. Returns {} if
    layout_meta.jsonl hasn't been written yet -- the check is then skipped entirely rather than
    treated as a failure (Stage 3 can still run standalone, e.g. in unit tests that hand-build a
    Region without running layout.detect() first)."""
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


def _align_reference_lines(raw_text: str, expected_n_lines: int, source: str, page_id: str) -> list[str] | None:
    """Split a reference text blob into lines and check it against the detected region count for this
    page. Alignment is refused (returns None) unless the counts match EXACTLY — positional line-to-line
    correspondence is only trustworthy when both sides agree on how many physical lines exist; even a
    1-line difference means everything after the divergence point is compared against the wrong line.
    (This is still a real limitation even at an exact count match: IA could split/merge lines
    differently in a way that happens to net out to the same total. A true per-line sequence alignment
    — e.g. the Levenshtein-projection approach A1's notebook describes for gold-set construction — is a
    heavier batch job better suited to an offline script, not this per-page hot path.)
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines or expected_n_lines == 0:
        return None
    if len(lines) != expected_n_lines:
        logger.warning(
            f"page {page_id}: {source} has {len(lines)} lines vs. {expected_n_lines} detected regions "
            f"— refusing positional alignment (exact match required)"
        )
        return None
    return lines


def _load_page_reference_lines(page_id: str, cfg: dict, expected_n_lines: int) -> list[str] | None:
    """Best-effort IA silver-reference lines for a page, for the CER gate.

    Convention: data/raw/<doc_id>/<page_id>.ref.txt, one line per physical line, same top-to-bottom
    order as our own detected regions. NOTE: scripts/get_data.sh does not fetch this today — nothing in
    this repo populates it yet, so until that's wired up, every non-gold line will correctly get
    reject_reason='reference_alignment_failed' rather than a false accept. That's the intended safe
    default, not a bug.
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
    raw_text = open(ref_path, encoding="utf-8").read()
    return _align_reference_lines(raw_text, expected_n_lines, "IA reference", page_id)


def _normalize(text: str) -> str:
    """Conservative normalization: Unicode NFC + whitespace collapse only. Deliberately does NOT strip
    Bengali punctuation, digits, or any characters — over-normalizing would corrupt the very text a
    citation is supposed to quote verbatim."""
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
    """CER = (S+D+I) / N, N = len(reference). Caller must ensure `ref` is non-empty -- an empty
    reference is an alignment failure, not a CER of 0 or infinity."""
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
        """4px pad, strictly clamped to [0,width] x [0,height] — protects Bengali matras and conjunct
        ascenders/descenders sitting right at the detected box edge."""
        img_path = _image_path_for_page(region.page_id, self.full_cfg)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read processed image for page {region.page_id} at {img_path}")
        x, y, w, h = region.bbox
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        return img[y0:y1, x0:x1]

    def ocr_config_string(self) -> str:
        """The exact Tesseract config used for per-line transcription, as one literal string
        (--oem 1 --psm 7 -l ben) — stamped into every ocr_meta.jsonl row for reproducibility."""
        lang = self.cfg.get("lang", "ben")
        oem = self.cfg.get("oem", 1)
        return f"--oem {oem} --psm 7 -l {lang}"

    def _ocr_line(self, crop) -> tuple[str, float]:
        """OCR one line-crop. Returns (raw_text, mean_word_confidence in [0,1])."""
        if crop.size == 0:
            return "", 0.0
        data = pytesseract.image_to_data(crop, config=self.ocr_config_string(), output_type=pytesseract.Output.DICT)
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
    """Regions -> line-level Chunks, gated by a silver-label CER quality check.

    Granularity/order: regions are consumed exactly as vision/layout.py returned them, grouped by page
    but never re-sorted — that ordering is what lets region_id be regenerated deterministically here and
    still match layout_meta.jsonl's, cross-checked explicitly below rather than assumed.

    Every region gets an ocr_meta.jsonl row, whether it's accepted or not:
      - accepted=True  -> a Chunk is created AND the full record is written.
      - accepted=False -> NO Chunk is created (keeps the vector index clean), but the full audit record
        is still written (raw/normalized OCR text, reference text if any, confidence, cer,
        cer_threshold, ocr_config, reject_reason).

    Per-region decision order (first match wins):
      1. Layout provenance check — if layout_meta.jsonl exists but this region_id is missing from it,
         or its bbox doesn't match, reject as layout_provenance_mismatch. If we can't trust which
         physical region this even is, no downstream judgement (even a perfect CER match) should
         override that doubt.
      2. Too-short check — REJECT_TOO_SHORT, universal, before any tier branch.
      3. Gold check — if this page has a human-verified text in grading_kit/labels.jsonl, and this
         specific line's OCR output exactly matches (CER==0.0) the correspondingly-aligned gold line,
         accept as "gold". A page being in labels.jsonl is NOT enough by itself; that only proves a
         human verified the page's text, not that OUR OCR of this exact line matches it.
      4. Standard silver/raw gate — CER against the IA reference, accept as "silver" if <= max_cer_target,
         else reject as cer_above_threshold. This is also where a gold-page line falls if it didn't earn
         gold (e.g. Tesseract misread that one line) — it is NOT auto-rejected just for missing gold.
    """
    reader = Reader(cfg)
    gold_texts = _load_gold_page_texts(cfg)
    ocr_cfg = cfg.get("ocr", {})
    max_cer = ocr_cfg.get("max_cer_target", _DEFAULT_MAX_CER)
    min_chars = ocr_cfg.get("min_line_chars", _DEFAULT_MIN_LINE_CHARS)
    layout_lookup = _load_layout_region_lookup(cfg)
    ocr_config_str = reader.ocr_config_string()

    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = processed_dir / _META_SIDECAR

    by_page: dict[str, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page_id, []).append(r)

    chunks: list[Chunk] = []
    meta_rows: list[dict] = []

    for page_id, page_regions in by_page.items():
        doc_id = page_id.rsplit("_p", 1)[0]
        n = len(page_regions)
        ref_lines = _load_page_reference_lines(page_id, cfg, expected_n_lines=n)
        gold_text = gold_texts.get(page_id)
        gold_lines = _align_reference_lines(gold_text, n, "gold label", page_id) if gold_text else None

        for idx, region in enumerate(page_regions):
            region_id = f"{page_id}_r{idx:03d}"
            chunk_id = f"{page_id}_l{idx:03d}"

            raw_text, conf = reader._ocr_line(reader._crop(region))
            norm_text = _normalize(raw_text)

            row = {
                "region_id": region_id,
                "chunk_id": chunk_id,
                "page_id": page_id,
                "ocr_confidence": conf,
                "cer": None,
                "cer_threshold": max_cer,
                "ocr_config": ocr_config_str,
                "ocr_text_raw": raw_text,
                "ocr_text_normalized": norm_text,
                "reference_text_normalized": None,
                "evidence_tier": None,
                "accepted": False,
                "reject_reason": None,
            }

            # 1. Layout provenance check — takes precedence over every other decision below.
            layout_row = layout_lookup.get(region_id)
            provenance_ok = True
            if layout_lookup and layout_row is None:
                logger.warning(f"{region_id}: not found in {_LAYOUT_META} — Stage 2/3 region order may have desynced")
                provenance_ok = False
            elif layout_row is not None and tuple(layout_row["bbox"]) != tuple(region.bbox):
                logger.warning(f"{region_id}: bbox mismatch vs. {_LAYOUT_META} — Stage 2/3 region order may have desynced")
                provenance_ok = False

            if not provenance_ok:
                row["reject_reason"] = "layout_provenance_mismatch"
                row["evidence_tier"] = "raw"
                meta_rows.append(row)
                continue

            # 2. Too-short check — universal, before any tier branch.
            if len(norm_text) < min_chars:
                row["reject_reason"] = "REJECT_TOO_SHORT"
                row["evidence_tier"] = "raw"
                meta_rows.append(row)
                continue

            # 3. Gold check — requires an exact (CER==0.0) match against the aligned gold line, not
            # merely being on a page that has SOME gold label.
            gold_line = gold_lines[idx] if gold_lines is not None and idx < len(gold_lines) else None
            gold_norm = _normalize(gold_line) if gold_line else ""
            if gold_norm:
                gold_cer = _cer(norm_text, gold_norm)
                if gold_cer == 0.0:
                    row["reference_text_normalized"] = gold_norm
                    row["cer"] = 0.0
                    row["accepted"] = True
                    row["evidence_tier"] = "gold"
                    chunks.append(Chunk(id=chunk_id, doc_id=doc_id, text=norm_text, page_ids=[page_id]))
                    meta_rows.append(row)
                    continue
                # else: falls through to the standard silver/raw gate below, same as any other line —
                # missing gold (even on a gold-reviewed page) is not an automatic rejection.

            # 4. Standard silver/raw gate against the IA reference.
            ref_line = ref_lines[idx] if ref_lines is not None and idx < len(ref_lines) else None
            ref_norm = _normalize(ref_line) if ref_line else ""
            if not ref_norm:
                row["reject_reason"] = "reference_alignment_failed"
                row["evidence_tier"] = "raw"
                meta_rows.append(row)
                continue

            row["reference_text_normalized"] = ref_norm
            row["cer"] = _cer(norm_text, ref_norm)
            if row["cer"] <= max_cer:
                row["accepted"] = True
                row["evidence_tier"] = "silver"
                chunks.append(Chunk(id=chunk_id, doc_id=doc_id, text=norm_text, page_ids=[page_id]))
            else:
                row["reject_reason"] = "cer_above_threshold"
                row["evidence_tier"] = "raw"

            meta_rows.append(row)

    with open(meta_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_accepted = sum(1 for r in meta_rows if r["accepted"])
    logger.info(f"OCR'd {len(meta_rows)} regions across {len(by_page)} pages: "
                f"{n_accepted} accepted -> {len(chunks)} chunks, {len(meta_rows) - n_accepted} rejected -> {meta_path}")
    return chunks