"""Stage 4 — chunk text"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

_OCR_META = "ocr_meta.jsonl"
_CHUNK_META = "chunk_meta.jsonl"


def _load_meta(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not path.exists():
        return meta
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta[row["chunk_id"]] = row
    return meta


def _normalize(text: str) -> str:
    """A1 Section 2: Unicode NFC normalisation + whitespace cleanup, preserving stanza/line breaks
    (we join with a single space at the line level already; NFC only touches character composition,
    it never merges or drops lines)."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk line-level OCR chunks (one per Chunk, id like '<page_id>_l003') into retrieval-sized
    windows of cfg['index']['chunk_tokens'] whitespace tokens with cfg['index']['overlap'] tokens of
    overlap, merging consecutive lines within the same literary work (doc_id) only — never across
    documents, which would violate the document-level split A1 Section 2 relies on to avoid leakage.

    Reads per-line OCR confidence + silver/gold tier from data/processed/ocr_meta.jsonl (contracts.Chunk
    has no field for either) and writes the merged-chunk equivalent to data/processed/chunk_meta.jsonl:
    tier is 'gold' only if every constituent line is gold (conservative — a merged chunk discloses the
    weaker tier of anything it contains, per A1's tier(a_i) disclosure requirement, Section 1)."""
    index_cfg = cfg.get("index", {})
    chunk_tokens = index_cfg.get("chunk_tokens", 256)
    overlap = index_cfg.get("overlap", 32)
    if overlap >= chunk_tokens:
        raise ValueError(f"index.overlap ({overlap}) must be < index.chunk_tokens ({chunk_tokens})")

    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    ocr_meta = _load_meta(processed_dir / _OCR_META)

    # group input line-chunks by doc_id, preserving the order they arrived in (loader/ocr already sort
    # by page_id, and within a page by top-to-bottom y, so this preserves reading order)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    out: list[Chunk] = []
    meta_rows: list[dict] = []

    for doc_id, lines in by_doc.items():
        # flatten to a token stream, remembering which source line + tier/confidence each token came from
        tokens: list[str] = []
        token_source: list[Chunk] = []
        for line in lines:
            words = _normalize(line.text).split()
            tokens.extend(words)
            token_source.extend([line] * len(words))

        step = chunk_tokens - overlap
        idx = 0
        pos = 0
        n = len(tokens)
        if n == 0:
            continue
        while pos < n:
            end = min(pos + chunk_tokens, n)
            window_tokens = tokens[pos:end]
            window_sources = token_source[pos:end]
            if not window_tokens:
                break

            text = " ".join(window_tokens)
            page_ids = sorted({pid for line in window_sources for pid in line.page_ids})
            chunk_id = f"{doc_id}_c{idx:05d}"
            out.append(Chunk(id=chunk_id, doc_id=doc_id, text=text, page_ids=page_ids))

            source_ids = {s.id for s in window_sources}
            confs = [ocr_meta[sid]["ocr_confidence"] for sid in source_ids if sid in ocr_meta]
            tiers = [ocr_meta[sid]["evidence_tier"] for sid in source_ids if sid in ocr_meta]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            tier = "gold" if tiers and all(t == "gold" for t in tiers) else "silver"
            meta_rows.append({"chunk_id": chunk_id, "ocr_confidence": mean_conf, "tier": tier,
                               "source_line_ids": sorted(source_ids)})

            idx += 1
            if end == n:
                break
            pos += step

    with open(processed_dir / _CHUNK_META, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"re-chunked {len(chunks)} lines into {len(out)} index chunks "
                f"(chunk_tokens={chunk_tokens}, overlap={overlap})")
    return out