"""Stage 4 — vector store"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.jsonl"


def _load_chunk_meta(cfg: dict) -> dict[str, dict]:
    processed_dir = Path(cfg.get("paths", {}).get("processed_dir", "data/processed"))
    meta_path = processed_dir / "chunk_meta.jsonl"
    meta: dict[str, dict] = {}
    if not meta_path.exists():
        return meta
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta[row["chunk_id"]] = row
    return meta


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']) + a chunk-metadata sidecar so results can be
    reconstructed as full Chunk objects (with score set) plus the OCR confidence / silver-gold tier
    from index/chunk.py's chunk_meta.jsonl (contracts.Chunk has no field for either — see design_choices.md).

    Row i of the persisted metadata corresponds to FAISS internal id i (== row i of `vectors`), which is
    how load()/the future retriever map a vector-search hit back to its Chunk + confidence + tier.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu is required for index/store.py (see pyproject.toml). "
            "Install it in your project environment: `uv sync` (or `pip install faiss-cpu`)."
        ) from e

    if len(chunks) != vectors.shape[0]:
        raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors — must be 1:1 and same order")

    index_cfg = cfg.get("index", {})
    index_type = index_cfg.get("type", "faiss:hnsw")
    dim = vectors.shape[1]

    if index_type == "faiss:hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    elif index_type == "faiss:flat":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"unsupported index.type '{index_type}' — expected 'faiss:hnsw' or 'faiss:flat'")

    index.add(vectors.astype("float32"))

    index_dir = Path(cfg.get("paths", {}).get("index_dir", "data/processed/index"))
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / _INDEX_FILE))

    chunk_meta = _load_chunk_meta(cfg)
    with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = chunk_meta.get(c.id, {})
            row = {
                "id": c.id, "doc_id": c.doc_id, "text": c.text, "page_ids": c.page_ids,
                "ocr_confidence": meta.get("ocr_confidence", 0.0),
                "tier": meta.get("tier", "silver"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"built {index_type} index with {index.ntotal} vectors (dim={dim}) -> {index_dir}")


def load(cfg: dict):
    """Load the persisted index + chunk metadata.
    Returns (faiss_index, chunk_rows) where chunk_rows[i] is the metadata dict for FAISS internal id i
    (id, doc_id, text, page_ids, ocr_confidence, tier) — consumed by retrieval/retriever.py (A3)."""
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu is required for index/store.py (see pyproject.toml). "
            "Install it in your project environment: `uv sync` (or `pip install faiss-cpu`)."
        ) from e

    index_dir = Path(cfg.get("paths", {}).get("index_dir", "data/processed/index"))
    index_path, chunks_path = index_dir / _INDEX_FILE, index_dir / _CHUNKS_FILE
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"no index found at {index_dir} — run scripts/build_index.sh first")

    index = faiss.read_index(str(index_path))
    chunk_rows: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk_rows.append(json.loads(line))

    logger.info(f"loaded index with {index.ntotal} vectors and {len(chunk_rows)} chunk records from {index_dir}")
    return index, chunk_rows