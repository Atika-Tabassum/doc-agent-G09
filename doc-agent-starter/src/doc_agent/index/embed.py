"""Stage 4 — embed chunks"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

_MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}


def _resolve_device(cfg: dict) -> str:
    """Honor cfg['device'], but fall back to CPU if CUDA was requested but isn't actually
    available -- a GPU-less dev machine or CI runner must not hard-fail here."""
    requested = cfg.get("device", "cpu")
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "cfg['device']='cuda' but no CUDA device is available -- falling back to cpu"
            )
            return "cpu"
    return requested


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    """Lazily import + cache sentence-transformers models, keyed by (model_name, device), so
    repeated encode() calls (e.g. `make ingest index` running the pipeline more than once) don't
    reload the model from disk/HF hub every time."""
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        logger.info(f"loading embedding model {model_name} on {device}")
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed with cfg['embed']['model']. Returns L2-normalized float32 vectors, shape
    (len(chunks), dim), in the same row order as `chunks` -- store.py's FAISS index uses
    METRIC_INNER_PRODUCT, so normalizing here is what makes that behave as cosine similarity."""
    embed_cfg = cfg.get("embed", {})
    model_name = embed_cfg.get("model", "intfloat/multilingual-e5-base")
    batch_size = embed_cfg.get("batch_size", 32)
    device = _resolve_device(cfg)

    model = _load_model(model_name, device)

    texts = [c.text for c in chunks]
    if "e5" in model_name.lower():
        # e5 models expect a "passage: " prefix on indexed text (and "query: " at query time --
        # that half is A3's retriever.py's concern, not this stage's).
        texts = [f"passage: {t}" for t in texts]

    vectors = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
    )
    vectors = vectors.astype("float32")

    logger.info(f"embedded {len(chunks)} chunks with {model_name} (dim={vectors.shape[1]})")
    return vectors
