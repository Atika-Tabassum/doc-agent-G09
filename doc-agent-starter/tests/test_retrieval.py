"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

import numpy as np
import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk, embed


class _FakeEmbedModel:
    """Deterministic stand-in for a SentenceTransformer, injected via monkeypatching
    embed._load_model. Avoids a real network/HF-hub download in every test run while still
    exercising embed.encode()'s own logic (e5 prefixing, batching, dtype/shape handling) --
    only the model's inference is faked, not encode() itself. Records every call's args so
    tests can assert on what text actually reached the "model" (e.g. e5 prefixing)."""

    def __init__(self, dim: int = 8, seed: int = 42):
        self.dim = dim
        self._rng = np.random.default_rng(seed)  # fixed seed -> reproducible vectors across runs
        self.calls: list[dict] = []

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True):
        # convert_to_numpy is accepted-but-unused: kept only so this fake's signature matches the
        # real SentenceTransformer.encode() that embed.py calls with it as a keyword argument --
        # this fake always returns a numpy array regardless of the flag's value.
        self.calls.append({"texts": list(texts), "batch_size": batch_size})
        vectors = self._rng.random((len(texts), self.dim)).astype("float32")
        if normalize_embeddings:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def test_split_reads_evidence_tier_key_without_crashing(tmp_path):
    """Regression test: vision/ocr.py writes each line's tier to ocr_meta.jsonl under the key
    "evidence_tier" (never "tier") — chunk.split() must read that same key, or it crashes with
    KeyError on any real OCR output that has at least one accepted line."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    ocr_meta_rows = [
        {"chunk_id": "testwork_p0001_l000", "ocr_confidence": 0.95, "evidence_tier": "gold"},
        {"chunk_id": "testwork_p0001_l001", "ocr_confidence": 0.90, "evidence_tier": "gold"},
        {"chunk_id": "testwork_p0001_l002", "ocr_confidence": 0.80, "evidence_tier": "silver"},
    ]
    with open(processed_dir / "ocr_meta.jsonl", "w", encoding="utf-8") as f:
        for row in ocr_meta_rows:
            f.write(json.dumps(row) + "\n")

    page_ids = ["testwork_p0001"]
    lines = [
        Chunk(id="testwork_p0001_l000", doc_id="testwork", text="প্রথম লাইন", page_ids=page_ids),
        Chunk(id="testwork_p0001_l001", doc_id="testwork", text="দ্বিতীয় লাইন", page_ids=page_ids),
        Chunk(id="testwork_p0001_l002", doc_id="testwork", text="তৃতীয় লাইন", page_ids=page_ids),
    ]

    cfg = {
        "index": {"chunk_tokens": 50, "overlap": 5},
        "paths": {"processed_dir": str(processed_dir)},
    }

    out = chunk.split(lines, cfg)  # must not raise KeyError

    assert len(out) == 1  # all 3 short lines fit in one 50-token window
    rows = [json.loads(line) for line in open(processed_dir / "chunk_meta.jsonl", encoding="utf-8")]
    assert len(rows) == 1
    # two gold source lines + one silver -> conservative aggregation: "silver", not "gold"
    assert rows[0]["tier"] == "silver"


def test_split_merges_lines_within_a_document_with_overlap(tmp_path):
    """chunk_tokens=6, overlap=2 -> step=4. Three 4-token lines (12 tokens total) on three
    different pages must produce 3 windows of sizes [6, 6, 4], where each window's last 2 tokens
    equal the next window's first 2 tokens, and each window's page_ids is the sorted union of
    every source line it touches."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    lines = [
        Chunk(id="workA_p0001_l000", doc_id="workA", text="w1 w2 w3 w4", page_ids=["workA_p0001"]),
        Chunk(id="workA_p0002_l000", doc_id="workA", text="w5 w6 w7 w8", page_ids=["workA_p0002"]),
        Chunk(
            id="workA_p0003_l000", doc_id="workA", text="w9 w10 w11 w12", page_ids=["workA_p0003"]
        ),
    ]
    cfg = {
        "index": {"chunk_tokens": 6, "overlap": 2},
        "paths": {"processed_dir": str(processed_dir)},
    }

    out = chunk.split(lines, cfg)

    assert len(out) == 3
    token_counts = [len(c.text.split()) for c in out]
    assert token_counts == [6, 6, 4]

    # overlap: each window's last 2 tokens == the next window's first 2 tokens
    for a, b in zip(out, out[1:], strict=False):
        assert a.text.split()[-2:] == b.text.split()[:2]

    assert out[0].page_ids == ["workA_p0001", "workA_p0002"]
    assert out[1].page_ids == ["workA_p0002", "workA_p0003"]
    assert out[2].page_ids == ["workA_p0003"]


def test_split_never_merges_across_documents(tmp_path):
    """chunk.split() must never merge lines across different doc_id -- doing so would
    reintroduce the leakage the A1 document-level train/val/test split exists to prevent."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    lines = [
        Chunk(
            id="workA_p0001_l000", doc_id="workA", text="a1 a2 a3 a4 a5", page_ids=["workA_p0001"]
        ),
        Chunk(
            id="workB_p0001_l000", doc_id="workB", text="b1 b2 b3 b4 b5", page_ids=["workB_p0001"]
        ),
    ]
    cfg = {
        "index": {"chunk_tokens": 6, "overlap": 2},
        "paths": {"processed_dir": str(processed_dir)},
    }

    out = chunk.split(lines, cfg)

    assert len(out) == 2
    by_doc = {c.doc_id: c for c in out}
    assert set(by_doc.keys()) == {"workA", "workB"}
    a_tokens = set(by_doc["workA"].text.split())
    b_tokens = set(by_doc["workB"].text.split())
    assert a_tokens == {"a1", "a2", "a3", "a4", "a5"}
    assert b_tokens == {"b1", "b2", "b3", "b4", "b5"}
    assert not a_tokens & b_tokens  # neither chunk's text contains a token from the other doc


def test_split_rejects_overlap_gte_chunk_tokens(tmp_path):
    """overlap must be strictly less than chunk_tokens -- equal or greater makes the sliding
    window either stall (step <= 0) or go backwards."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    lines = [Chunk(id="w_p0001_l000", doc_id="w", text="x y z", page_ids=["w_p0001"])]
    base_cfg = {"paths": {"processed_dir": str(processed_dir)}}

    with pytest.raises(ValueError):
        chunk.split(lines, {**base_cfg, "index": {"chunk_tokens": 10, "overlap": 10}})

    with pytest.raises(ValueError):
        chunk.split(lines, {**base_cfg, "index": {"chunk_tokens": 10, "overlap": 15}})


def test_encode_output_shape_dtype_and_unit_norm(monkeypatch):
    """embed.encode()'s core contract: one row per chunk, correct dim, float32, L2-normalized --
    store.py's FAISS index uses METRIC_INNER_PRODUCT, so an un-normalized row would silently break
    cosine-similarity search without ever raising an error. Monkeypatching _load_model means this
    never touches the network or downloads the real ~1.1GB multilingual-e5-base model."""
    fake_model = _FakeEmbedModel(dim=8)
    monkeypatch.setattr(embed, "_load_model", lambda model_name, device: fake_model)

    chunks = [
        Chunk(id="doc1_c00000", doc_id="doc1", text="প্রথম বাক্য", page_ids=["doc1_p001"]),
        Chunk(id="doc1_c00001", doc_id="doc1", text="দ্বিতীয় বাক্য", page_ids=["doc1_p001"]),
        Chunk(id="doc1_c00002", doc_id="doc1", text="তৃতীয় বাক্য", page_ids=["doc1_p002"]),
    ]
    cfg = {"device": "cpu", "embed": {"model": "intfloat/multilingual-e5-base", "batch_size": 2}}

    vectors = embed.encode(chunks, cfg)

    assert vectors.shape == (len(chunks), 8)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_prefixes_e5_model_input_with_passage(monkeypatch):
    """e5-family models (per the model card intfloat/multilingual-e5-base ships under) expect
    indexed text prefixed with "passage: " -- without it, retrieval quality silently degrades
    rather than erroring, so this needs its own explicit test rather than relying on the shape/
    dtype test above to happen to catch it. Asserting on _FakeEmbedModel.calls (what text the
    "model" actually received) checks the prefix is applied, not just that some output came back."""
    fake_model = _FakeEmbedModel(dim=8)
    monkeypatch.setattr(embed, "_load_model", lambda model_name, device: fake_model)

    chunks = [Chunk(id="doc1_c00000", doc_id="doc1", text="মূল লাইন", page_ids=["doc1_p001"])]
    cfg = {"device": "cpu", "embed": {"model": "intfloat/multilingual-e5-base", "batch_size": 1}}

    embed.encode(chunks, cfg)

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["texts"] == ["passage: মূল লাইন"]


@pytest.mark.skip(reason="downloads real model; run manually")
def test_encode_real_model_smoke():
    """One-time manual check that the REAL intfloat/multilingual-e5-base model (not the fake
    above) actually works end-to-end -- the mocked tests above only prove embed.encode()'s own
    logic is correct, not that the real model/weights load and produce sane output. Skip-marked
    so `make test`/CI never needs network access or a ~1.1GB download; vanilla pytest has no CLI
    flag to override an individual @pytest.mark.skip, so to run this manually, comment out the
    skip marker above, run `pytest tests/test_retrieval.py -k real_model_smoke -v`, then restore
    the marker before committing."""
    chunks = [
        Chunk(id="doc1_c00000", doc_id="doc1", text="বাস্তব মডেল পরীক্ষা", page_ids=["doc1_p001"])
    ]
    cfg = {"device": "cpu", "embed": {"model": "intfloat/multilingual-e5-base", "batch_size": 1}}

    vectors = embed.encode(chunks, cfg)

    assert vectors.shape == (1, 768)
    assert vectors.dtype == np.float32
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-3
