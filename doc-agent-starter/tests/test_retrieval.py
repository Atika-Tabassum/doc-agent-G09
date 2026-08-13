"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

import numpy as np
import pytest
from PIL import Image

from doc_agent import pipeline
from doc_agent.contracts import Chunk, Region
from doc_agent.index import chunk, embed, store


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


def test_store_build_and_load_round_trip(tmp_path):
    """store.build()/load() is the persistence boundary between index/{embed,chunk}.py and the
    (still-A3) retriever -- a mismatch here (wrong row order, dropped field) would silently
    corrupt every future retrieval result without any of the embed/chunk-level tests catching it.
    Using index.type "faiss:flat" (exact IndexFlatIP) rather than the production "faiss:hnsw"
    default keeps this deterministic; HNSW's approximate search isn't what's under test here."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    index_dir = processed_dir / "index"
    cfg = {
        "paths": {"processed_dir": str(processed_dir), "index_dir": str(index_dir)},
        "index": {"type": "faiss:flat"},
    }

    chunks = [
        Chunk(id="doc1_c00000", doc_id="doc1", text="prothom", page_ids=["doc1_p001"]),
        Chunk(id="doc1_c00001", doc_id="doc1", text="dwitiyo", page_ids=["doc1_p001", "doc1_p002"]),
    ]
    # Vectors don't need to come from a real model for this test -- store.build()/load() treat
    # them as opaque float32 rows; only their count/order relative to `chunks` matters here.
    rng = np.random.default_rng(0)
    vectors = rng.random((2, 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    store.build(chunks, vectors, cfg)
    faiss_index, chunk_rows = store.load(cfg)

    assert faiss_index.ntotal == 2
    assert len(chunk_rows) == 2
    for original_chunk, row in zip(chunks, chunk_rows, strict=True):
        assert row["id"] == original_chunk.id
        assert row["doc_id"] == original_chunk.doc_id
        assert row["text"] == original_chunk.text
        assert row["page_ids"] == original_chunk.page_ids


def test_store_load_raises_file_not_found_with_helpful_message(tmp_path):
    """store.load() is the first thing a fresh checkout (or a CI runner) hits if someone forgets
    to build the index first -- the error needs to actually say what to run next, not just that
    something's missing, so this test locks the message wording as part of the contract, not just
    the exception type."""
    missing_index_dir = tmp_path / "data" / "processed" / "index"
    cfg = {"paths": {"index_dir": str(missing_index_dir)}}

    with pytest.raises(FileNotFoundError) as exc_info:
        store.load(cfg)

    assert str(missing_index_dir) in str(exc_info.value)
    assert "run scripts/build_index.sh first" in str(exc_info.value)


def test_store_nearest_neighbor_returns_exact_match(tmp_path):
    """Proves the FAISS index actually implements similarity search correctly (not just that
    build()/load() round-trip files) -- searching with a chunk's own embedding should return
    that exact chunk as the top hit with score ~1.0 (cosine similarity of a unit vector with
    itself). "faiss:flat" (exact IndexFlatIP) is used deliberately instead of the production
    "faiss:hnsw" default, since HNSW is an *approximate* nearest-neighbor index -- it isn't
    guaranteed to return the true top-1 on every query, which would make this test flaky."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    index_dir = processed_dir / "index"
    cfg = {
        "paths": {"processed_dir": str(processed_dir), "index_dir": str(index_dir)},
        "index": {"type": "faiss:flat"},
    }

    chunks = [
        Chunk(id=f"doc1_c{i:05d}", doc_id="doc1", text=f"line {i}", page_ids=["doc1_p001"])
        for i in range(5)
    ]
    rng = np.random.default_rng(1)
    vectors = rng.random((5, 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    store.build(chunks, vectors, cfg)
    faiss_index, chunk_rows = store.load(cfg)

    query = vectors[2:3]  # an exact copy of chunk index 2's own embedding
    scores, ids = faiss_index.search(query, 1)

    assert chunk_rows[ids[0][0]]["id"] == "doc1_c00002"
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


def test_store_build_propagates_chunk_meta_tier_and_confidence(tmp_path):
    """contracts.Chunk (fixed, can't add fields) has no slot for OCR confidence or silver/gold
    tier -- store.build() has to join them in from chunk_meta.jsonl (written by chunk.split(),
    see chunk.py) by chunk id, and this is the only place that join happens. A silent keying bug
    here (e.g. matching by list position instead of id) would make every downstream evidence-tier
    disclosure wrong without any of the round-trip/NN tests above noticing, since they don't
    populate chunk_meta.jsonl at all."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    index_dir = processed_dir / "index"

    meta_rows = [
        {
            "chunk_id": "doc1_c00000",
            "ocr_confidence": 0.91,
            "tier": "gold",
            "source_line_ids": ["doc1_p001_l01"],
        },
        {
            "chunk_id": "doc1_c00001",
            "ocr_confidence": 0.42,
            "tier": "silver",
            "source_line_ids": ["doc1_p001_l02"],
        },
    ]
    with open(processed_dir / "chunk_meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = {
        "paths": {"processed_dir": str(processed_dir), "index_dir": str(index_dir)},
        "index": {"type": "faiss:flat"},
    }
    chunks = [
        Chunk(id="doc1_c00000", doc_id="doc1", text="gold line", page_ids=["doc1_p001"]),
        Chunk(id="doc1_c00001", doc_id="doc1", text="silver line", page_ids=["doc1_p001"]),
    ]
    # Orthogonal unit rows -- values don't matter, only that build()/load() treat them opaquely.
    vectors = np.eye(2, 8, dtype="float32")

    store.build(chunks, vectors, cfg)
    _, chunk_rows = store.load(cfg)
    by_id = {row["id"]: row for row in chunk_rows}

    assert by_id["doc1_c00000"]["tier"] == "gold"
    assert by_id["doc1_c00000"]["ocr_confidence"] == pytest.approx(0.91)
    assert by_id["doc1_c00001"]["tier"] == "silver"
    assert by_id["doc1_c00001"]["ocr_confidence"] == pytest.approx(0.42)


def test_full_index_pipeline_chunk_embed_store_round_trip(tmp_path, monkeypatch):
    """Safety net for the whole Stage-4 chain: chunk.split() -> embed.encode() -> store.build()
    -> store.load(), on hand-built line-level Chunks (same shape as the existing chunk.split
    tests above -- no real OCR/Tesseract needed). Each stage above this point is unit-tested in
    isolation (Milestones 2-5's chunk tests, this file's embed/store tests), but isolation tests
    can't catch a contract mismatch AT the seam between two stages (e.g. chunk.split producing an
    id chunk_meta.jsonl doesn't have a matching row for). This test chains the real functions
    together and checks the seams: chunk count == vector count == indexed count, the e5 prefix
    still fires inside the chain, and a chunk's own re-embedded text retrieves itself as top-1.

    Deliberately does NOT go through pipeline.build_knowledge_base() -- that's what Milestone 18's
    tripwire tests are for (DP-3); this test's job is proving MY stage's internal consistency,
    not exercising hooks.py/wiring.py."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    index_dir = processed_dir / "index"

    lines = [
        Chunk(id="doc1_p001_l01", doc_id="doc1", text="প্রথম লাইন এখানে", page_ids=["doc1_p001"]),
        Chunk(
            id="doc1_p001_l02", doc_id="doc1", text="দ্বিতীয় লাইন এখানে", page_ids=["doc1_p001"]
        ),
        Chunk(
            id="doc1_p002_l01",
            doc_id="doc1",
            text="তৃতীয় লাইন এখানে পাতা দুই",
            page_ids=["doc1_p002"],
        ),
    ]
    ocr_meta_rows = [
        {"chunk_id": "doc1_p001_l01", "ocr_confidence": 0.95, "evidence_tier": "gold"},
        {"chunk_id": "doc1_p001_l02", "ocr_confidence": 0.93, "evidence_tier": "gold"},
        {"chunk_id": "doc1_p002_l01", "ocr_confidence": 0.60, "evidence_tier": "silver"},
    ]
    with open(processed_dir / "ocr_meta.jsonl", "w", encoding="utf-8") as f:
        for row in ocr_meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = {
        "paths": {"processed_dir": str(processed_dir), "index_dir": str(index_dir)},
        "index": {"chunk_tokens": 6, "overlap": 2, "type": "faiss:flat"},
        "embed": {"model": "intfloat/multilingual-e5-base", "batch_size": 2},
        "device": "cpu",
    }

    index_chunks = chunk.split(lines, cfg)
    assert len(index_chunks) > 0

    fake_model = _FakeEmbedModel(dim=8)
    monkeypatch.setattr(embed, "_load_model", lambda model_name, device: fake_model)
    vectors = embed.encode(index_chunks, cfg)
    assert vectors.shape[0] == len(index_chunks)

    store.build(index_chunks, vectors, cfg)
    faiss_index, chunk_rows = store.load(cfg)

    assert faiss_index.ntotal == len(index_chunks)
    assert len(chunk_rows) == len(index_chunks)

    # e5 prefixing survives the full chain, not just the isolated embed tests above.
    assert fake_model.calls[0]["texts"][0] == f"passage: {index_chunks[0].text}"

    # A chunk's own vector, re-queried through the built index, should retrieve itself as top-1.
    query_vector = vectors[0:1]
    _, ids = faiss_index.search(query_vector, 1)
    assert chunk_rows[ids[0][0]]["id"] == index_chunks[0].id


def test_build_knowledge_base_blocked_by_enhance_stub_with_current_config(tmp_path):
    """pipeline.build_knowledge_base() is the REAL entry point scripts/build_index.sh invokes --
    every test above calls chunk.split()/embed.encode()/store.build() directly and never exercises
    wiring.register_all()/hooks.run() at all, so none of them would ever catch this.

    configs/config.yaml now has enhance.enabled: false (Enhancer's VAE/diffusion stage is bonus-
    only -- gated to data speciality E1 "severely degraded scans", which this group did not
    choose; we chose E26 "dirty-ocr", built in index/chunk.py -- so it was turned off rather than
    implemented). This test's own cfg still sets enhance.enabled=True regardless of the live
    config value, to pin Enhancer.apply()'s stub behaviour (`raise NotImplementedError("Stage 1:
    apply enhancer")`) as a permanent regression check: if anyone re-enables this bonus stage
    without implementing it, the failure should stay cheap and obvious to hit locally, not only
    surface during an expensive real corpus run. Not index/chunk/embed/store's file to fix if it
    ever needs a real implementation -- see test_build_knowledge_base_blocked_by_pii_stub_once_
    enhance_disabled below for the second (now-fixed) stub blocker on this same real entry point."""
    raw_dir = tmp_path / "data" / "raw" / "testwork"
    raw_dir.mkdir(parents=True)
    # A blank white image is fine here: Enhancer.apply() raises unconditionally, before it would
    # ever look at pixel content -- and even if preprocess.run's blank-page filter (ink ratio
    # < 0.002) drops this page first, enhance.run() still calls apply() on an empty list and still
    # raises the same way, so this test doesn't depend on the filter's exact threshold either.
    Image.new("L", (50, 50), color=255).save(raw_dir / "1.png")

    cfg = {
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "index_dir": str(tmp_path / "data" / "processed" / "index"),
        },
        # enhance.enabled: True regardless of the live configs/config.yaml (now False) -- pins
        # Enhancer.apply()'s stub behaviour as a permanent regression check (see docstring).
        "enhance": {"enabled": True, "model": "unet_small", "type": "diffusion"},
        # wiring.register_all() eagerly constructs agent/guardrails.py::Guardrails(cfg), which
        # reads cfg["agent"] in __init__ even though guardrails' own logic (ON_TOOL_CALL) never
        # fires during build_knowledge_base() -- a real run's full config.yaml always has this
        # section, so an empty dict here is enough to satisfy that constructor without pretending
        # to test anything guardrails-related.
        "agent": {},
    }

    with pytest.raises(NotImplementedError, match="enhance"):
        pipeline.build_knowledge_base(cfg)


def test_build_knowledge_base_blocked_by_pii_stub_once_enhance_disabled(tmp_path, monkeypatch):
    """Second stub blocker on the real build_knowledge_base() path, reached once Stage 1's enhance
    stub (test above) is bypassed via enhance.py's OWN documented enabled=False switch -- that's
    using the disable path enhance.run() already implements, not a workaround or a stub-into-a-
    no-op hack. layout.detect/ocr.transcribe are monkeypatched to bypass their pytesseract
    dependency: this machine has no Tesseract install (DP-1), and that's irrelevant to what's
    under test here anyway -- test_ocr.py already covers real OCR behaviour on machines that do
    have it. The point of THIS test is proving wiring.register_all()'s AFTER_OCR registration
    (governance/pii.py::register._scrub) actually fires when the real entry point runs, and that
    it actually redacts -- governance/pii.py is now implemented for real (was DP-3's second
    cross-team blocker), so this asserts a clean run through store.build() instead of a crash, and
    plants a fake email in the canned OCR output to prove _scrub's redaction survives into the
    persisted chunks.jsonl (not just that build_knowledge_base() didn't raise)."""
    raw_dir = tmp_path / "data" / "raw" / "testwork"
    raw_dir.mkdir(parents=True)
    # Unlike the sibling test above (where Enhancer.apply() raises before ever looking at pixel
    # content), this run reaches preprocess.run()'s blank-page filter for real -- a page dropped
    # there never gets processed_dir created, which chunk.split() needs to exist later. A plain
    # white image would be dropped, so paint a small dark block to clear the ink-ratio threshold.
    img = Image.new("L", (50, 50), color=255)
    img.paste(0, (10, 10, 40, 40))
    img.save(raw_dir / "1.png")

    cfg = {
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "index_dir": str(tmp_path / "data" / "processed" / "index"),
        },
        "enhance": {"enabled": False, "model": "unet_small", "type": "diffusion"},
        "agent": {},  # see the sibling test above for why wiring.register_all() needs this key
        "index": {"chunk_tokens": 50, "overlap": 5, "type": "faiss:flat"},
        "embed": {"model": "intfloat/multilingual-e5-base", "batch_size": 2},
        "device": "cpu",
    }
    # Canned, deterministic output -- content is irrelevant except the planted email, which is
    # only there to prove AFTER_OCR's PII scrub actually ran. pipeline.py imports `layout`/`ocr` as
    # modules (not their functions), so patching the attribute on the module object patches what
    # pipeline.build_knowledge_base() actually calls.
    monkeypatch.setattr(
        pipeline.layout,
        "detect",
        lambda pages, cfg: [Region(page_id="testwork_p0001", bbox=(0, 0, 10, 10), kind="text")],
    )
    monkeypatch.setattr(
        pipeline.ocr,
        "transcribe",
        lambda regions, cfg: [
            Chunk(
                id="testwork_p0001_l000",
                doc_id="testwork",
                text="line, contact rabi@example.com for details",
                page_ids=["testwork_p0001"],
            )
        ],
    )
    fake_model = _FakeEmbedModel(dim=8)
    monkeypatch.setattr(embed, "_load_model", lambda model_name, device: fake_model)

    pipeline.build_knowledge_base(cfg)

    faiss_index, chunk_rows = store.load(cfg)
    assert faiss_index.ntotal == len(chunk_rows) == 1
    assert "rabi@example.com" not in chunk_rows[0]["text"]
    assert "[REDACTED_EMAIL]" in chunk_rows[0]["text"]
