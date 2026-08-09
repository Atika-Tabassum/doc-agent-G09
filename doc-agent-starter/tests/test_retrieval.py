"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk


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
