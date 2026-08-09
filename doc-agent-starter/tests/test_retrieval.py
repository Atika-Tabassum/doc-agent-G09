"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

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
