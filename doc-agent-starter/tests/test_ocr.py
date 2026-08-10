"""Unit test home for OCR. CI runs these."""
from __future__ import annotations

import json

from doc_agent.ingest import loader, preprocess
from doc_agent.vision import layout, ocr

# synthetic_corpus fixture now lives in tests/conftest.py (shared with test_ingest.py, which
# depends on it too) -- pytest injects it automatically into any test function in this directory
# that takes a `synthetic_corpus` parameter, no import needed.


def _actual_ocr_lines(cfg: dict, page_id: str) -> list[str]:
    """Run the real pipeline once (no gold/reference present) to discover what THIS environment's
    Tesseract actually reads for a page's lines, in region order. Hardcoding an assumed OCR output
    string is fragile across Tesseract versions, installed fonts, and antialiasing -- even
    machine-printed English can come out with a stray character difference on a different machine.
    Deriving the expected text from a real run keeps these tests deterministic regardless of
    environment, while still genuinely exercising the gate logic (not real-world OCR accuracy)."""
    pages = preprocess.run(loader.load_pages(cfg), cfg)
    regions = layout.detect(pages, cfg)
    ocr.transcribe(regions, cfg)  # every region gets an ocr_meta.jsonl row, even when rejected
    from pathlib import Path
    rows = sorted(
        [r for r in (json.loads(l) for l in
         open(Path(cfg["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8"))
         if r["page_id"] == page_id],
        key=lambda r: r["region_id"],
    )
    return [r["ocr_text_normalized"] for r in rows]


def test_transcribe_produces_line_chunks_for_accepted_regions(synthetic_corpus, tmp_path):
    """A region only becomes a Chunk once it's ACCEPTED by the Stage 3 gate (gold, or a passing CER
    against a reference) -- an unverifiable line (no gold, no reachable reference) must NOT silently
    become an indexable Chunk. Uses a gold label built from this environment's actual Tesseract output
    (not an assumed string) -- gold requires CER==0.0, not mere page membership."""
    actual_lines = _actual_ocr_lines(synthetic_corpus, "testwork_p0001")
    assert actual_lines  # sanity: the fixture image produced something to work with

    labels_path = tmp_path / "grading_kit" / "labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps({"page_id": "testwork_p0001", "text": "\n".join(actual_lines)}) + "\n", encoding="utf-8"
    )

    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    assert len(regions) > 0
    assert all(r.kind == "text" for r in regions)

    chunks = ocr.transcribe(regions, synthetic_corpus)
    assert len(chunks) > 0                               # the exactly-matching gold lines were accepted
    assert all(c.text.strip() for c in chunks)
    assert all(len(c.page_ids) == 1 for c in chunks)


def test_transcribe_rejects_unverifiable_lines_with_full_audit_trail(synthetic_corpus):
    """No gold label and no reachable IA reference -> every region must be REJECTED (no Chunk created,
    keeping the vector index clean), but every region still gets a full audit row in ocr_meta.jsonl --
    a rejected line must never be silently dropped with no trace."""
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    chunks = ocr.transcribe(regions, synthetic_corpus)

    from pathlib import Path
    meta_path = Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl"
    assert meta_path.exists()
    rows = [json.loads(line) for line in open(meta_path, encoding="utf-8")]

    assert len(chunks) == 0                              # nothing verifiable -> nothing indexed
    assert len(rows) == len(regions)                      # but every region still gets an audit row
    assert all(0.0 <= r["ocr_confidence"] <= 1.0 for r in rows)
    assert all(r["accepted"] is False for r in rows)
    assert all(r["evidence_tier"] == "raw" for r in rows)
    assert all(r["reject_reason"] == "reference_alignment_failed" for r in rows)
    assert all(r["ocr_text_raw"] and r["ocr_text_normalized"] for r in rows)   # raw OCR still captured
    assert all(r["reference_text_normalized"] is None and r["cer"] is None for r in rows)


def test_transcribe_marks_heldout_pages_as_gold(synthetic_corpus, tmp_path):
    actual_lines = _actual_ocr_lines(synthetic_corpus, "testwork_p0001")
    assert actual_lines

    labels_path = tmp_path / "grading_kit" / "labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps({"page_id": "testwork_p0001", "text": "\n".join(actual_lines)}) + "\n", encoding="utf-8"
    )

    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    chunks = ocr.transcribe(regions, synthetic_corpus)

    from pathlib import Path
    rows = [json.loads(line) for line in
            open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8")]
    p1_rows = [r for r in rows if r["chunk_id"].startswith("testwork_p0001_")]
    p2_rows = [r for r in rows if r["chunk_id"].startswith("testwork_p0002_")]
    # gold page, exact match: accepted as "gold"
    assert p1_rows and all(r["evidence_tier"] == "gold" and r["accepted"] for r in p1_rows)
    # non-gold page with no reachable reference: still correctly rejected, not silently "silver"
    assert p2_rows and all(r["evidence_tier"] == "raw" and not r["accepted"] for r in p2_rows)


def test_transcribe_gold_requires_exact_match_not_just_page_membership(synthetic_corpus, tmp_path):
    """A page being in grading_kit/labels.jsonl is NOT sufficient for gold -- only a line whose OCR
    exactly matches (CER==0.0) the correspondingly-aligned gold text earns 'gold'. A mismatched line
    on the same gold page must fall through to the standard silver/raw gate, not be auto-rejected or
    auto-accepted as gold."""
    actual_lines = _actual_ocr_lines(synthetic_corpus, "testwork_p0001")
    assert len(actual_lines) >= 3

    # first two lines: this environment's real Tesseract output (will exact-match); third: deliberately wrong
    gold_lines = list(actual_lines)
    gold_lines[2] = "THIS DOES NOT MATCH TESSERACT"

    labels_path = tmp_path / "grading_kit" / "labels.jsonl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps({"page_id": "testwork_p0001", "text": "\n".join(gold_lines)}) + "\n", encoding="utf-8"
    )

    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    ocr.transcribe(regions, synthetic_corpus)

    from pathlib import Path
    rows = sorted(
        [r for r in (json.loads(l) for l in
         open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8"))
         if r["page_id"] == "testwork_p0001"],
        key=lambda r: r["region_id"],
    )
    assert rows[0]["evidence_tier"] == "gold" and rows[0]["cer"] == 0.0
    assert rows[1]["evidence_tier"] == "gold" and rows[1]["cer"] == 0.0
    # the mismatched line must NOT be labeled gold just because its page is in labels.jsonl
    assert rows[2]["evidence_tier"] != "gold"
    assert not rows[2]["accepted"]  # no IA reference available either -> falls through to rejection


def test_transcribe_reference_alignment_requires_exact_line_count(synthetic_corpus):
    """A reference file whose line count doesn't exactly match the detected region count must refuse
    alignment entirely (reference_alignment_failed for every line on that page), even when the two
    counts are close -- positional correspondence isn't trustworthy on anything less than an exact
    match, since a single-line offset silently misaligns every line after it."""
    from pathlib import Path
    raw_dir = Path(synthetic_corpus["paths"]["raw_dir"]) / "testwork"
    # page 1 has 3 detected lines; give it 4 reference lines (a "close" mismatch)
    (raw_dir / "testwork_p0001.ref.txt").write_text(
        "The quick brown fox jumps.\nOver the lazy dog again.\nThird line of test text.\nEXTRA LINE\n",
        encoding="utf-8",
    )
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    ocr.transcribe(regions, synthetic_corpus)

    rows = [r for r in (json.loads(l) for l in
            open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8"))
            if r["page_id"] == "testwork_p0001"]
    assert rows and all(r["reject_reason"] == "reference_alignment_failed" for r in rows)


def test_transcribe_cer_gate_accepts_matching_and_rejects_mismatched_lines(synthetic_corpus):
    """With a real (non-gold) reference file present, a line whose OCR closely matches its reference
    line must be accepted as 'silver'; a line that doesn't must be rejected as cer_above_threshold."""
    actual_lines = _actual_ocr_lines(synthetic_corpus, "testwork_p0001")
    assert len(actual_lines) >= 3

    # first two lines: this environment's real Tesseract output (will exact-match); third: garbage
    ref_lines = list(actual_lines)
    ref_lines[2] = "COMPLETELY UNRELATED GARBAGE TEXT HERE"

    from pathlib import Path
    raw_dir = Path(synthetic_corpus["paths"]["raw_dir"]) / "testwork"
    (raw_dir / "testwork_p0001.ref.txt").write_text("\n".join(ref_lines) + "\n", encoding="utf-8")

    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    chunks = ocr.transcribe(regions, synthetic_corpus)

    rows = sorted(
        [json.loads(line) for line in
         open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8")
         if json.loads(line)["page_id"] == "testwork_p0001"],
        key=lambda r: r["region_id"],
    )
    assert rows[0]["accepted"] and rows[0]["evidence_tier"] == "silver" and rows[0]["cer"] == 0.0
    assert rows[1]["accepted"] and rows[1]["evidence_tier"] == "silver" and rows[1]["cer"] == 0.0
    assert not rows[2]["accepted"] and rows[2]["reject_reason"] == "cer_above_threshold"
    assert rows[2]["evidence_tier"] == "raw"
    assert rows[2]["reference_text_normalized"] == "COMPLETELY UNRELATED GARBAGE TEXT HERE"


def test_transcribe_rejects_too_short_lines_regardless_of_tier(synthetic_corpus):
    """A line whose normalized OCR output is shorter than cfg['ocr']['min_line_chars'] must be
    rejected as REJECT_TOO_SHORT -- checked before the gold/silver branch, so it applies universally."""
    synthetic_corpus["ocr"]["min_line_chars"] = 1000  # guarantees every real transcribed line fails it
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    chunks = ocr.transcribe(regions, synthetic_corpus)

    from pathlib import Path
    rows = [json.loads(line) for line in
            open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8")]
    assert len(chunks) == 0
    assert all(r["reject_reason"] == "REJECT_TOO_SHORT" for r in rows)
    assert all(r["evidence_tier"] == "raw" for r in rows)
    assert all(r["ocr_text_raw"] for r in rows)  # still captured for debugging even when rejected


def test_transcribe_region_id_matches_layout_meta(synthetic_corpus):
    """ocr_meta.jsonl's region_id must be the exact same identifier layout_meta.jsonl assigned to that
    region, so a downstream cite tool can join the two sidecars on a real key."""
    from pathlib import Path
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    ocr.transcribe(regions, synthetic_corpus)

    processed_dir = Path(synthetic_corpus["paths"]["processed_dir"])
    layout_rows = {r["region_id"]: r for r in
                   (json.loads(l) for l in open(processed_dir / "layout_meta.jsonl", encoding="utf-8"))}
    ocr_rows = [json.loads(l) for l in open(processed_dir / "ocr_meta.jsonl", encoding="utf-8")]

    assert ocr_rows  # sanity: there's something to check
    for row in ocr_rows:
        assert row["region_id"] in layout_rows, f"{row['region_id']} has no matching layout_meta.jsonl entry"


def test_transcribe_rejects_layout_provenance_mismatch(synthetic_corpus):
    """If a region's bbox doesn't match what layout_meta.jsonl recorded for that region_id (a Stage
    2/3 desync), it must be REJECTED with reject_reason='layout_provenance_mismatch' -- not silently
    accepted with a warning, and not a hard crash that would kill the whole batch ingest run."""
    from pathlib import Path
    from doc_agent.contracts import Region

    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    tampered = list(regions)
    tampered[0] = Region(page_id=tampered[0].page_id, bbox=(9999, 9999, 5, 5), kind="text")

    chunks = ocr.transcribe(tampered, synthetic_corpus)  # must not raise

    rows = [json.loads(line) for line in
            open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8")]
    tampered_row = next(r for r in rows if r["region_id"] == "testwork_p0001_r000")
    assert tampered_row["reject_reason"] == "layout_provenance_mismatch"
    assert tampered_row["evidence_tier"] == "raw"
    assert not tampered_row["accepted"]
    assert len(rows) == len(tampered)  # batch continued -- every region still produced an audit row


def test_transcribe_every_row_has_reproducibility_fields(synthetic_corpus):
    """Every ocr_meta.jsonl row -- accepted or rejected -- must carry the exact cer_threshold and
    ocr_config used to produce that decision, so the row is reconstructable even after configs/
    config.yaml changes later."""
    synthetic_corpus["ocr"]["max_cer_target"] = 0.2
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)
    ocr.transcribe(regions, synthetic_corpus)

    from pathlib import Path
    rows = [json.loads(line) for line in
            open(Path(synthetic_corpus["paths"]["processed_dir"]) / "ocr_meta.jsonl", encoding="utf-8")]
    assert rows
    assert all(r["cer_threshold"] == 0.2 for r in rows)
    lang = synthetic_corpus["ocr"]["lang"]
    assert all(r["ocr_config"] == f"--oem 1 --psm 7 -l {lang}" for r in rows)


def test_layout_orders_lines_top_to_bottom_left_to_right(synthetic_corpus):
    """Reading order must come from an explicit spatial sort, not Tesseract's raw dict iteration
    order (a stage direction or heading Tesseract assigns a *different* line_num to, but that sits on
    the same physical row as body text, must still come out left-to-right within that row)."""
    from doc_agent.vision.layout import _group_words_into_lines, _spatial_reading_order, _y_tolerance

    # Deliberately fed in scrambled (non-reading) order, with two entries on the same visual row
    # (close center_y) but different Tesseract line_num -- exactly the centered-heading /
    # right-aligned-stage-direction case from the review.
    data = {
        "text":      ["রাইট", "লেফট",  "সেকেন্ড"],
        "conf":      [90, 91, 89],
        "left":      [600, 50, 50],
        "top":       [100, 102, 200],
        "width":     [80, 100, 90],
        "height":    [30, 28, 30],
        "block_num": [1, 1, 1],
        "par_num":   [1, 1, 1],
        "line_num":  [1, 2, 3],   # note: the two same-row words have DIFFERENT line_num
    }
    lines = _group_words_into_lines(data, score_thr=0.0)
    ordered = _spatial_reading_order(lines, _y_tolerance(lines, cfg={}))

    # same-row pair (center_y ~115-116) must be merged into one band and sorted left (x=50) before
    # right (x=600); the third, vertically-distant line must come last.
    assert [ln["bbox"][0] for ln in ordered] == [50, 600, 50]
    assert ordered[-1]["line_num"] == 3


def test_layout_writes_traceability_sidecar(synthetic_corpus):
    """contracts.Region has no field for Tesseract's block/par/line provenance (fixed) -- it must be
    recoverable from layout_meta.jsonl instead, keyed to match the returned Region order 1:1."""
    from pathlib import Path
    pages = preprocess.run(loader.load_pages(synthetic_corpus), synthetic_corpus)
    regions = layout.detect(pages, synthetic_corpus)

    meta_path = Path(synthetic_corpus["paths"]["processed_dir"]) / "layout_meta.jsonl"
    assert meta_path.exists()
    rows = [json.loads(line) for line in open(meta_path, encoding="utf-8")]
    assert len(rows) == len(regions)
    for row in rows:
        assert {"region_id", "page_id", "bbox", "block_num", "par_num", "line_num"} <= row.keys()

    # region_id indices must be 0..n-1 per page, in the same order that page's regions appear in the
    # returned `regions` list — this is what lets a downstream consumer match a Region back to its
    # Tesseract provenance by position, since contracts.Region itself carries no id.
    by_page: dict[str, list[str]] = {}
    for row in rows:
        by_page.setdefault(row["page_id"], []).append(row["region_id"])
    for page_id, ids in by_page.items():
        assert ids == [f"{page_id}_r{i:03d}" for i in range(len(ids))]