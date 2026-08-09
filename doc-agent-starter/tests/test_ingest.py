"""Unit test home for ingest. IMPLEMENT — CI runs these."""
import pytest

# @pytest.mark.skip(reason="students: implement ingest unit tests")
# def test_ingest_placeholder():
#     assert True
"""Unit test home for ingest. CI runs these."""
from doc_agent.ingest import loader, preprocess


def test_load_pages_groups_by_document_and_sorts(synthetic_corpus):
    pages = loader.load_pages(synthetic_corpus)
    assert [p.id for p in pages] == ["testwork_p0001", "testwork_p0002"]
    assert all(p.doc_id == "testwork" for p in pages)


def test_load_pages_missing_dir_raises(tmp_path):
    cfg = {"paths": {"raw_dir": str(tmp_path / "nonexistent")}}
    
    with pytest.raises(FileNotFoundError):
        loader.load_pages(cfg)


def test_preprocess_drops_blank_pages(synthetic_corpus, tmp_path):
    from PIL import Image
    raw_dir = tmp_path / "data" / "raw" / "testwork"
    Image.new("L", (800, 600), color=255).save(raw_dir / "3.png")  # blank page, no text drawn

    pages = loader.load_pages(synthetic_corpus)
    assert len(pages) == 3

    out = preprocess.run(pages, synthetic_corpus)
    assert len(out) == 2  # the blank page is dropped
    assert all(p.id != "testwork_p0003" for p in out)


def test_preprocess_writes_processed_images(synthetic_corpus):
    from pathlib import Path
    pages = loader.load_pages(synthetic_corpus)
    out = preprocess.run(pages, synthetic_corpus)
    for p in out:
        assert Path(p.image_path).exists()
        assert "processed" in p.image_path
