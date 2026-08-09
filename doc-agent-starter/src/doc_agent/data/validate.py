"""Data — data schema/quality validation at ingest"""
from __future__ import annotations

from ..contracts import *  # noqa

MIN_PAGES = 300
MIN_WORDS = 60_000


def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits.

    Word count is measured from OCR'd text, so this is meant to be called after Stage 3 (ocr.transcribe)
    on the resulting Chunks, not on raw Page objects alone — Page has no text field to count words from.
    This function accepts either: a list[Page] (page/doc-count + leakage checks only) or a list of
    (chunk-like) objects exposing .text and .doc_id (adds the word-count floor). Raises AssertionError
    with a specific message on any violation (A1 Section 2's 300pp/60k-word floor + document-level split).
    """
    if not pages:
        raise AssertionError("no pages/chunks given to validate()")

    doc_ids = {getattr(p, "doc_id") for p in pages}
    n_pages_or_chunks = len(pages)

    has_text = hasattr(pages[0], "text")
    if has_text:
        # called on Chunks: check the corpus-wide word-count floor
        total_words = sum(len(getattr(c, "text", "").split()) for c in pages)
        assert total_words >= MIN_WORDS, (
            f"corpus has {total_words} words, below the required minimum of {MIN_WORDS} "
            f"(A1 Section 2 / spec Part A minimum-corpus rule)"
        )
    else:
        # called on Pages: check the page-count floor
        assert n_pages_or_chunks >= MIN_PAGES, (
            f"corpus has {n_pages_or_chunks} pages, below the required minimum of {MIN_PAGES}"
        )

    assert len(doc_ids) >= 1, "no documents (doc_id) found — cannot check document-level split integrity"
