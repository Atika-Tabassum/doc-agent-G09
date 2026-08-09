"""Stage 1 — load scanned page-images"""
from __future__ import annotations

import re
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Accepts either:
#   data/raw/<doc_id>/<anything with a page number>.ext      (one subfolder per literary work — preferred)
#   data/raw/<doc_id>_p<page_number>.ext                     (flat layout, doc_id parsed from the filename)
_PAGE_NUM_RE = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in the stem = physical page number
_FLAT_DOC_RE = re.compile(r"^(?P<doc_id>.+?)_p(?P<page>\d+)$")


def _page_num(stem: str) -> int:
    m = _PAGE_NUM_RE.search(stem)
    if not m:
        raise ValueError(f"cannot find a page number in filename stem '{stem}'")
    return int(m.group(1))


def _resolve_doc_and_page(path: Path, raw_dir: Path) -> tuple[str, int]:
    rel = path.relative_to(raw_dir)
    if len(rel.parts) >= 2:
        # data/raw/<doc_id>/.../<file>.ext — the top-level subfolder is the work
        doc_id = rel.parts[0]
        page_num = _page_num(path.stem)
        return doc_id, page_num
    m = _FLAT_DOC_RE.match(path.stem)
    if m:
        return m.group("doc_id"), int(m.group("page"))
    # last resort: no doc grouping info at all in a flat layout — treat the whole corpus as one doc.
    return "corpus", _page_num(path.stem)


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page], sorted by (doc_id, page_number).

    data/raw/ is expected to hold one subfolder per literary work (the document-level split unit
    A1 Section 2 relies on for leakage-free train/val/test splits), each containing that work's
    page-image scans. See scripts/get_data.sh for how this layout gets populated.
    """
    raw_dir = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Run scripts/get_data.sh to populate the corpus first."
        )

    candidates = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not candidates:
        raise FileNotFoundError(
            f"no page images found under {raw_dir} "
            f"(looked for {sorted(IMAGE_EXTS)}). Run scripts/get_data.sh first."
        )

    pages: list[Page] = []
    for path in candidates:
        doc_id, page_num = _resolve_doc_and_page(path, raw_dir)
        page_id = f"{doc_id}_p{page_num:04d}"
        pages.append(Page(id=page_id, image_path=str(path), doc_id=doc_id))

    pages.sort(key=lambda p: p.id)
    logger.info(f"loaded {len(pages)} pages across {len({p.doc_id for p in pages})} documents from {raw_dir}")
    return pages
