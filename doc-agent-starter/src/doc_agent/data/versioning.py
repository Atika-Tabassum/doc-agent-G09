"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id.

    Content-hashes every file under corpus_dir (sorted by relative path, so the hash is stable
    regardless of filesystem iteration order) and records (version_id -> file list + sizes) to
    <corpus_dir>/../corpus_versions.jsonl. A1 Section 5, MLOps facet: version by content hash of
    (volume_id, pipeline_config_version) so a bad model/config swap can be rolled back without
    reprocessing the whole corpus.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files under {root} to version")

    hasher = hashlib.sha256()
    manifest = []
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        hasher.update(h.encode())
        manifest.append({"path": str(f.relative_to(root)), "sha256": h, "size": f.stat().st_size})

    version_id = hasher.hexdigest()[:16]
    log_path = root.parent / "corpus_versions.jsonl"
    with open(log_path, "a", encoding="utf-8") as out:
        out.write(json.dumps({"version_id": version_id, "corpus_dir": str(root),
                               "n_files": len(files), "manifest": manifest}, ensure_ascii=False) + "\n")

    return version_id
