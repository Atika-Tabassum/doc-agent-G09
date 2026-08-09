#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# IMPLEMENT: download/prepare your corpus (public-domain / openly licensed).
set -euo pipefail
echo "TODO: fetch corpus into data/raw/"
IA_IDENTIFIER="RABINDRARACHANABALI"
RAW_DIR="data/raw"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Fetching Internet Archive item metadata for ${IA_IDENTIFIER}..."
curl -fsSL "https://archive.org/metadata/${IA_IDENTIFIER}" -o "${TMP_DIR}/metadata.json"

# The item is the full 27-volume set; Volume 25 is the scoped A1 corpus (A1 Section 2). Internet
# Archive serves a per-item page-image JP2 stack via the *_jp2.zip file, and a companion single-page
# PDF whose pages we can rasterise directly if the JP2 stack isn't present for this sub-item.
VOL25_PDF="${TMP_DIR}/vol25.pdf"
echo "Downloading Volume 25 scan (PDF, image-only, no text layer)..."
curl -fsSL "https://archive.org/download/${IA_IDENTIFIER}/RABINDRA%20RACHANABALI%20-%2025TH%20VOL.pdf" -o "${VOL25_PDF}" \
  || { echo "Adjust the exact sub-item filename above to match the real IA item listing for this volume." >&2; exit 1; }

WORK_DIR="${RAW_DIR}/rachanabali_vol25"
mkdir -p "${WORK_DIR}"

echo "Rasterising pages to ${WORK_DIR}/ at 300 DPI..."
pdftoppm -png -r 300 "${VOL25_PDF}" "${WORK_DIR}/page"

# pdftoppm names output page-000001.png etc.; rename to the <doc_id>/<page_number>.ext convention
# ingest/loader.py expects (a bare page number is enough — the subfolder name supplies doc_id).
i=1
for f in "${WORK_DIR}"/page-*.png; do
  mv "$f" "${WORK_DIR}/${i}.png"
  i=$((i + 1))
done

n_pages=$(find "${WORK_DIR}" -name '*.png' | wc -l)
echo "Done: ${n_pages} pages written to ${WORK_DIR}/"
echo
echo "IMPORTANT — this script alone does NOT split Volume 25 into its 12 constituent literary works."
echo "It writes every page under one doc_id (rachanabali_vol25/), which is WRONG for the document-level"
echo "split A1 Section 2 depends on (train/val/test must never share a work). You still need to:"
echo "  1. Identify each work's page range (from the volume's running headers / table of contents)."
echo "  2. Re-split ${WORK_DIR}/ into data/raw/<work_name>/<page>.png per work, using your SPLIT_MAP."
echo "  3. THEN run: python -c \"from doc_agent.data import versioning; print(versioning.snapshot('${RAW_DIR}'))\""
