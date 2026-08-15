# #!/usr/bin/env bash
# # A1 — fetch or recreate your scanned corpus into data/raw/
# # IMPLEMENT: download/prepare your corpus (public-domain / openly licensed).
# set -euo pipefail
# echo "TODO: fetch corpus into data/raw/"
# IA_IDENTIFIER="RABINDRARACHANABALI"
# RAW_DIR="data/raw"
# TMP_DIR="$(mktemp -d)"
# trap 'rm -rf "$TMP_DIR"' EXIT

# echo "Fetching Internet Archive item metadata for ${IA_IDENTIFIER}..."
# curl -fsSL "https://archive.org/metadata/${IA_IDENTIFIER}" -o "${TMP_DIR}/metadata.json"

# # The item is the full 27-volume set; Volume 25 is the scoped A1 corpus (A1 Section 2). Internet
# # Archive serves a per-item page-image JP2 stack via the *_jp2.zip file, and a companion single-page
# # PDF whose pages we can rasterise directly if the JP2 stack isn't present for this sub-item.
# VOL25_PDF="${TMP_DIR}/vol25.pdf"
# echo "Downloading Volume 25 scan (PDF, image-only, no text layer)..."
# curl -fsSL "https://archive.org/download/${IA_IDENTIFIER}/RABINDRA%20RACHANABALI%20-%2025TH%20VOL.pdf" -o "${VOL25_PDF}" \
#   || { echo "Adjust the exact sub-item filename above to match the real IA item listing for this volume." >&2; exit 1; }

# WORK_DIR="${RAW_DIR}/rachanabali_vol25"
# mkdir -p "${WORK_DIR}"

# echo "Rasterising pages to ${WORK_DIR}/ at 300 DPI..."
# pdftoppm -png -r 300 "${VOL25_PDF}" "${WORK_DIR}/page"

# # pdftoppm names output page-000001.png etc.; rename to the <doc_id>/<page_number>.ext convention
# # ingest/loader.py expects (a bare page number is enough — the subfolder name supplies doc_id).
# i=1
# for f in "${WORK_DIR}"/page-*.png; do
#   mv "$f" "${WORK_DIR}/${i}.png"
#   i=$((i + 1))
# done

# n_pages=$(find "${WORK_DIR}" -name '*.png' | wc -l)
# echo "Done: ${n_pages} pages written to ${WORK_DIR}/"
# echo
# echo "IMPORTANT — this script alone does NOT split Volume 25 into its 12 constituent literary works."
# echo "It writes every page under one doc_id (rachanabali_vol25/), which is WRONG for the document-level"
# echo "split A1 Section 2 depends on (train/val/test must never share a work). You still need to:"
# echo "  1. Identify each work's page range (from the volume's running headers / table of contents)."
# echo "  2. Re-split ${WORK_DIR}/ into data/raw/<work_name>/<page>.png per work, using your SPLIT_MAP."
# echo "  3. THEN run: python -c \"from doc_agent.data import versioning; print(versioning.snapshot('${RAW_DIR}'))\""
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

# --- Split rachanabali_vol25/ into one subfolder per literary work ---------------------------
# loader.py prefers data/raw/<doc_id>/<page>.ext (one subfolder per work) over a flat layout --
# that's the document-level grouping A1 Section 2's train/val/test split relies on (a whole work
# must land in exactly one partition, never split across pages). Page ranges below are this
# sub-item's actual table-of-contents boundaries within the 437-page Volume 25 scan (verified
# against the volume's running headers), not an estimate. Page numbers are kept as the ORIGINAL
# volume-wide page number (not renumbered per-work from 1) so a filename like tin_sangi/218.png
# still tells you which physical page of the scanned volume it came from.
echo
echo "Splitting ${WORK_DIR}/ into per-work subfolders..."

WORK_NAMES=(arogya chitrangada chandalika tin_sangi bishwaparichay)
WORK_STARTS=(37 121 153 209 329)
WORK_ENDS=(120 152 208 328 432)

n_split=0
n_unassigned=0
for f in "${WORK_DIR}"/*.png; do
  page_num=$(basename "$f" .png)
  work=""
  for idx in "${!WORK_NAMES[@]}"; do
    if (( page_num >= WORK_STARTS[idx] && page_num <= WORK_ENDS[idx] )); then
      work="${WORK_NAMES[idx]}"
      break
    fi
  done
  if [[ -n "$work" ]]; then
    mkdir -p "${RAW_DIR}/${work}"
    mv "$f" "${RAW_DIR}/${work}/${page_num}.png"
    n_split=$((n_split + 1))
  else
    # Outside all 5 works' ranges -- front/back matter (title page, TOC, colophon, etc.), not
    # part of this project's corpus scope. Dropped rather than left as an unsplit, undocumented
    # doc_id, which would silently duplicate/pollute the document-level split.
    rm -f "$f"
    n_unassigned=$((n_unassigned + 1))
  fi
done
rmdir "${WORK_DIR}" 2>/dev/null || true

echo "Split ${n_split} pages into 5 work subfolders under ${RAW_DIR}/ (dropped ${n_unassigned} front/back-matter pages outside all 5 ranges)."
for idx in "${!WORK_NAMES[@]}"; do
  work="${WORK_NAMES[idx]}"
  count=$(find "${RAW_DIR}/${work}" -name '*.png' 2>/dev/null | wc -l)
  echo "  ${work}: ${count} pages (expected $(( WORK_ENDS[idx] - WORK_STARTS[idx] + 1 )))"
done

echo
echo "Next: python -c \"from doc_agent.data import versioning; print(versioning.snapshot('${RAW_DIR}'))\""