# Per-stage design choices (A2 deliverable)

| Stage | Problem statement | Data | Model | Methods | Design | Development | Deployment | MLOps |
|---|---|---|---|---|---|---|---|---|
| 0 Frame | Given a scanned Rachanabali (Vol. 25) page, answer a question with a page/line citation and an explicit silver/gold evidence-tier disclosure, or abstain. Groundedness ≥ 0.80. | 437 scanned pages, CC0, Vol. 25, split by literary work (train 65% / val 22% / test 13% by word count). | — (framing only, no model at Stage 0). | — | Splitting by work, not page, guarantees no document appears in two partitions. | eda.ipynb runs top-to-bottom from a clean clone + get_data.sh. | Batch offline for ingest/OCR; only the agent loop is online per query. | Content-hash corpus versions (data/versioning.py). |
| 1 Ingest+Enhance | Clean scans without inventing content: deskew, denoise, binarize; no generative repair. | 1-bit bitonal PNGs embedded in the source PDF, ~300 DPI, single volume. | Classical: Hough/minAreaRect deskew, conditional NLM denoise, Sauvola-style adaptive threshold (Otsu fallback). Enhancer (VAE/diffusion) present and `enhance.enabled: true`. | Deskew → conditional denoise → binarize → blank-page drop (ink-pixel ratio < 0.002). | Stage 1's output contract is image+metadata only, never text — a bad preprocessing decision degrades an image a human can re-inspect, never silently corrupts already-extracted text. | Runs against `data/raw/<work>/<page>.png`; writes to `data/processed/<work>/`. | One-time batch pass over the 437-page volume, cached. | Re-ingest triggered by any change to OCR model/chunker/layout config. |
| 2 Layout | Detect text lines only — no tables/figures/multi-column headings exist in this corpus (A1 Section 3). | Predominantly single-column; variation from poetry indentation/headings, not page structure. | Tesseract-ben's own line-detection pass (`image_to_data`, block/paragraph/line grouping) — no trained detector. | Group word boxes into lines by (block, par, line); drop lines below `layout.score_thr`. | Deliberately not a trained column/layout model: Section 3's findings say a dedicated detector would solve a problem this corpus doesn't have. | — | Runs once per page during the same batch ingest pass as Stage 1. | Line regions are not persisted independently; consumed immediately by Stage 3. |
| 3 OCR | Transcribe each detected line; disclose OCR confidence and silver/gold tier per line (contracts.py has neither field — recorded in `ocr_meta.jsonl`). | Silver-only reference text (IA's own OCR is the only text layer this volume ships with); tier = gold only for lines inside `grading_kit/heldout_pages` + `labels.jsonl`. | **Tesseract-ben**, frozen `--oem 1 --psm 3`, config swept on validation-only lines (A1 notebook §5.0): test CER 0.0899 / WER 0.3085, under the 0.15 target. CRNN+BiLSTM+CTC and `baidu/Unlimited-OCR` scoped but gated off (need a GPU we did not have for A1). | Crop each line region with a small pixel margin, re-OCR with `--psm 7` (single line), average word confidences. | One Chunk per transcribed line; empty-text lines are dropped rather than kept as noise. | — | Deployed model is CPU-only by design — no GPU dependency for the committed recognizer. | `ocr_meta.jsonl` is the audit trail linking every chunk back to its confidence + tier — required by the Explainable NFR's `tier(a_i)` condition. |
| 4 Index | Re-chunk line-level OCR output into retrieval-sized windows; embed; build a searchable index. | Never merges chunks across documents (would reintroduce the leakage Section 2's document-level split exists to prevent). | Embedding: multilingual-e5-base (chosen over an English-only embedder — the corpus is Bengali). Index: FAISS HNSW, cosine via inner product on normalised vectors. | Sliding token window (`chunk_tokens=256`, `overlap=32`); confidence/tier aggregated per merged chunk (mean confidence; tier = gold only if every constituent line is gold — conservative). | `index/store.py` persists a `chunks.jsonl` sidecar in the same row order as the FAISS vectors, mapping a vector-search hit back to (id, doc_id, text, page_ids, confidence, tier) for the A3 retriever. | Unit-tested end-to-end (`tests/test_retrieval.py`): chunk windowing/overlap/cross-document isolation, embed shape/dtype/L2-norm/e5-prefix (mocked, plus one skip-marked real-model smoke test), store build/load round-trip + FAISS nearest-neighbor search + chunk_meta tier/confidence propagation, a full chunk→embed→store integration test, and two tripwire tests proving `pipeline.build_knowledge_base()` — the real entry point `scripts/build_index.sh` calls — currently fails *before* reaching this stage (`ingest/enhance.py`'s stub fires first, `governance/pii.py`'s second; both `NotImplementedError`, neither this stage's file to fix). A demo-scale run (`notebooks/kb_demo.ipynb`, synthetic 6-line corpus) completed successfully end-to-end: 4 merged chunks, embedding dim 768, tier breakdown `{gold: 1, silver: 3}`. The real 437-page corpus has not been indexed yet — blocked on Tesseract-ben not being installed locally plus the two upstream stubs above, tracked as open cross-team items. | Demo-scale run: batch, CPU (`_resolve_device()` fell back from `config.yaml`'s `device: cuda` — no CUDA device available in this environment); no meaningful timing claim at this scale. The real corpus run's intended mode is unchanged from the original plan — batch, `device: cuda` per `config.yaml` — once Tesseract-ben and the two upstream stub fixes are available. | `chunk_meta.jsonl` records `source_line_ids` per merged chunk, so a citation can be traced back to its exact source line even after merging. |
| 5 Retrieval | *(A3 scope — not built in A2. `retrieval/retriever.py` still raises NotImplementedError by design; the config values it will read — k/k_step/k_max/weak_threshold — are pre-declared in configs/config.yaml.)* | | | | | | | |
| 6 Agent | *(A3 scope.)* | | | | | | | |
| 7 RL/RLVR | *(➕ bonus, A3/A4 scope.)* | | | | | | | |
| 8 Serving | *(A4 scope.)* | | | | | | | |
| 9 Eval | *(A3/A4 scope.)* | | | | | | | |

## Stage 4 (Index) — detailed design notes

The table row above is the required per-facet summary; this section expands it with the actual
reasoning, trade-offs, and verification evidence behind each Stage 4 decision — grounded only in
what `index/chunk.py`, `index/embed.py`, `index/store.py`, `configs/config.yaml`, and
`tests/test_retrieval.py` actually implement and verify, not aspirational numbers.

**Chunking: sliding token window, not per-sentence or per-paragraph.** `chunk.split()` groups
line-level OCR `Chunk`s by `doc_id`, flattens each document's lines into one whitespace-token
stream (order preserved from Stage 2/3, never re-sorted), and slides a window of
`chunk_tokens=256` tokens across it, stepping forward by `chunk_tokens - overlap = 224` tokens each
time. A fixed-size token window was chosen over sentence/paragraph-boundary chunking because the
line-level input has no reliable sentence-boundary signal of its own (OCR line breaks follow the
scanned page's physical layout, not sentence structure) — imposing a semantic boundary heuristic on
top of that would be guessing at structure the data doesn't actually carry. The trade-off is that a
chunk can end mid-sentence; overlap (below) is what keeps that from silently dropping context.

**Overlap: 32 tokens in production, preserving continuity across the cut.** Each window shares its
last `overlap` tokens with the next window's first `overlap` tokens — the mechanism is verified
directly by `test_split_merges_lines_within_a_document_with_overlap`, which runs the same windowing
logic at a smaller, hand-checkable scale (`chunk_tokens=6, overlap=2`) and asserts
`chunks[i].text.split()[-2:] == chunks[i+1].text.split()[:2]` for consecutive windows — the same
sharing property `chunk_tokens=256, overlap=32` produces in production, just easier to verify by eye
at 6 tokens than at 256. The practical effect: a sentence or clause that happens to straddle a window
boundary still appears in full inside at least one of the two adjacent chunks, rather than being
split with neither chunk holding the complete thought. `overlap >= chunk_tokens` is rejected with a
`ValueError` (`test_split_rejects_overlap_gte_chunk_tokens`) because the step would then be zero or
negative — the window would stall in place or walk backwards instead of advancing.

**Document-boundary isolation.** Chunks are built per `doc_id` and never merged across documents —
`by_doc: dict[str, list[Chunk]]` groups lines before any windowing happens, so there is no code path
where two different `doc_id`s' tokens end up in the same window
(`test_split_never_merges_across_documents`). This isn't a performance optimization; it's a
correctness requirement carried down from A1 Section 2's document-level train/val/test split — a
chunk that mixed text from two documents would make it possible for evaluation-time evidence to
leak information from a document nominally outside that split.

**Embedding model: `intfloat/multilingual-e5-base` (768-dim), chosen for language fit over an
English-only alternative.** `configs/config.yaml` originally pointed `embed.model` at
`sentence-transformers/all-MiniLM-L6-v2` — a strong, fast, but English-only sentence encoder — before
being corrected to `multilingual-e5-base`. The English-only model was never a viable long-term
choice for this corpus: literary Bengali text has no reliable English-embedding coverage, so
similarity scores from an English-only encoder would not be measuring semantic closeness in any
meaningful sense for Bengali queries against Bengali chunks. `multilingual-e5-base` was picked
instead because it's explicitly trained for cross-lingual retrieval (E5's "text embeddings by
weakly-supervised contrastive pre-training" line of models) with Bengali among its covered
languages, at a 768-dim size that still runs on CPU at demo scale (see Device Handling below)
without requiring GPU infrastructure this environment doesn't have.

**E5's asymmetric prefix convention.** E5 models are trained with a document/query asymmetry: text
being indexed is expected to be prefixed with `"passage: "`, and text being searched with is
expected to be prefixed with `"query: "` — mixing these up (or omitting them) doesn't error, it just
quietly degrades retrieval quality, since the model was never trained to treat unprefixed or
wrongly-prefixed text as either role. `embed.encode()` applies `"passage: "` automatically whenever
`"e5"` appears in the configured model name (`embed.py`), verified by
`test_encode_prefixes_e5_model_input_with_passage`, which asserts the exact string the mocked model
receives. The query-side `"query: "` prefix is deliberately **not** applied inside `embed.encode()`
— that half of the asymmetry belongs to A3's `retrieval/retriever.py`, which doesn't exist yet;
`notebooks/kb_demo.ipynb` demonstrates the correct query-side prefixing manually, once, for
illustration, rather than pre-empting retriever.py's responsibility.

**L2 normalization and why cosine similarity is the intended metric.** `embed.encode()` calls
`model.encode(..., normalize_embeddings=True)`, producing unit-length (L2 norm ≈ 1.0) vectors —
verified by `test_encode_output_shape_dtype_and_unit_norm` (`np.testing.assert_allclose(norms, 1.0, atol=1e-5)`).
This matters because `store.py` builds its FAISS index with `METRIC_INNER_PRODUCT`: the inner
product of two arbitrary vectors is dominated by their magnitudes and is not a similarity measure on
its own, but the inner product of two *unit* vectors is exactly their cosine similarity. Skipping
normalization would silently turn every similarity search into a magnitude comparison instead of a
direction (semantic-closeness) comparison, without raising any error — nothing about a mismatched
metric fails loudly. `test_store_nearest_neighbor_returns_exact_match` closes the loop end-to-end:
searching the built index with a chunk's own (already-normalized) embedding returns that chunk as
the top hit with score ≈ 1.0, confirming the normalization → inner-product → cosine chain actually
holds at runtime, not just in theory.

**FAISS index: HNSW for the real corpus, Flat for anything small.** `configs/config.yaml` sets
`index.type: faiss:hnsw`, which `store.build()` implements as `IndexHNSWFlat(dim, 32, METRIC_INNER_PRODUCT)`
with `efConstruction=200` — a graph-based approximate-nearest-neighbor index whose sub-linear query
time is the right trade-off once the corpus is large enough (the real 437-page corpus, once indexed,
is expected to produce on the order of hundreds of chunks at `chunk_tokens=256`). `store.py` also
supports `index.type: faiss:flat` (`IndexFlatIP`, exact brute-force search) as an explicit
alternative, used throughout `tests/test_retrieval.py` and `notebooks/kb_demo.ipynb` instead of HNSW
— deliberately, for two reasons verified directly rather than assumed: (1) HNSW's approximate search
makes an exact-top-1 assertion non-deterministic at small scale, which would make tests flaky; (2)
`faiss-cpu`'s `IndexHNSWFlat` was confirmed in this environment to **segfault** when `.add()` is
called with fewer vectors than the graph's `M` parameter (32) — reproduced directly with a 1-vector,
768-dim `.add()` call. This is a real property of the small-N-vs-M edge case in this FAISS build, not
a hypothetical concern, and it's the reason the demo notebook explicitly overrides `index.type` to
`faiss:flat` rather than using the production default at demo scale.

**Metadata propagation: a three-hop sidecar chain, because `contracts.Chunk` has no field for it.**
`contracts.py` is fixed and `Chunk` carries no OCR-confidence or evidence-tier field, so that
information travels alongside the `Chunk` objects in JSONL sidecars instead: Stage 3 writes
`ocr_meta.jsonl` keyed by line-level `chunk_id`, with `ocr_confidence` (float) and `evidence_tier`
(`"gold"`/`"silver"`/`"raw"`) per accepted line. `chunk.split()` reads that sidecar (fixing a real
bug where it previously read a key, `"tier"`, that `ocr_meta.jsonl` never wrote — regression-tested
by `test_split_reads_evidence_tier_key_without_crashing`) and aggregates it per merged chunk: mean
`ocr_confidence` across constituent lines, and `tier = "gold"` **only if every constituent line is
gold**, else `"silver"` — a conservative rule, since a merged chunk should disclose the weakest
evidence it contains, not the strongest. This is written to `chunk_meta.jsonl` keyed by the merged
`chunk_id`, along with `source_line_ids` so a citation can still be traced back to its exact source
line after merging. `store.build()` performs the final join, reading `chunk_meta.jsonl` by chunk id
and writing `chunks.jsonl` rows (`id, doc_id, text, page_ids, ocr_confidence, tier`) in the same
positional order as the FAISS vectors, so a vector-search hit's FAISS internal id maps directly to
its metadata row. `test_store_build_propagates_chunk_meta_tier_and_confidence` verifies this join is
correctly keyed (not, for example, silently matching by list position).

**Device handling: CPU-safe by default, not GPU-required.** `embed._resolve_device()` honors
`cfg['device']` but never hard-fails if `"cuda"` is requested and unavailable — it checks
`torch.cuda.is_available()` and falls back to `"cpu"` with a logged warning. This was exercised for
real, not just written defensively: this development environment has no CUDA-capable GPU
(`torch.cuda.is_available()` returns `False` here). The unit tests and the real-model smoke test
request `cfg['device'] = "cpu"` directly, so they don't exercise the fallback branch itself — but
`notebooks/kb_demo.ipynb` loads the real `configs/config.yaml`, whose `device: cuda` setting *does*
hit the fallback path, logging `"cfg['device']='cuda' but no CUDA device is available -- falling
back to cpu"` and then completing the embedding run (including the one-time
`multilingual-e5-base` download and load) successfully on CPU. The loaded model is cached per
`(model_name, device)` key (`embed._load_model`) so repeated `encode()` calls within one process
don't reload it from disk/hub each time.

**Verification summary.** Every claim above traces to a specific, currently-passing test in
`tests/test_retrieval.py` or a specific, actually-executed cell in `notebooks/kb_demo.ipynb` — not a
projected or assumed result: `test_split_reads_evidence_tier_key_without_crashing`,
`test_split_merges_lines_within_a_document_with_overlap`, `test_split_never_merges_across_documents`,
`test_split_rejects_overlap_gte_chunk_tokens`, `test_encode_output_shape_dtype_and_unit_norm`,
`test_encode_prefixes_e5_model_input_with_passage`, `test_encode_real_model_smoke` (skip-marked;
run manually once against the real model — shape `(1, 768)`, norm ≈ 1.0, confirmed passing),
`test_store_build_and_load_round_trip`, `test_store_load_raises_file_not_found_with_helpful_message`,
`test_store_nearest_neighbor_returns_exact_match`,
`test_store_build_propagates_chunk_meta_tier_and_confidence`,
`test_full_index_pipeline_chunk_embed_store_round_trip`, and the two
`pipeline.build_knowledge_base()` tripwire tests. The demo notebook's actually-executed, demo-scale
run (6 synthetic lines, `chunk_tokens=15`/`overlap=3` for demo purposes only — **not** the production
`256`/`32`) produced 4 merged chunks, 768-dim vectors, and a tier breakdown of `{gold: 1, silver: 3}`;
those are the only Stage 4 numbers currently backed by an actual run — the real 437-page corpus has
not been indexed yet (see the Development/Deployment cells above and `reports/pipeline_diagram.md`
for exactly what's still blocking that).
