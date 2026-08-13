"""Governance — PII detection + redaction (mandatory)"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..contracts import *  # noqa

_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?(?:\d[-\s]?){7,12}\d(?!\d)"),
    "ID_NUMBER": re.compile(r"(?<!\d)\d{9,16}(?!\d)"),
}

# Only these contract fields carry free text worth scrubbing. Deliberately NOT id/doc_id/page_ids/
# chunk_id/span: other stages (chunk_meta.jsonl, store.py, citations) key off those as join
# identifiers, and redaction must never mutate them.
_CONTENT_FIELDS: dict[type, tuple[str, ...]] = {
    Chunk: ("text",),
    Query: ("text",),
    Answer: ("text",),
}


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans. Regex-based, generic categories only (email/phone/
    long-digit-ID) -- this corpus is 1940s-published literature (Rabindranath Tagore's own text),
    so real PII is not expected in it, and a name/NER-based detector would risk redacting his own
    characters' names. Overlapping matches are resolved by keeping the earliest start, then the
    longest match at that position."""
    candidates = sorted(
        ((m.start(), m.end(), kind) for kind, pat in _PATTERNS.items() for m in pat.finditer(text)),
        key=lambda s: (s[0], -(s[1] - s[0])),
    )
    spans: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, kind in candidates:
        if start >= last_end:
            spans.append((start, end, kind))
            last_end = end
    return spans


def redact(text: str) -> str:
    """Replace every detected PII span with a `[REDACTED_<TYPE>]` placeholder."""
    for start, end, kind in sorted(detect(text), key=lambda s: s[0], reverse=True):
        text = text[:start] + f"[REDACTED_{kind}]" + text[end:]
    return text


def _redact_in_place(value: object) -> None:
    """Recurse through a hook ctx value, redacting free-text IN PLACE.

    Why in place and not a rebuilt copy: every real `hooks.run(seam, ctx)` call site in the fixed
    pipeline/agent code (`pipeline.py`, `agent/agent.py`) discards the return value -- e.g.
    `hooks.run(hooks.AFTER_OCR, {"chunks": text})` never reassigns `text`. The `{"chunks": text}`
    dict literal just wraps a reference to the SAME list `text` already points to, so mutating the
    Chunk objects inside that list (or the dict's own string values) is the only way redaction can
    actually take effect on the real path -- returning a new, separately-redacted structure here
    would be silently discarded by the caller.
    """
    if isinstance(value, BaseModel):
        content_fields = _CONTENT_FIELDS.get(type(value), ())
        for f in content_fields:
            setattr(value, f, redact(getattr(value, f)))
        for f in type(value).model_fields:
            if f in content_fields:
                continue
            _redact_in_place(getattr(value, f))
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str):
                value[k] = redact(v)
            else:
                _redact_in_place(v)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str):
                value[i] = redact(v)
            else:
                _redact_in_place(v)
    # Tuples are immutable and contracts.py's only tuple fields (Region.bbox, Citation.span) hold
    # ints, never free text -- nothing to do for them, and no other type carries text to redact.


def register(hooks) -> None:
    """Wire PII redaction into the pipeline.

    AFTER_OCR scrubs {"chunks": list[Chunk]} before indexing. BEFORE_ANSWER scrubs
    {"state": {"query": ..., "obs": [...]}} -- the agent's pre-synthesis query + tool evidence,
    not an Answer object (that only exists at AFTER_ANSWER, which PII is deliberately not wired
    to: redacting the evidence going IN is what keeps PII out of whatever synthesize() produces).
    ON_LOG scrubs whatever dict a future logging integration passes through this seam -- no call
    site fires it yet in the fixed pipeline/agent code, so this just has to be safe on an
    arbitrary dict, not tuned to one exact shape.
    """

    def _scrub(ctx: dict) -> dict:
        _redact_in_place(ctx)
        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)  # scrub logs
