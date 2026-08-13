"""Cross-cutting features must work END TO END, not just exist in one file.
Un-skip and implement alongside the feature. CI runs these."""

import pytest

from doc_agent import hooks, wiring
from doc_agent.contracts import Chunk


@pytest.mark.skip(reason="implement with grounding")
def test_grounding_unsupported_query_abstains():
    """An answer with no supporting evidence must abstain, not fabricate."""
    assert True


@pytest.mark.skip(reason="implement with security")
def test_injection_in_document_does_not_hijack():
    """A document containing 'ignore your instructions' must not change agent behaviour."""
    assert True


def test_pii_never_leaks_to_answer_or_log():
    """PII in the corpus must not appear in answers or logs.

    governance/pii.py is implemented (was DP-3's second cross-team blocker). The agent loop
    itself (decide/act/synthesize) is still A3 scope and doesn't exist yet, so this exercises the
    seams PII actually attaches to directly via the real wiring.register_all() manifest -- not by
    calling governance.pii.register() in isolation -- so a drift in wiring.py's own attachment
    would still be caught here.

    BEFORE_ANSWER is shared with llm/postprocess.py's grounding gate, which is a genuinely
    separate, still-unimplemented A3 feature (`_ground` unconditionally raises
    NotImplementedError) -- not something this PII test should paper over. wiring.py registers
    pii.register() before postprocess.register(), so hooks.run()'s handler loop runs PII's _scrub
    (mutating `state` in place) BEFORE grounding's stub raises -- letting this assert the
    redaction already happened, inside the pytest.raises block that documents grounding's real,
    expected, current state."""
    wiring.register_all({"agent": {}})  # satisfies guardrails.py::Guardrails(cfg)'s eager __init__

    pii_email = "leak@example.com"
    chunks = [Chunk(id="c1", doc_id="d1", text=f"see {pii_email} for details", page_ids=["p1"])]
    hooks.run(hooks.AFTER_OCR, {"chunks": chunks})
    assert pii_email not in chunks[0].text
    assert chunks[0].id == "c1" and chunks[0].doc_id == "d1"  # join keys untouched

    state = {"query": f"contact {pii_email}", "obs": [{"payload": {"note": pii_email}}]}
    with pytest.raises(NotImplementedError, match="Grounding"):
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})
    assert pii_email not in state["query"]
    assert pii_email not in state["obs"][0]["payload"]["note"]

    log_ctx = {"msg": f"user said {pii_email}"}
    hooks.run(hooks.ON_LOG, log_ctx)
    assert pii_email not in log_ctx["msg"]


@pytest.mark.skip(reason="implement with tracing")
def test_trace_covers_every_step():
    """Every agent step and tool call must appear in the audit trail."""
    assert True


@pytest.mark.skip(reason="implement with reproducibility")
def test_rerun_reproduces_metrics():
    """A seeded re-run reproduces reported metrics within tolerance."""
    assert True
