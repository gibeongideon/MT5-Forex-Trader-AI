"""Deterministic-gate tests for the V5 agent risk overlay (no real API calls)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.v5.agents import (OverlayRunner, build_evidence_pack, clamp_scales,
                           parse_json_block)

KEPT = ["GOLD", "EURUSD"]


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock(text)]
        self.usage = None


class FakeClient:
    """Yields queued responses; counts calls; can raise."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return FakeResponse(self.responses.pop(0))


def _evidence():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        {a: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))) for a in KEPT},
        index=idx)
    return build_evidence_pack(
        date="2026-07-05",
        targets={"GOLD": 0.25, "EURUSD": -0.1},
        prev_positions={"GOLD": 0.2, "EURUSD": 0.0},
        actions={"GOLD": "ADD", "EURUSD": "OPEN"},
        close=close, kept=KEPT)


GOOD_CHAIN = [
    '{"summary": "ok", "notable_risks": [], "notable_supports": []}',
    '{"thesis": "up", "conviction": "high"}',
    '{"thesis": "down", "biggest_risk": "vol", "conviction": "low"}',
    '{"scales": {"GOLD": 0.8, "EURUSD": 1.0}, "rationale": "minor vol concern"}',
]


def test_clamp_scales_gates():
    raw = {"scales": {"GOLD": 1.7, "EURUSD": -0.2}}
    assert clamp_scales(raw, KEPT) == {"GOLD": 1.0, "EURUSD": 0.0}
    assert clamp_scales({"scales": {"GOLD": "abc"}}, KEPT) == {"GOLD": 1.0, "EURUSD": 1.0}
    assert clamp_scales({}, KEPT) == {"GOLD": 1.0, "EURUSD": 1.0}
    assert clamp_scales("not a dict", KEPT) == {"GOLD": 1.0, "EURUSD": 1.0}
    assert clamp_scales({"scales": {"GOLD": float("nan")}}, KEPT)["GOLD"] == 1.0


def test_parse_json_block_variants():
    assert parse_json_block('{"a": 1}') == {"a": 1}
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block('Sure! Here it is:\n{"scales": {"GOLD": 0.5}} hope it helps') == \
        {"scales": {"GOLD": 0.5}}
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_json_block('{"truncated": ')


def test_full_chain_applies_pm_scales(tmp_path):
    client = FakeClient(list(GOOD_CHAIN))
    scales, meta = OverlayRunner(tmp_path, client=client).run(_evidence())
    assert client.calls == 4
    assert scales == {"GOLD": 0.8, "EURUSD": 1.0}
    assert meta["reason"] == "agent_decision"
    decisions = (tmp_path / "agent_decisions" / "2026-07-05.jsonl").read_text()
    assert decisions.count("\n") == 5  # 4 agents + final record


def test_api_error_fails_open(tmp_path):
    client = FakeClient(error=RuntimeError("api down"))
    scales, meta = OverlayRunner(tmp_path, client=client).run(_evidence())
    assert scales == {"GOLD": 1.0, "EURUSD": 1.0}
    assert meta["reason"] == "fail_open"


def test_garbage_response_fails_open(tmp_path):
    client = FakeClient(["this is not json at all"])
    scales, meta = OverlayRunner(tmp_path, client=client).run(_evidence())
    assert scales == {"GOLD": 1.0, "EURUSD": 1.0}
    assert meta["reason"] == "fail_open"


def test_budget_exceeded_fails_open(tmp_path):
    client = FakeClient(list(GOOD_CHAIN))
    scales, meta = OverlayRunner(tmp_path, client=client, call_budget=2).run(_evidence())
    assert client.calls == 2
    assert scales == {"GOLD": 1.0, "EURUSD": 1.0}
    assert meta["reason"] == "fail_open"


def test_cache_hit_makes_zero_client_calls(tmp_path):
    ev = _evidence()
    c1 = FakeClient(list(GOOD_CHAIN))
    s1, _ = OverlayRunner(tmp_path, client=c1).run(ev)
    assert c1.calls == 4
    c2 = FakeClient(list(GOOD_CHAIN))
    s2, meta2 = OverlayRunner(tmp_path, client=c2).run(ev)
    assert c2.calls == 0
    assert s1 == s2
    assert meta2["reason"] == "agent_decision"


def test_evidence_pack_is_json_serializable():
    json.dumps(_evidence())
