"""V5 AI agent risk overlay — Lumibot-style agent team, ported minimally.

A researcher -> bull -> bear -> PM chain (sequential summary-chaining, the
Lumibot `AgentManager` pattern) reviews the champion's daily target positions
and may SCALE each position in [0, 1]. It can never flip a direction, add a
trade, or exceed the champion's size — the analysts' response schemas contain
no trading fields at all, and the PM's scales are clamped in code.

Scientific status: LLM decisions cannot be backtested without hindsight
leakage (the model's pretraining knows market history), so this layer makes
NO backtest Sharpe claims. It is evaluated forward only, via the two-run-id
paper A/B in scripts/v5_basket_runner.py (--overlay).

Deterministic guarantees (code, not prompts):
  - scales clamped to [0, 1]; missing/invalid alias -> 1.0
  - hard model-call budget per day (default 4)
  - ANY exception (missing key, API error, parse failure, budget) ->
    all scales 1.0 with reason "fail_open": the champion is never blocked
  - sha256 response cache -> same-day re-runs are idempotent and free
  - every call journaled to agent_decisions/YYYY-MM-DD.jsonl (no API keys)
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
CALL_BUDGET = 4

_RESULT_RULES = (
    "Reply with a single JSON object only — no markdown fences, no prose "
    "before or after."
)

AGENTS = [
    dict(
        name="researcher",
        system=(
            "You are the research analyst on a systematic trading desk. The "
            "desk runs a validated trend-following portfolio; your job is to "
            "summarize the evidence pack objectively — trend alignment, "
            "volatility regime, position changes — without recommending "
            "trades. " + _RESULT_RULES
        ),
        schema='{"summary": "<120 words", "notable_risks": ["..."], "notable_supports": ["..."]}',
    ),
    dict(
        name="bull",
        system=(
            "You are the bull-case analyst. Given the research summary and "
            "evidence pack, argue the strongest case FOR carrying the "
            "champion's target positions at full size today. You have no "
            "trading authority. " + _RESULT_RULES
        ),
        schema='{"thesis": "<100 words", "conviction": "high"|"medium"|"low"}',
    ),
    dict(
        name="bear",
        system=(
            "You are the bear-case/risk analyst. Given the research summary, "
            "the bull thesis, and the evidence pack, challenge the trade: "
            "where is the champion most exposed today? You have no trading "
            "authority. " + _RESULT_RULES
        ),
        schema='{"thesis": "<100 words", "biggest_risk": "...", "conviction": "high"|"medium"|"low"}',
    ),
    dict(
        name="pm",
        system=(
            "You are the portfolio manager. You may only DE-RISK: for each "
            "instrument alias, output a scale between 0.0 (veto today's "
            "position) and 1.0 (carry at full champion size). You cannot "
            "flip direction or add exposure. Default to 1.0 unless the bear "
            "case identifies a concrete, instrument-specific risk; scale "
            "below 0.5 only for severe, well-evidenced concerns. "
            + _RESULT_RULES
        ),
        schema='{"scales": {"<alias>": 0.0-1.0, ...}, "rationale": "<80 words"}',
    ),
]


def parse_json_block(text: str) -> dict:
    """Extract a JSON object from model text, tolerating fences and nesting."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])  # raises on failure -> fail_open
    raise ValueError("no JSON object found in model response")


def build_evidence_pack(*, date: str, targets: dict, prev_positions: dict,
                        actions: dict, close: pd.DataFrame, kept: list) -> dict:
    """JSON-serializable market/portfolio evidence. All stats past-only."""
    returns = close.pct_change(fill_method=None)
    sigma = returns.shift(1).ewm(halflife=42, min_periods=20).std()
    pct = sigma.rolling(252, min_periods=60).rank(pct=True)
    sma200 = close.shift(1).rolling(200, min_periods=200).mean()

    instruments = {}
    for a in kept:
        c = close[a].dropna()
        instruments[a] = {
            "target_position": targets.get(a, 0.0),
            "previous_position": prev_positions.get(a, 0.0),
            "action": actions.get(a, "hold"),
            "trend_vs_200d_sma": ("up" if close[a].iloc[-1] > sma200[a].iloc[-1]
                                  else "down") if np.isfinite(sma200[a].iloc[-1]) else "warmup",
            "vol_percentile_1y": round(float(pct[a].iloc[-1]), 2)
            if np.isfinite(pct[a].iloc[-1]) else None,
            "ret_5d_pct": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2)
            if len(c) > 6 else None,
            "ret_21d_pct": round(float(c.iloc[-1] / c.iloc[-22] - 1) * 100, 2)
            if len(c) > 22 else None,
            "ret_63d_pct": round(float(c.iloc[-1] / c.iloc[-64] - 1) * 100, 2)
            if len(c) > 64 else None,
        }
    return {
        "date": date,
        "weekday": pd.Timestamp(date).day_name(),
        "strategy": "vol-targeted trend portfolio (validated; Sharpe claims "
                    "come from the backtest, not from you)",
        "instruments": instruments,
        "gross_target": round(sum(abs(v) for v in targets.values()), 3),
    }


def clamp_scales(raw: dict, kept: list) -> dict:
    """Deterministic gate: [0,1] clamp; anything missing/invalid -> 1.0."""
    out = {}
    scales = raw.get("scales", {}) if isinstance(raw, dict) else {}
    for a in kept:
        try:
            v = float(scales[a])
            out[a] = min(1.0, max(0.0, v)) if np.isfinite(v) else 1.0
        except (KeyError, TypeError, ValueError):
            out[a] = 1.0
    return out


class OverlayRunner:
    """Runs the 4-agent chain with cache, budget, journaling, and fail-open."""

    def __init__(self, run_dir: str | Path, model: str = MODEL,
                 call_budget: int = CALL_BUDGET, client=None):
        self.run_dir = Path(run_dir)
        self.model = model
        self.call_budget = call_budget
        self._client = client
        self._calls = 0

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.run_dir / "agent_cache" / f"{key}.json"

    def _journal(self, record: dict) -> None:
        d = self.run_dir / "agent_decisions"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{record.get('date', 'unknown')}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _call_agent(self, agent: dict, user_content: str, date: str) -> dict:
        key = hashlib.sha256(
            f"{self.model}|{agent['system']}|{user_content}".encode()
        ).hexdigest()
        cached = self._cache_path(key)
        if cached.exists():
            payload = json.loads(cached.read_text())
            self._journal({"date": date, "agent": agent["name"], "cache_hit": True,
                           "prompt_sha256": key, "parsed": payload["parsed"]})
            return payload["parsed"]

        if self._calls >= self.call_budget:
            raise RuntimeError(f"agent model-call budget ({self.call_budget}) exceeded")
        self._calls += 1

        t0 = time.time()
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=agent["system"],
            messages=[{"role": "user", "content": user_content}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        parsed = parse_json_block(text)
        usage = getattr(resp, "usage", None)
        record = {
            "date": date, "agent": agent["name"], "cache_hit": False,
            "prompt_sha256": key, "raw_text": text, "parsed": parsed,
            "latency_s": round(time.time() - t0, 2),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        self._journal(record)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"parsed": parsed, "raw_text": text}))
        return parsed

    def run(self, evidence: dict) -> tuple[dict, dict]:
        """Returns (scales_per_alias, meta). NEVER raises — fail-open."""
        kept = list(evidence["instruments"])
        date = evidence.get("date", "unknown")
        self._calls = 0
        try:
            pack = json.dumps(evidence, indent=2, sort_keys=True, default=str)
            context = {}
            researcher, bull, bear, pm = AGENTS
            context["research"] = self._call_agent(
                researcher,
                f"Evidence pack:\n{pack}\n\nRespond as: {researcher['schema']}",
                date)
            context["bull"] = self._call_agent(
                bull,
                f"Evidence pack:\n{pack}\n\nResearch summary:\n"
                f"{json.dumps(context['research'])}\n\nRespond as: {bull['schema']}",
                date)
            context["bear"] = self._call_agent(
                bear,
                f"Evidence pack:\n{pack}\n\nResearch summary:\n"
                f"{json.dumps(context['research'])}\n\nBull thesis:\n"
                f"{json.dumps(context['bull'])}\n\nRespond as: {bear['schema']}",
                date)
            pm_raw = self._call_agent(
                pm,
                f"Evidence pack:\n{pack}\n\nResearch summary:\n"
                f"{json.dumps(context['research'])}\n\nBull thesis:\n"
                f"{json.dumps(context['bull'])}\n\nBear thesis:\n"
                f"{json.dumps(context['bear'])}\n\n"
                f"Instrument aliases requiring a scale: {kept}\n"
                f"Respond as: {pm['schema']}",
                date)
            scales = clamp_scales(pm_raw, kept)
            meta = {"reason": "agent_decision",
                    "rationale": pm_raw.get("rationale", ""),
                    "model_calls": self._calls}
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract
            scales = {a: 1.0 for a in kept}
            meta = {"reason": "fail_open", "error": f"{type(exc).__name__}: {exc}",
                    "model_calls": self._calls}
            print(f"  ! agent overlay FAIL-OPEN (champion unchanged): {meta['error']}")
        self._journal({"date": date, "agent": "_final", **meta, "scales": scales})
        return scales, meta


def run_overlay(evidence: dict, run_dir: str | Path, model: str = MODEL,
                client=None) -> tuple[dict, dict]:
    """Module-level convenience wrapper used by the basket runner."""
    return OverlayRunner(run_dir, model=model, client=client).run(evidence)
