"""Server-level checks: JSON safety and tool registration over the real protocol."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from asclepius import server


def test_json_safe_strips_non_finite_floats():
    payload = {
        "vif": float("inf"),
        "ci": [float("nan"), 0.5],
        "nested": {"p": -float("inf"), "ok": 1.25},
        "np": np.float64("nan"),
        "int": np.int64(7),
        "flag": np.bool_(True),
    }
    safe = server._json_safe(payload)

    assert safe["vif"] is None
    assert safe["ci"] == [None, 0.5]
    assert safe["nested"]["p"] is None
    assert safe["nested"]["ok"] == 1.25
    assert safe["np"] is None
    assert safe["int"] == 7
    assert safe["flag"] is True

    # The whole point: it must survive a strict JSON round trip.
    text = json.dumps(safe, allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text


def test_json_safe_preserves_ordinary_values():
    payload = {"a": 1, "b": "x", "c": [1.5, 2.5], "d": None, "e": True}
    assert server._json_safe(payload) == payload


def test_guard_reports_errors_instead_of_raising():
    def boom():
        raise ValueError("something broke")

    out = server._guard(boom)
    assert "error" in out
    assert "something broke" in out["error"]


@pytest.mark.anyio
async def test_tools_are_registered_and_described():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}

    expected = {
        "whoop_status",
        "whoop_sync",
        "whoop_metrics",
        "whoop_daily",
        "whoop_correlate",
        "whoop_rank_drivers",
        "whoop_regress",
        "whoop_compare_periods",
        "whoop_anomalies",
    }
    assert expected <= names

    # Every tool needs a description, since that is all the model sees.
    for t in tools:
        assert t.description and len(t.description) > 30, t.name


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_empty_cache_returns_actionable_message(tmp_path, monkeypatch):
    """With no data synced, tools must explain what to do rather than crash."""
    import pandas as pd

    monkeypatch.setattr(server, "_frame_cache", (0, pd.DataFrame()))
    monkeypatch.setattr(server, "_frame", lambda force=False: pd.DataFrame())

    out = server.whoop_metrics()
    assert "error" in out
    assert "sync" in out["error"].lower()
