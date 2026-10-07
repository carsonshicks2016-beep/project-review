import pytest
from unittest.mock import patch
from bellwether import pipeline, llm

@patch("bellwether.pipeline.config.require")
@patch("bellwether.pipeline.llm.complete")
def test_run_triage_passes_material(mock_complete, mock_require):
    mock_require.return_value = "gemini"
    # Mock the LLM returning a positive material triage
    mock_complete.return_value = llm.TriageSchema(
        is_material=True,
        reason="Mentions a dry hole"
    )
    
    res = pipeline.run_triage("We plugged and abandoned the well due to finding a dry hole.")
    assert res.is_material is True
    assert "dry hole" in res.reason
    mock_complete.assert_called_once()
    args, kwargs = mock_complete.call_args
    assert "dry hole" in args[0]
    assert kwargs["provider"] == "gemini"

@patch("bellwether.pipeline.config.require")
@patch("bellwether.pipeline.llm.complete")
def test_run_deep_read_valid_signal(mock_complete, mock_require):
    mock_require.return_value = "gemini"
    
    mock_complete.return_value = llm.SignalSchema(
        catalyst_type="guidance_change",
        direction="bullish",
        materiality=8,
        confidence=0.9,
        summary="Raised production guidance.",
        evidence_quote="We are raising our 2026 production guidance."
    )
    
    res = pipeline.run_deep_read("We are raising our 2026 production guidance.")
    assert res is not None
    assert res.catalyst_type == "guidance_change"
    assert res.materiality == 8

@patch("bellwether.pipeline.config.require")
@patch("bellwether.pipeline.llm.complete")
def test_run_deep_read_no_signal(mock_complete, mock_require):
    mock_require.return_value = "gemini"
    
    # Simulate the model returning materiality=0 for boilerplate
    mock_complete.return_value = llm.SignalSchema(
        catalyst_type="other",
        direction="neutral",
        materiality=0,
        confidence=1.0,
        summary="Nothing to see here.",
        evidence_quote=""
    )
    
    res = pipeline.run_deep_read("This is standard boilerplate.")
    assert res is None  # Should return None if materiality is 0
