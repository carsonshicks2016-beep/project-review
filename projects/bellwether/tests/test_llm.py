import pytest
from unittest.mock import patch, MagicMock
from bellwether import llm

@patch("bellwether.llm.config.USE_MOCK_LLM", False)
@patch("bellwether.llm.config.require")
@patch("bellwether.llm._complete_gemini")
def test_complete_gemini_routing(mock_gemini, mock_require):
    mock_require.return_value = "gemini"
    mock_gemini.return_value = "test response"
    
    res = llm.complete("Hello", provider="gemini")
    assert res == "test response"
    mock_gemini.assert_called_once_with("Hello", None, model=None)

@patch("bellwether.llm.config.USE_MOCK_LLM", False)
@patch("bellwether.llm.config.require")
@patch("bellwether.llm._complete_anthropic")
def test_complete_anthropic_routing(mock_anthropic, mock_require):
    mock_require.return_value = "anthropic"
    mock_anthropic.return_value = "claude response"
    
    res = llm.complete("Hello", provider="anthropic")
    assert res == "claude response"
    mock_anthropic.assert_called_once_with("Hello", None, model=None)

def test_signal_schema_validation():
    # Valid data
    data = {
        "catalyst_type": "guidance_change",
        "direction": "bullish",
        "materiality": 8,
        "confidence": 0.9,
        "summary": "Increased guidance.",
        "evidence_quote": "We expect higher revenues."
    }
    signal = llm.SignalSchema(**data)
    assert signal.materiality == 8

@patch("bellwether.llm.config.USE_MOCK_LLM", False)
@patch("requests.post")
def test_complete_ollama(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "content": "test ollama response"
        }
    }
    mock_post.return_value = mock_response
    
    res = llm.complete("Hello", provider="ollama", model="llama3.2:3b")
    assert res == "test ollama response"
    mock_post.assert_called_once()
