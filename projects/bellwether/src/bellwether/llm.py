"""Phase 3 — Provider-agnostic LLM client.

Wraps Google Gemini and Anthropic Claude behind a single `complete` interface.
Supports structured JSON output and retries.
"""

from __future__ import annotations

import json
import time
from typing import Type, TypeVar, Any

from pydantic import BaseModel, Field

from . import config
from .logging_setup import get_logger

log = get_logger("bellwether.llm")

T = TypeVar("T", bound=BaseModel)

class SignalSchema(BaseModel):
    catalyst_type: str = Field(description="new_contract | insider_buy_cluster | guidance_change | litigation | activist_13d | mgmt_change | regulatory_decision | impairment | other")
    direction: str = Field(description="bullish | bearish | neutral")
    materiality: int = Field(description="0-10, how market-moving")
    confidence: float = Field(description="0-1, model's own certainty")
    summary: str = Field(description="≤2 sentences, plain English")
    evidence_quote: str = Field(description="the exact sentence from the filing that drives this")

class TriageSchema(BaseModel):
    is_material: bool = Field(description="Could this filing plausibly be material?")
    reason: str = Field(description="One-line why")

def _complete_gemini(prompt: str, schema: Type[T] | None = None, model: str | None = None) -> Any:
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=config.require("GEMINI_API_KEY"))
    model = model or "gemini-2.5-flash"
    
    config_dict = {}
    if schema:
        config_dict["response_mime_type"] = "application/json"
        config_dict["response_schema"] = schema
        
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_dict) if config_dict else None
    )
    
    if schema:
        return schema.model_validate_json(response.text)
    return response.text

def _complete_anthropic(prompt: str, schema: Type[T] | None = None, model: str | None = None) -> Any:
    import anthropic
    
    client = anthropic.Anthropic(api_key=config.require("ANTHROPIC_API_KEY"))
    model = model or "claude-3-5-sonnet-20241022"
    
    if schema:
        # Anthropic Tool Use for structured output
        tool = {
            "name": "extract_signal",
            "description": "Extract structured data from the document",
            "input_schema": schema.model_json_schema()
        }
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[tool],
            tool_choice={"type": "tool", "name": "extract_signal"},
            messages=[{"role": "user", "content": prompt}]
        )
        for block in response.content:
            if block.type == "tool_use":
                return schema(**block.input)
        raise ValueError("Anthropic did not return tool use block")
    else:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

def _complete_mock(prompt: str, schema: Type[T] | None = None) -> Any:
    import hashlib
    import re
    
    h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)
    
    if schema is TriageSchema:
        is_material = (h % 100) < 35  # 35% pass rate
        reason = "Detected potential corporate catalyst in text" if is_material else "Routine boilerplate or non-material disclosure"
        return TriageSchema(is_material=is_material, reason=reason)
        
    if schema is SignalSchema:
        catalysts = [
            "new_contract", "insider_buy_cluster", "guidance_change", 
            "litigation", "activist_13d", "mgmt_change", 
            "regulatory_decision", "impairment", "other"
        ]
        catalyst_type = catalysts[h % len(catalysts)]
        
        directions = ["bullish", "bearish", "neutral"]
        direction = directions[(h >> 4) % len(directions)]
        
        materiality = 1 + ((h >> 8) % 9)  # 1 to 9
        confidence = round(0.7 + ((h >> 12) % 31) / 100.0, 2)
        
        # Extract a real sentence from the prompt
        text_part = prompt
        for marker in ["Text:\n", "Text excerpt:\n", "Text:\r\n", "Text excerpt:\r\n"]:
            if marker in prompt:
                text_part = prompt.split(marker, 1)[1]
                break
        
        sentences = [s.strip() for s in re.split(r"[.!?\n]", text_part) if len(s.strip()) > 30]
        if sentences:
            evidence_quote = sentences[(h >> 16) % len(sentences)]
            if len(evidence_quote) > 150:
                evidence_quote = evidence_quote[:147] + "..."
        else:
            evidence_quote = "The Company entered into a material agreement on the date specified herein."
            
        summary_templates = {
            "new_contract": "The company entered into a new material contract representing significant future revenue potential.",
            "insider_buy_cluster": "A cluster of open-market purchases by key executives indicates strong alignment and conviction in future performance.",
            "guidance_change": "Management updated financial guidance for the upcoming quarters, reflecting revised expectations.",
            "litigation": "The company disclosed a new legal proceeding or regulatory inquiry that could result in future liability.",
            "activist_13d": "An activist shareholder filed a Schedule 13D, disclosing a significant stake and intent to engage with management.",
            "mgmt_change": "The company announced a transition in key executive or board leadership, appointing new management.",
            "regulatory_decision": "A regulatory agency issued a decision or approval affecting the company's core products or operations.",
            "impairment": "The company recorded a material asset impairment or write-down, impacting quarterly net income.",
            "other": "The filing disclosed a material corporate development details in the transaction reports."
        }
        summary = summary_templates.get(catalyst_type, "The filing disclosed a material corporate development.")
        
        return SignalSchema(
            catalyst_type=catalyst_type,
            direction=direction,
            materiality=materiality,
            confidence=confidence,
            summary=summary,
            evidence_quote=evidence_quote
        )
        
    return "Mock text response based on prompt hash: " + str(h)

def _schema_template(schema: Type[BaseModel]) -> str:
    if schema.__name__ == "TriageSchema":
        return """{
  "is_material": true | false,
  "reason": "a one-line explanation of why"
}"""
    elif schema.__name__ == "SignalSchema":
        return """{
  "catalyst_type": "new_contract" | "insider_buy_cluster" | "guidance_change" | "litigation" | "activist_13d" | "mgmt_change" | "regulatory_decision" | "impairment" | "other",
  "direction": "bullish" | "bearish" | "neutral",
  "materiality": 0-10,
  "confidence": 0.0-1.0,
  "summary": "≤2 sentences explaining the event",
  "evidence_quote": "copy-pasted exact quote from filing"
}"""
    import json
    return json.dumps(schema.model_json_schema(), indent=2)

def _complete_ollama(prompt: str, schema: Type[T] | None = None, model: str | None = None) -> Any:
    import requests
    
    model = model or "llama3.2:3b"
    url = "http://localhost:11434/api/chat"
    
    if schema:
        template = _schema_template(schema)
        prompt = f"{prompt}\n\nYou must return a JSON object strictly matching this template format:\n{template}\n\nDo not nest it or return any other text, only the valid JSON response."
        
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    if schema:
        payload["format"] = "json"
        
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    
    content = resp.json()["message"]["content"]
    
    if schema:
        clean_content = content.strip()
        if clean_content.startswith("```"):
            lines = clean_content.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                clean_content = "\n".join(lines[1:-1])
        return schema.model_validate_json(clean_content)
        
    return content

def complete(prompt: str, schema: Type[T] | None = None, provider: str | None = None, model: str | None = None, retries: int = 5) -> Any:
    """Send prompt to the specified provider and optionally enforce a JSON schema."""
    if config.USE_MOCK_LLM:
        return _complete_mock(prompt, schema)
        
    provider = provider or config.require("DEEP_READ_PROVIDER")
    
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            if provider == "gemini":
                res = _complete_gemini(prompt, schema, model=model)
            elif provider == "anthropic":
                res = _complete_anthropic(prompt, schema, model=model)
            elif provider == "ollama":
                res = _complete_ollama(prompt, schema, model=model)
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
            log.info("LLM %s complete in %.1fs", provider, time.time() - t0)
            return res
        except Exception as e:
            if attempt < retries:
                err_str = str(e).upper()
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str
                if is_rate_limit:
                    sleep_time = 30 * (attempt + 1)
                    log.warning("LLM %s rate limited (%s). Sleeping %ds before retry...", provider, e, sleep_time)
                    time.sleep(sleep_time)
                else:
                    sleep_time = 2 ** attempt
                    log.warning("LLM %s failed (%s), retrying in %ds...", provider, e, sleep_time)
                    time.sleep(sleep_time)
            else:
                log.error("LLM %s failed all retries: %s", provider, e)
                raise
