"""Phase 3 — Orchestration Pipeline.

Wires parsing, triage, and deep-read models together.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from . import db
from . import llm
from . import parse
from .logging_setup import get_logger

log = get_logger("bellwether.pipeline")

TRIAGE_PROMPT = """
You are a highly sensitive financial triage system. Your job is to read this excerpt from a SEC filing and determine if it could plausibly contain a market-moving catalyst.

We are looking for any mention of the following triggers (or synonymous events):
- Litigation or regulatory action
- Termination, resignation, or change of key management/board members
- Going-concern notices or bankruptcy
- Impairment of assets or write-downs
- Dilution, unregistered equity, or stock offerings
- New material contracts, mergers, or acquisitions
- Changes in financial guidance
- Dry holes, plugged wells, or severe operational failures

Analyze the text and output JSON.
is_material: true if there is ANY chance this is a catalyst.
reason: a one-line explanation of why.

Text excerpt:
{text}
"""

DEEP_READ_PROMPT = """
You are an expert financial analyst. Read the following text from an SEC filing and extract the core market-moving catalyst.

Follow the schema rigidly.
- catalyst_type: Must be one of the enum values (new_contract | insider_buy_cluster | guidance_change | litigation | activist_13d | mgmt_change | regulatory_decision | impairment | other).
- direction: bullish | bearish | neutral
- materiality: 0 to 10 (10 being bankruptcy/buyout, 5 being a standard new contract, 1 being boilerplate).
- confidence: 0.0 to 1.0 (how certain you are that you're interpreting the event correctly).
- summary: Less than 2 sentences explaining the event.
- evidence_quote: YOU MUST COPY AND PASTE THE EXACT SENTENCE FROM THE TEXT that proves your conclusion. No paraphrasing.

If the text contains NO material event, do not return a JSON object (or if forced, return materiality 0 with type "other").

Text:
{text}
"""

def run_triage(text: str) -> llm.TriageSchema:
    """Run the low-cost triage model."""
    prompt = TRIAGE_PROMPT.format(text=text)
    return llm.complete(
        prompt, 
        schema=llm.TriageSchema, 
        provider=config.require("TRIAGE_PROVIDER"),
        model=config.require("TRIAGE_MODEL")
    )

def run_deep_read(text: str) -> llm.SignalSchema | None:
    """Run the configured deep-read model."""
    prompt = DEEP_READ_PROMPT.format(text=text)
    try:
        signal = llm.complete(
            prompt, 
            schema=llm.SignalSchema, 
            provider=config.require("DEEP_READ_PROVIDER"),
            model=config.require("DEEP_READ_MODEL")
        )
        if signal.materiality > 0:
            return signal
    except Exception as e:
        log.warning("Deep read failed to extract valid signal schema: %s", e)
    return None

def process_filing(conn, filing_row) -> list[int]:
    """End-to-end processing of a single filing. Returns list of inserted signal IDs."""
    filing_id = filing_row["id"]
    raw_path = Path(filing_row["raw_path"])
    form_type = filing_row["form_type"]
    
    if not raw_path.exists():
        log.error("Filing %s missing raw path: %s", filing_id, raw_path)
        return []
        
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    
    # 1. Parse
    if form_type in ("4", "4/A"):
        clean_text = parse.parse_xml_form4(raw_text)
    else:
        clean_text = parse.html_to_markdown(raw_text)
        
    # 2. Triage Bypass Check
    bypass = not parse.needs_triage(form_type, clean_text)
    
    # 3. Chunk
    chunks = parse.chunk_document(clean_text)
    
    signal_ids: list[int] = []
    
    for i, chunk in enumerate(chunks):
        log.debug("Processing chunk %d/%d for filing %s", i+1, len(chunks), filing_id)
        
        # 4. Triage
        if not bypass:
            triage_res = run_triage(chunk)
            if not triage_res.is_material:
                continue
            log.info("Filing %s chunk %d passed triage: %s", filing_id, i+1, triage_res.reason)
            
        # 5. Deep Read
        signal = run_deep_read(chunk)
        if signal:
            log.info("Extracted signal: %s (%s)", signal.summary, signal.catalyst_type)
            signal_id = db.insert_signal(conn, filing_id, signal, config.require("DEEP_READ_PROVIDER"))
            signal_ids.append(signal_id)
            
    return signal_ids
