"""Phase 3 — Parsing and Triage logic.

Structure-preserving HTML to Markdown (so LLMs can read financials),
XML parsing for Form 4s, and deterministic triage routing.
"""

from __future__ import annotations

import re
from pathlib import Path
from bs4 import BeautifulSoup

from .logging_setup import get_logger

log = get_logger("bellwether.parse")

HIGH_VALUE_ITEMS = {"1.01", "1.03", "2.06", "3.02", "5.02"}

def html_to_markdown(html: str) -> str:
    """Basic structure-preserving HTML to Markdown.
    
    Converts tables to text grids so the LLM doesn't see number soup.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Process tables first
    for table in soup.find_all("table"):
        markdown_table = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True).replace("\n", " ") for cell in row.find_all(["th", "td"])]
            if any(cells):  # Skip empty rows
                markdown_table.append(" | ".join(cells))
        if markdown_table:
            # Replace the table tag with a text block
            table.replace_with("\n" + "\n".join(markdown_table) + "\n")
            
    # Now extract the remaining text, preserving structure where possible
    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def parse_xml_form4(xml: str) -> str:
    """Extract key transaction details from Form 4 XML."""
    soup = BeautifulSoup(xml, "xml")
    lines = []
    lines.append("FORM 4 / INSIDER TRADING")
    
    issuer = soup.find("issuerName")
    if issuer:
        lines.append(f"Issuer: {issuer.text}")
        
    reporting = soup.find("rptOwnerName")
    if reporting:
        lines.append(f"Reporting Owner: {reporting.text}")
        
    for tx in soup.find_all("nonDerivativeTransaction"):
        title = tx.find("securityTitle")
        date = tx.find("transactionDate")
        code = tx.find("transactionCode")
        shares = tx.find("transactionShares")
        price = tx.find("transactionPricePerShare")
        acq_disp = tx.find("transactionAcquiredDisposedCode")
        
        if all([title, date, shares, price, acq_disp]):
            lines.append(
                f"Transaction: {date.text.strip()} | {acq_disp.text.strip()} "
                f"{shares.text.strip()} shares of {title.text.strip()} "
                f"@ ${price.text.strip()}"
            )
            
    return "\n".join(lines)

def needs_triage(form_type: str, document_text: str) -> bool:
    """Deterministic routing: False if it goes straight to deep-read."""
    if form_type in ("4", "4/A", "SCHEDULE 13D", "SCHEDULE 13D/A"):
        return False  # Form 4 and 13D bypass triage
        
    if form_type in ("8-K", "8-K/A"):
        # Check for high-value items in the text
        # Usually formatted like "Item 1.01" or "Item 1.01."
        for item in HIGH_VALUE_ITEMS:
            # simple regex to find "Item X.XX"
            if re.search(rf"Item\s+{re.escape(item)}", document_text, re.IGNORECASE):
                log.info("Bypassing triage for high-value item: %s", item)
                return False
                
    return True

def chunk_document(text: str, max_chars: int = 100000) -> list[str]:
    """Split oversized filings into chunks with overlap."""
    if len(text) <= max_chars:
        return [text]
        
    overlap = 2000
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += max_chars - overlap
    return chunks
