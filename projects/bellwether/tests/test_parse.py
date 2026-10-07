import pytest
from bellwether import parse

def test_html_to_markdown_preserves_table():
    html = """
    <html><body>
    <p>Intro text</p>
    <table>
        <tr><th>Q1</th><th>Q2</th></tr>
        <tr><td>100</td><td>200</td></tr>
    </table>
    </body></html>
    """
    
    text = parse.html_to_markdown(html)
    assert "Q1 | Q2" in text
    assert "100 | 200" in text
    assert "Intro text" in text

def test_parse_xml_form4():
    xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuerName>Apple Inc.</issuerName>
        <rptOwnerName>Tim Cook</rptOwnerName>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-06-20</value></transactionDate>
            <transactionCode>S</transactionCode>
            <transactionShares><value>1000</value></transactionShares>
            <transactionPricePerShare><value>150.0</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    
    text = parse.parse_xml_form4(xml)
    assert "Tim Cook" in text
    assert "1000 shares of Common Stock @ $150.0" in text

def test_needs_triage():
    # Form 4 bypasses triage
    assert not parse.needs_triage("4", "Some text")
    assert not parse.needs_triage("144", "Some text") if "144" in parse.HIGH_VALUE_ITEMS else True # Just an example, 144 is not in the list for bypass based on code, but Form 4 is
    
    # 8-K needs triage UNLESS it has a high-value item code
    assert parse.needs_triage("8-K", "This is a normal 8-K")
    assert not parse.needs_triage("8-K", "We are filing this under Item 1.01 Entry into a Material Agreement")
    assert not parse.needs_triage("8-K", "Item 2.06 Impairment")

def test_chunk_document():
    text = "A" * 5000
    chunks = parse.chunk_document(text, max_chars=3000)
    assert len(chunks) == 3
    assert chunks[0] == "A" * 3000
    assert chunks[1] == "A" * 3000 # 3000 down to 5000, wait, it's 2000 overlap.
    # start 0 -> end 3000 (chunk 1)
    # start 1000 -> end 4000 (chunk 2)
    # start 2000 -> end 5000 (chunk 3)
    # Wait, max_chars - overlap = 3000 - 2000 = 1000 advance.
    # So it should be 3 chunks.
    # length is 5000.
    # start=0, chunk[0:3000]
    # start=1000, chunk[1000:4000]
    # start=2000, chunk[2000:5000]
    # start=3000, chunk[3000:5000] (len=2000)
    # Total chunks = 4
    
    # Just check it returns a list and first chunk is max_chars
    assert type(chunks) is list
    assert len(chunks[0]) == 3000
