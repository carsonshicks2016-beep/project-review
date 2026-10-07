"""
Vault Scribe — Web Scraper Pipeline
Fetches and extracts clean text and metadata from URLs.
Useful for piping research articles directly into the Intake Analyzer.
"""

import re
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class ScrapeResult:
    title: str
    content: str
    source_url: str
    error: str | None = None


def scrape_url(url: str) -> ScrapeResult:
    """
    Fetch a URL and extract the main title and body text.
    Handles general web pages.
    """
    if not url.startswith("http"):
        url = "https://" + url

    try:
        # We use a standard User-Agent to avoid simple bot blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Follow redirects, wait up to 10 seconds
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text().strip()

        # Remove garbage tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Try to find the main article container
        main_content = soup.find("main") or soup.find("article") or soup.find("div", role="main")
        
        if not main_content:
            # Fallback to the body
            main_content = soup.body

        if not main_content:
            return ScrapeResult(title=title, content="Could not extract text.", source_url=url)

        # Extract text from paragraphs and headers
        text_blocks = []
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 10:  # ignore tiny UI fragments
                # If it's a header, maybe add some markdown prefix (simplistic)
                if element.name.startswith("h"):
                    level = element.name[1]
                    text = f"{'#' * int(level)} {text}"
                elif element.name == "li":
                    text = f"- {text}"
                
                text_blocks.append(text)

        clean_text = "\n\n".join(text_blocks)

        # If it's too short, maybe they have everything in divs (bad semantic HTML)
        if len(clean_text) < 500:
            # aggressive fallback: just get all text
            raw = main_content.get_text(separator="\n", strip=True)
            # clean up multiple blank lines
            clean_text = re.sub(r'\n{3,}', '\n\n', raw)

        # Append source URL at the top
        final_content = f"Source: {url}\n\n{clean_text}"

        return ScrapeResult(title=title, content=final_content, source_url=url)

    except Exception as e:
        return ScrapeResult(title="", content="", source_url=url, error=str(e))
