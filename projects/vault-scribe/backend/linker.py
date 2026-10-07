"""
Vault Scribe — Auto-Linker
Finds and suggests wikilinks by fuzzy-matching content against all note titles
in the vault index. Inserts [[wikilinks]] at first mention and builds a
Related Notes section.
"""

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from backend.config import LINK_MATCH_THRESHOLD
from backend.indexer import get_indexer


@dataclass
class SuggestedLink:
    """A suggested wikilink to insert."""
    target_title: str       # The note title to link to
    match_text: str         # The text in the content that matched
    score: float            # Fuzzy match score (0-100)
    match_type: str         # "exact" or "fuzzy"
    context: str            # Surrounding text for display
    position: int           # Character position in content where match occurs


@dataclass
class LinkingResult:
    """Result of auto-linking analysis."""
    suggested_links: list[SuggestedLink]
    linked_content: str         # Content with wikilinks inserted
    related_notes: list[str]    # Note titles for the Related Notes section
    parent_suggestion: str      # Suggested > Parent: value


def find_links(content: str, category: str = "", exclude_title: str = "") -> LinkingResult:
    """
    Analyze content and find all potential wikilinks.

    Args:
        content: The raw text to analyze.
        category: The classified category (used for parent suggestion).
        exclude_title: Don't suggest linking to this title (the note being created).

    Returns:
        LinkingResult with suggested links and auto-linked content.
    """
    indexer = get_indexer()
    index = indexer.index

    if not index.all_titles:
        return LinkingResult(
            suggested_links=[],
            linked_content=content,
            related_notes=[],
            parent_suggestion="",
        )

    suggestions: list[SuggestedLink] = []
    seen_titles: set[str] = set()

    # ── Strategy 1: Exact title matches ─────────────────────────────────
    # Check if any note title appears verbatim in the content
    for title in index.all_titles:
        if title == exclude_title:
            continue
        if len(title) < 3:  # Skip very short titles to avoid false positives
            continue

        # Case-insensitive search for the exact title
        pattern = re.compile(re.escape(title), re.IGNORECASE)
        match = pattern.search(content)
        if match:
            # Extract context (30 chars before and after)
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 30)
            context = content[start:end].strip()

            suggestions.append(SuggestedLink(
                target_title=title,
                match_text=match.group(0),
                score=100.0,
                match_type="exact",
                context=context,
                position=match.start(),
            ))
            seen_titles.add(title)

    # ── Strategy 2: Fuzzy matching on significant phrases ───────────────
    # Extract significant multi-word phrases from content
    phrases = _extract_significant_phrases(content)

    for phrase in phrases:
        if phrase.lower() in {t.lower() for t in seen_titles}:
            continue

        # Fuzzy match against all titles
        results = process.extract(
            phrase,
            index.all_titles,
            scorer=fuzz.token_sort_ratio,
            limit=3,
        )

        for title, score, _ in results:
            if title == exclude_title:
                continue
            if title in seen_titles:
                continue
            if score >= LINK_MATCH_THRESHOLD:
                # Find where the phrase appears in content
                pos = content.lower().find(phrase.lower())
                start = max(0, pos - 30)
                end = min(len(content), pos + len(phrase) + 30)
                context = content[start:end].strip() if pos >= 0 else phrase

                suggestions.append(SuggestedLink(
                    target_title=title,
                    match_text=phrase,
                    score=score,
                    match_type="fuzzy",
                    context=context,
                    position=pos if pos >= 0 else 0,
                ))
                seen_titles.add(title)

    # ── Strategy 3: Known entity matching ───────────────────────────────
    # Look for rsIDs, gene names, pathway names that match existing notes
    entities = _extract_entities(content)
    for entity in entities:
        if entity in seen_titles:
            continue

        # Check exact match first
        if entity in index.notes and entity != exclude_title:
            pos = content.find(entity)
            start = max(0, pos - 30)
            end = min(len(content), pos + len(entity) + 30)
            context = content[start:end].strip() if pos >= 0 else entity

            suggestions.append(SuggestedLink(
                target_title=entity,
                match_text=entity,
                score=100.0,
                match_type="exact",
                context=context,
                position=pos if pos >= 0 else 0,
            ))
            seen_titles.add(entity)

    # ── Sort by score descending ────────────────────────────────────────
    suggestions.sort(key=lambda s: s.score, reverse=True)

    # ── Build auto-linked content ───────────────────────────────────────
    linked_content = _insert_wikilinks(content, suggestions)

    # ── Determine related notes ─────────────────────────────────────────
    related = [s.target_title for s in suggestions[:15]]

    # ── Suggest parent ──────────────────────────────────────────────────
    from backend.config import PARENT_MAP
    parent = PARENT_MAP.get(category, "")

    return LinkingResult(
        suggested_links=suggestions,
        linked_content=linked_content,
        related_notes=related,
        parent_suggestion=parent,
    )


def _extract_significant_phrases(content: str) -> list[str]:
    """Extract meaningful multi-word phrases from content for fuzzy matching."""
    # Remove markdown syntax
    clean = re.sub(r'[#>|`*_\[\]]', ' ', content)
    clean = re.sub(r'---+', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    phrases = set()

    # Extract capitalized phrases (likely proper nouns / technical terms)
    # e.g., "Magnesium Glycinate", "COMT Val/Val", "HPA Axis"
    cap_pattern = re.compile(r'(?:[A-Z][a-z]+(?:\s+(?:&|and|of|the|in|for|—|-)\s+)?){2,}[A-Z]?[a-z]*')
    for match in cap_pattern.finditer(content):
        phrase = match.group(0).strip()
        if 5 < len(phrase) < 80:
            phrases.add(phrase)

    # Extract technical terms with special patterns
    tech_patterns = [
        r'[A-Z]{2,}[- ]?\d*[A-Za-z]*',          # AMPK, GLUT4, CYP2D6, mTORC1
        r'[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+',       # Multi-word proper nouns
        r'rs\d+\s*(?:—|[-–])\s*[A-Z]+',            # rsID — GENE patterns
    ]
    for pat in tech_patterns:
        for match in re.finditer(pat, content):
            term = match.group(0).strip()
            if 3 < len(term) < 60:
                phrases.add(term)

    return list(phrases)


def _extract_entities(content: str) -> list[str]:
    """Extract known entity types (rsIDs, gene names, etc.) from content."""
    entities = set()

    # rsIDs
    for match in re.finditer(r'rs\d{4,}', content):
        entities.add(match.group(0))

    # Gene-style patterns (e.g., COMT, MTHFR, ACTN3)
    for match in re.finditer(r'\b[A-Z][A-Z0-9]{2,6}\b', content):
        gene = match.group(0)
        # Filter out common non-gene abbreviations
        if gene not in {"THE", "AND", "FOR", "NOT", "BUT", "HAS", "WAS",
                        "ARE", "CAN", "ALL", "SET", "GET", "NEW", "OLD",
                        "DAY", "USE", "TRY", "RUN", "PUT", "ADD", "END",
                        "URL", "API", "CSS", "HTML", "PDF", "TBD"}:
            entities.add(gene)

    # rsID — GENE patterns (full note title format)
    for match in re.finditer(r'(rs\d+)\s*(?:—|[-–])\s*([A-Z][A-Z0-9]+)', content):
        entities.add(f"{match.group(1)} — {match.group(2)}")

    return list(entities)


def _insert_wikilinks(content: str, suggestions: list[SuggestedLink]) -> str:
    """Insert [[wikilinks]] at first mention of each matched title in the content."""
    if not suggestions:
        return content

    # Only insert for high-confidence matches
    to_insert = [s for s in suggestions if s.score >= 90]

    # Sort by position descending so we can replace from end to start
    # without messing up positions
    to_insert.sort(key=lambda s: s.position, reverse=True)

    result = content
    inserted_titles = set()

    for suggestion in to_insert:
        if suggestion.target_title in inserted_titles:
            continue

        # Find the exact match text in content (first occurrence only)
        pattern = re.compile(re.escape(suggestion.match_text), re.IGNORECASE)
        match = pattern.search(result)
        if match:
            # Don't insert inside existing wikilinks or headings
            before = result[:match.start()]
            if before.count("[[") > before.count("]]"):
                continue  # Inside a wikilink already
            line_start = before.rfind("\n") + 1
            line_prefix = result[line_start:match.start()].lstrip()
            if line_prefix.startswith("#"):
                continue  # Inside a heading

            # Replace with wikilink
            original = match.group(0)
            if original == suggestion.target_title:
                replacement = f"[[{suggestion.target_title}]]"
            else:
                replacement = f"[[{suggestion.target_title}|{original}]]"

            result = result[:match.start()] + replacement + result[match.end():]
            inserted_titles.add(suggestion.target_title)

    return result
