"""
Vault Scribe — Content Classifier
Rule-based content classification that maps raw text to the correct vault folder
and note template. Uses weighted keyword scoring derived from the vault's actual content.
"""

import re
from dataclasses import dataclass

from backend.config import (
    CLASSIFICATION_KEYWORDS,
    FOLDER_MAP,
    PARENT_MAP,
    SNP_TAG_MAP,
)


@dataclass
class Classification:
    """Result of classifying a piece of content."""
    category: str                   # e.g., "biochemistry", "snp", "nutrition"
    folder: str                     # e.g., "knowledge/reference/biochemistry"
    template: str                   # e.g., "biochemistry", "snp"
    parent: str                     # e.g., "[[Biochemistry]]"
    confidence: str                 # "high", "medium", "low"
    score: float                    # raw weighted score
    reasoning: str                  # human-readable explanation
    all_scores: dict[str, float]    # all category scores for transparency
    snp_tag: str = ""               # if category is SNP, the suggested #tag


def classify_content(content: str, template_override: str = "") -> Classification:
    """
    Classify raw content into a vault category using weighted keyword matching.

    Args:
        content: The raw text to classify.
        template_override: If set, forces this template/category.

    Returns:
        Classification with category, folder, template, confidence, etc.
    """
    # ── Handle template override ────────────────────────────────────────
    if template_override and template_override in FOLDER_MAP:
        return Classification(
            category=template_override,
            folder=FOLDER_MAP.get(template_override, "inbox"),
            template=_category_to_template(template_override),
            parent=PARENT_MAP.get(template_override, ""),
            confidence="high",
            score=100.0,
            reasoning=f"Template manually set to '{template_override}'.",
            all_scores={template_override: 100.0},
        )

    # ── Score each category ─────────────────────────────────────────────
    scores: dict[str, float] = {}
    match_details: dict[str, list[str]] = {}

    for category, keywords in CLASSIFICATION_KEYWORDS.items():
        total_score = 0.0
        matched = []

        for pattern, weight in keywords:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Scale by number of matches (diminishing returns)
                count = len(matches)
                scaled = weight * (1 + 0.3 * min(count - 1, 5))
                total_score += scaled
                matched.append(f"{pattern} ({count}×, +{scaled:.1f})")

        if total_score > 0:
            scores[category] = total_score
            match_details[category] = matched

    # ── Determine winner ────────────────────────────────────────────────
    if not scores:
        return Classification(
            category="inbox",
            folder=FOLDER_MAP["inbox"],
            template="reference",
            parent="",
            confidence="low",
            score=0.0,
            reasoning="No keyword matches found. Defaulting to inbox.",
            all_scores={},
        )

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = ranked[0]

    # Determine confidence based on score and gap to runner-up
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
    gap = best_score - runner_up_score

    if best_score >= 15 and gap >= 5:
        confidence = "high"
    elif best_score >= 8:
        confidence = "medium"
    else:
        confidence = "low"

    # Build reasoning
    top_matches = match_details.get(best_category, [])[:5]
    reasoning = f"Matched '{best_category}' (score: {best_score:.1f}). "
    reasoning += f"Top signals: {', '.join(top_matches[:3])}."
    if runner_up_score > 0:
        runner_up_cat = ranked[1][0]
        reasoning += f" Runner-up: '{runner_up_cat}' ({runner_up_score:.1f})."

    # Determine SNP tag if applicable
    snp_tag = ""
    if best_category == "snp":
        snp_tag = _detect_snp_tag(content)

    # Map category to template (some categories share templates)
    template = _category_to_template(best_category)

    return Classification(
        category=best_category,
        folder=FOLDER_MAP.get(best_category, "inbox"),
        template=template,
        parent=PARENT_MAP.get(best_category, ""),
        confidence=confidence,
        score=best_score,
        reasoning=reasoning,
        all_scores=dict(ranked),
        snp_tag=snp_tag,
    )


def _category_to_template(category: str) -> str:
    """Map a category slug to its template name."""
    # Categories that have their own template
    direct_templates = {
        "snp", "biochemistry", "fitness", "neuroacoustics",
        "nutrition", "pharmacogenomics", "graph",
    }
    if category in direct_templates:
        return category

    # Categories that use the generic reference template
    reference_categories = {
        "anatomy", "exercise-physiology", "pharmacology", "supplements",
        "immunology", "pathophysiology", "epidemiology", "toxicology",
        "microbiology", "molecular-biology", "gerontology", "lab-values",
        "neuroscience", "medical-entomology", "pubmed",
    }
    if category in reference_categories:
        return "reference"

    return "reference"


def _detect_snp_tag(content: str) -> str:
    """Detect the most appropriate SNP category tag from content."""
    content_lower = content.lower()
    tag_scores: dict[str, int] = {}

    for keyword, tag in SNP_TAG_MAP.items():
        count = content_lower.count(keyword.lower())
        if count > 0:
            tag_scores[tag] = tag_scores.get(tag, 0) + count

    if not tag_scores:
        return "#genetics"

    best_tag = max(tag_scores, key=tag_scores.get)
    return best_tag


def suggest_title(content: str, category: str) -> str:
    """Suggest a note title based on content and category."""
    content_stripped = content.strip()

    # If content starts with a markdown heading, use it
    lines = content_stripped.split("\n")
    for line in lines[:5]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    # For SNPs, try to extract rsID and gene
    if category == "snp":
        rs_match = re.search(r'(rs\d+)', content)
        gene_match = re.search(
            r'(?:gene[:\s]+|—\s*)([A-Z][A-Z0-9]{1,10})',
            content, re.IGNORECASE
        )
        if rs_match:
            rsid = rs_match.group(1)
            gene = gene_match.group(1) if gene_match else "Unknown"
            return f"{rsid} — {gene}"

    # For graph/atomic notes, use the first sentence as title (prose-as-title)
    if category == "graph":
        first_sentence = re.split(r'[.!?]', content_stripped)[0].strip()
        if len(first_sentence) > 10:
            return first_sentence[:100]

    # Default: use first meaningful line, cleaned up
    for line in lines:
        line = line.strip()
        if line and not line.startswith(">") and not line.startswith("#") and line != "---":
            # Truncate to reasonable title length
            title = line[:80]
            if len(line) > 80:
                title = title.rsplit(" ", 1)[0] + "..."
            return title

    return "Untitled Note"
