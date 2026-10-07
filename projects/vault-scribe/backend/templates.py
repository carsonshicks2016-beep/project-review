"""
Vault Scribe — Note Templates
Generates markdown notes matching the exact conventions of the carson-brain vault.
All templates use blockquote metadata headers (NOT yaml frontmatter).
"""

from datetime import datetime


def snp_template(
    rsid: str,
    gene: str,
    genotype: str = "—",
    impact: str = "Unknown",
    trait: str = "",
    evidence_score: int = 0,
    category_tag: str = "",
    clinical_significance: str = "",
    countermeasures: str = "",
    related_notes: list[str] | None = None,
) -> str:
    """Generate an SNP note matching knowledge/genetics/individual-snps/ convention."""
    related = related_notes or []
    related_section = "\n".join(f"- [[{n}]]" for n in related)
    if not related_section:
        related_section = "- [[Full SNP Registry]]"
    else:
        related_section = "- [[Full SNP Registry]]\n" + related_section

    tag_str = f"#snp #genetics {category_tag}".strip()

    return f"""# {rsid} — {gene}

> Parent: [[Full SNP Registry]]
> Genotype: **{genotype}**
> Impact: {impact}
> Trait: {trait}
> Evidence Score: {evidence_score}/100
> Tags: {tag_str}

---

## Clinical Significance
{clinical_significance or "To be documented."}

## Countermeasures & Protocols
{countermeasures or "- **Actionable Steps:** [ ]\\n- **Supplementation:** [ ]"}

## Evidence
- PubMed
- GWAS Catalog
- dbSNP

## Related Notes
{related_section}
"""


def biochemistry_template(
    title: str,
    subcategory: str = "Biochemistry",
    relevance: str = "",
    content: str = "",
    cross_references: list[str] | None = None,
) -> str:
    """Generate a biochemistry reference note matching knowledge/reference/biochemistry/ convention."""
    refs = cross_references or []
    refs_section = "\n".join(f"- [[{r}]]" for r in refs) if refs else ""

    return f"""# {title}

> Parent: Biochemistry
> Type: Academic Reference — {subcategory}
> Relevance: {relevance or "Connects to core vault biochemistry network."}

---

{content or "## Overview\\n\\nTo be documented."}

{"## Cross-References" + chr(10) + refs_section if refs_section else ""}
"""


def fitness_session_template(
    date: str = "",
    time_of_day: str = "Mid-Day",
    duration: int = 0,
    workout_type: str = "Full Body",
    total_volume: int = 0,
    whoop: str = "N/A",
    spotify: str = "N/A",
    is_elite: bool = False,
    summary: str = "",
    exercises: str = "",
    genomic_context: str = "",
) -> str:
    """Generate a fitness session note matching knowledge/fitness/sessions/ convention."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    date_time = f"{date} {datetime.now().strftime('%H:%M')}"
    elite_tag = " #Elite-Session" if is_elite else ""
    callout_type = "SUCCESS" if is_elite else "INFO"
    callout_label = "Elite Performance" if is_elite else "Standard Session"

    return f"""# {time_of_day} Workout: {date}{elite_tag}

> [!{callout_type}] {callout_label}
> - **Date:** {date_time}
> - **Duration:** {duration} minutes
> - **Type:** {workout_type}
> - **Total Volume:** {total_volume:,} lbs
> - **WHOOP:** {whoop}
> - **Spotify:** {spotify}

## Summary
{summary or "Session notes to be added."}

## 🧬 Athletic Genomic Context
{genomic_context or "- See [[Full SNP Registry]] for relevant athletic variants."}

## Exercises
{exercises or "### Exercise Name (Muscle Group)\\n| Set | Weight | Reps | Volume | Est. 1RM |\\n|-----|--------|------|--------|----------|\\n| 1   |        |      |        |          |"}

---
[[workouts|↩ Back to Workouts Hub]]
"""


def neuroacoustics_template(
    title: str,
    subcategory: str = "Psychoacoustics",
    genres: str = "",
    artists: str = "",
    content: str = "",
    cross_references: list[str] | None = None,
) -> str:
    """Generate a neuroacoustics note matching knowledge/neuroacoustics/ convention."""
    refs = cross_references or []
    refs_section = "\n".join(f"- [[{r}]]" for r in refs) if refs else ""

    meta_lines = [
        f"> Parent: [[_Neuroacoustics Master Index]]",
        f"> Type: Academic Reference — Psychoacoustics & {subcategory}",
    ]
    if genres:
        meta_lines.append(f"> Relevant Genres: {genres}")
    if artists:
        meta_lines.append(f"> Artists: {artists}")

    meta_block = "\n".join(meta_lines)

    return f"""# {title}

{meta_block}

---

{content or "## Overview\\n\\nTo be documented."}

{"## Cross-References" + chr(10) + refs_section if refs_section else ""}
"""


def nutrition_template(
    title: str,
    description: str = "",
    content: str = "",
) -> str:
    """Generate a nutrition database note matching knowledge/nutrition/ convention."""
    return f"""# {title}

> Parent: [[Nutrition Database Index]]
> {description or "Nutritional data reference. All values per **100 g**."}

---

{content or "## Items\\n| Item | Cal | P (g) | F (g) | C (g) | Fiber | Key Micronutrients | Diet Tags | Notes |\\n|------|-----|--------|--------|--------|-------|-------------------|-----------|-------|\\n|      |     |        |        |        |       |                   |           |       |"}
"""


def pharmacogenomics_template(
    title: str,
    variants_table: str = "",
    metabolizer_status: str = "",
    drug_classes: str = "",
    cpic_refs: str = "",
    related_notes: list[str] | None = None,
) -> str:
    """Generate a pharmacogenomics note matching knowledge/genetics/pharmacogenomics/ convention."""
    refs = related_notes or []
    refs_section = "\n".join(f"- [[{r}]]" for r in refs) if refs else ""

    return f"""# {title}

> Parent: [[Drug Interactions]]

## Your Variants
{variants_table or "| SNP | Genotype | Functional Status |\\n|-----|----------|-------------------|\\n|     |          |                   |"}

## Predicted Metabolizer Status
{metabolizer_status or "To be determined based on variant analysis."}

## What This Means for Specific Drug Classes
{drug_classes or "### Drug Class\\nClinical guidance to be documented."}

## CPIC Guideline Reference
{cpic_refs or "- CPIC guidelines pending review."}

{"## Related Notes" + chr(10) + refs_section if refs_section else ""}
"""


def graph_atomic_template(
    claim_title: str,
    claim: str = "",
    mechanism: str = "",
    thresholds: str = "",
    implications: str = "",
    linked_notes: list[str] | None = None,
) -> str:
    """Generate a graph/atomic note matching knowledge/graph/ convention (prose-as-title)."""
    links = linked_notes or []
    links_section = "\n".join(f"- [[{n}]]" for n in links) if links else ""

    return f"""# {claim_title}

## Claim
{claim or claim_title}

## Mechanism
{mechanism or "- To be documented."}

## Quantitative Thresholds
{thresholds or "- To be documented."}

## Practical Implications
{implications or "- To be documented."}

{"## Linked Notes" + chr(10) + links_section if links_section else ""}
"""


def hub_template(
    title: str,
    hub_type: str = "Master Hub",
    description: str = "",
    parent: str = "",
    categories: dict[str, list[str]] | None = None,
    cross_references: list[str] | None = None,
) -> str:
    """Generate a hub/index note."""
    parent_line = f"> Parent: [[{parent}]]" if parent else f"> Parent: Knowledge"
    cats = categories or {}
    refs = cross_references or []

    category_sections = ""
    for cat_name, notes in cats.items():
        links = "\n".join(f"- [[{n}]]" for n in notes)
        category_sections += f"\n## {cat_name}\n{links}\n"

    refs_section = "\n".join(f"- [[{r}]]" for r in refs) if refs else ""

    return f"""# {title}

{parent_line}
> Type: {hub_type} — {description or title}

---
{category_sections or "## Notes\\n- *(No notes linked yet)*"}
{"---" + chr(10) + chr(10) + "## Cross-References" + chr(10) + refs_section if refs_section else ""}
"""


def generic_reference_template(
    title: str,
    parent: str = "",
    note_type: str = "Academic Reference",
    subcategory: str = "",
    relevance: str = "",
    content: str = "",
    cross_references: list[str] | None = None,
) -> str:
    """Generate a generic reference note for any knowledge/reference/ subfolder."""
    refs = cross_references or []
    refs_section = "\n".join(f"- [[{r}]]" for r in refs) if refs else ""

    parent_line = f"> Parent: [[{parent}]]" if parent else "> Parent: Reference Library"
    type_str = f"{note_type} — {subcategory}" if subcategory else note_type

    return f"""# {title}

{parent_line}
> Type: {type_str}
{">" + " Relevance: " + relevance if relevance else ""}

---

{content or "## Overview\\n\\nTo be documented."}

{"## Cross-References" + chr(10) + refs_section if refs_section else ""}
"""


def experiment_template(
    title: str,
    intervention: str = "",
    hypothesis: str = "",
    dose: str = "",
    timing: str = "",
    duration: str = "30 days",
    tracking: str = "",
    linked_research: list[str] | None = None,
) -> str:
    """Generate an experiment tracking note."""
    date = datetime.now().strftime("%Y-%m-%d")
    links = linked_research or []
    links_section = "\n".join(f"- [[{n}]]" for n in links)

    return f"""# Experiment: {title}

> Parent: [[Experiments]]
> Start Date: {date}
> Status: Active
> Intervention: {intervention or title}
> Hypothesis: {hypothesis or "To be defined."}

---

## Protocol
- **Dose:** {dose or "TBD"}
- **Timing:** {timing or "TBD"}
- **Duration:** {duration}
- **Tracking:** {tracking or "Subjective notes, relevant biomarkers"}

## Daily Log
| Day | Date | Notes | Metric | Subjective |
|-----|------|-------|--------|------------|
| 1   | {date} | Baseline | — | — |

## Linked Research
{links_section or "- *(Add relevant research notes)*"}

## Outcome
*To be completed at trial end.*
"""


# ── Template Registry ───────────────────────────────────────────────────────
# Maps template names (used in UI and API) to their generator functions
TEMPLATE_REGISTRY: dict[str, callable] = {
    "snp": snp_template,
    "biochemistry": biochemistry_template,
    "fitness": fitness_session_template,
    "neuroacoustics": neuroacoustics_template,
    "nutrition": nutrition_template,
    "pharmacogenomics": pharmacogenomics_template,
    "graph": graph_atomic_template,
    "hub": hub_template,
    "reference": generic_reference_template,
    "experiment": experiment_template,
}

# Human-readable names for the UI
TEMPLATE_DISPLAY_NAMES: dict[str, str] = {
    "snp": "SNP",
    "biochemistry": "Biochemistry Reference",
    "fitness": "Fitness Session",
    "neuroacoustics": "Neuroacoustics",
    "nutrition": "Nutrition Database",
    "pharmacogenomics": "Pharmacogenomics",
    "graph": "Graph / Atomic",
    "hub": "Hub / Index",
    "reference": "Generic Reference",
    "experiment": "Experiment",
}
