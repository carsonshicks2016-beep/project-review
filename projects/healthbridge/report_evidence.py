"""
HealthBridge AI - Report evidence link helpers

Builds outward-facing literature and database links for biomarkers and SNPs.
These links are intentionally conservative: they surface primary-source search
paths and authoritative databases without overstating the certainty of any
single marker.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from urllib.parse import quote_plus


BIOMARKER_DISPLAY_TERMS = {
    "hba1c": "hemoglobin A1c",
    "homa_ir": "HOMA-IR insulin resistance index",
    "hscrp": "high sensitivity C-reactive protein",
    "apob": "apolipoprotein B",
    "lpa": "lipoprotein(a)",
    "il6": "interleukin-6",
    "vitamin_d": "25-hydroxy vitamin D",
    "vitamin_b12": "vitamin B12",
    "free_t4": "free thyroxine",
    "free_t3": "free triiodothyronine",
    "reverse_t3": "reverse triiodothyronine",
    "testosterone_total": "total testosterone",
    "testosterone_free": "free testosterone",
    "dhea_s": "DHEA-S",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "ggt": "gamma glutamyl transferase",
    "creatinine": "serum creatinine",
    "egfr": "estimated glomerular filtration rate",
    "bun": "blood urea nitrogen",
}

BIOMARKER_CATEGORY_QUERIES = {
    "Metabolic": "clinical utility meta-analysis",
    "Lipids": "cardiovascular risk meta-analysis",
    "Inflammation": "clinical utility meta-analysis",
    "Iron": "clinical utility review",
    "Vitamins": "clinical utility guideline review",
    "Minerals": "clinical utility review",
    "Thyroid": "guideline clinical utility review",
    "Hormones": "Endocrine Society guideline review",
    "Liver": "clinical utility review",
    "Kidney": "clinical utility review",
}

BIOMARKER_CURATED_SEARCHES = {
    "glucose": [
        ("PubMed clinical utility", '"fasting glucose" cardiovascular risk meta-analysis'),
        ("PubMed biological variation", '"fasting glucose" biological variation reliability'),
    ],
    "hba1c": [
        ("PubMed diabetes standards", '"hemoglobin A1c" diabetes diagnosis standards'),
        ("PubMed biological variation", '"hemoglobin A1c" biological variation reliability'),
    ],
    "insulin": [
        ("PubMed insulin resistance", '"fasting insulin" insulin resistance clinical utility'),
        ("PubMed biological variation", '"fasting insulin" biological variation reliability'),
    ],
    "apob": [
        ("PubMed apoB outcomes", '"apolipoprotein B" cardiovascular risk meta-analysis'),
        ("PubMed biological variation", '"apolipoprotein B" biological variation reliability'),
    ],
    "lpa": [
        ("PubMed Lp(a) outcomes", '"lipoprotein(a)" cardiovascular risk meta-analysis'),
        ("PubMed biological variation", '"lipoprotein(a)" biological variation reliability'),
    ],
    "hscrp": [
        ("PubMed hs-CRP outcomes", '"high sensitivity C-reactive protein" cardiovascular risk meta-analysis'),
        ("PubMed biological variation", '"high sensitivity C-reactive protein" biological variation reliability'),
    ],
    "vitamin_d": [
        ("PubMed vitamin D guideline literature", '"25-hydroxy vitamin D" guideline review'),
        ("PubMed biological variation", '"25-hydroxy vitamin D" biological variation reliability'),
    ],
    "tsh": [
        ("PubMed thyroid guideline literature", '"thyroid stimulating hormone" guideline review'),
        ("PubMed biological variation", '"thyroid stimulating hormone" biological variation reliability'),
    ],
    "testosterone_total": [
        ("PubMed testosterone guideline literature", '"total testosterone" hypogonadism guideline'),
        ("PubMed biological variation", '"total testosterone" biological variation reliability'),
    ],
    "cortisol": [
        ("PubMed cortisol interpretation", '"morning cortisol" clinical utility review'),
        ("PubMed biological variation", '"cortisol" diurnal variation reliability'),
    ],
}


def _dedupe_links(links: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for link in links:
        url = link.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(link)
    return deduped


def canonical_rsid(rsid: Optional[str]) -> str:
    value = (rsid or "").strip().lower()
    if not value:
        return ""
    if "_" in value:
        value = value.split("_", 1)[0]
    if not value.startswith("rs"):
        value = f"rs{value.lstrip('rs')}"
    return value


def pubmed_article_url(pmid: int | str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def pubmed_search_url(query: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(query)}"


def get_biomarker_evidence_links(
    biomarker: str,
    clinical_name: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict]:
    key = (biomarker or "").lower()
    term = BIOMARKER_DISPLAY_TERMS.get(key) or clinical_name or biomarker.replace("_", " ")
    utility_suffix = BIOMARKER_CATEGORY_QUERIES.get(category or "", "clinical utility review")

    links = []
    for label, query in BIOMARKER_CURATED_SEARCHES.get(key, []):
        links.append({
            "label": label,
            "url": pubmed_search_url(query),
            "source": "PubMed",
        })

    links.extend([
        {
            "label": "PubMed evidence search",
            "url": pubmed_search_url(f'"{term}" {utility_suffix}'),
            "source": "PubMed",
        },
        {
            "label": "PubMed variability search",
            "url": pubmed_search_url(f'"{term}" biological variation reliability'),
            "source": "PubMed",
        },
    ])

    return _dedupe_links(links)[:4]


def get_snp_reference_links(
    rsid: str,
    gene: Optional[str] = None,
    trait: Optional[str] = None,
    curated_evidence: Optional[List[Dict]] = None,
) -> List[Dict]:
    canonical = canonical_rsid(rsid)
    if not canonical:
        return []

    search_parts = [canonical]
    if gene:
        search_parts.append(gene)
    if trait:
        search_parts.append(trait)
    search_term = " ".join(search_parts)

    links = []

    for item in curated_evidence or []:
        pmid = item.get("pmid")
        if not pmid:
            continue
        title = item.get("journal", "PubMed article")
        year = item.get("year")
        label = f"{title} {year}".strip()
        links.append({
            "label": label,
            "url": pubmed_article_url(pmid),
            "source": "PubMed",
        })

    links.extend([
        {
            "label": "PubMed search",
            "url": pubmed_search_url(search_term),
            "source": "PubMed",
        },
        {
            "label": "GWAS Catalog",
            "url": f"https://www.ebi.ac.uk/gwas/variants/{canonical}",
            "source": "GWAS Catalog",
        },
        {
            "label": "dbSNP",
            "url": f"https://www.ncbi.nlm.nih.gov/snp/{canonical}",
            "source": "NCBI dbSNP",
        },
        {
            "label": "ClinVar search",
            "url": f"https://www.ncbi.nlm.nih.gov/clinvar/?term={quote_plus(canonical)}",
            "source": "ClinVar",
        },
    ])

    return _dedupe_links(links)[:6]
