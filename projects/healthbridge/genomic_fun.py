"""
HealthBridge AI - Fun phenotype and population-enrichment notes

These helpers intentionally keep the claims soft. They use a very small set of
observable-trait and population-enriched markers to create a "for fun" sketch,
not a serious ancestry or appearance model.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from report_evidence import get_snp_reference_links


PHENOTYPE_DISCLAIMER = (
    "For fun only. This sketch uses a handful of visible-trait variants and is "
    "nowhere near strong enough to identify appearance with high confidence. "
    "Hair color, eye color, skin tone, tanning, body composition, and facial "
    "features are strongly shaped by many other variants plus age and environment."
)

POPULATION_DISCLAIMER = (
    "For fun only. These are population-enrichment notes, not ancestry results. "
    "A few variants can hint at where an allele is commonly studied or enriched, "
    "but real ancestry inference requires genome-wide admixture methods."
)


def _flatten_lookup(dna_results: Dict) -> Dict[str, Dict]:
    lookup = {}
    for category, snps in (dna_results or {}).items():
        for snp in snps:
            rsid = snp.get("rsid")
            if rsid and rsid not in lookup:
                lookup[rsid] = snp
    return lookup


def _alleles(genotype: Optional[str]) -> List[str]:
    if not genotype:
        return []
    return [item.strip().upper() for item in genotype.replace("|", "/").split("/") if item.strip()]


def _count_allele(lookup: Dict[str, Dict], rsid: str, allele: str) -> int:
    return sum(1 for item in _alleles(lookup.get(rsid, {}).get("genotype")) if item == allele.upper())


def _variant_links(lookup: Dict[str, Dict], rsids: List[str]) -> List[Dict]:
    links = []
    seen = set()
    for rsid in rsids:
        variant = lookup.get(rsid)
        if not variant:
            continue
        for link in get_snp_reference_links(rsid, variant.get("gene"), variant.get("trait"))[:2]:
            url = link.get("url")
            if url and url not in seen:
                seen.add(url)
                links.append(link)
    return links[:4]


def _eye_color_card(lookup: Dict[str, Dict]) -> Optional[Dict]:
    if "rs12913832" not in lookup:
        return None

    base = _count_allele(lookup, "rs12913832", "A")
    oca2 = _count_allele(lookup, "rs1800407", "A")
    irf4 = _count_allele(lookup, "rs12203592", "T")

    if base >= 2:
        estimate = "Light eyes are more likely"
        confidence = "Moderate"
    elif base == 1:
        estimate = "Intermediate or hazel-leaning eyes are plausible"
        confidence = "Limited"
    else:
        estimate = "Brown or darker eyes are more likely"
        confidence = "Moderate"

    modifier_line = []
    if oca2:
        modifier_line.append("OCA2 pushes toward lighter pigmentation")
    if irf4:
        modifier_line.append("IRF4 also nudges toward lighter iris color")

    body = (
        "HERC2 rs12913832 is the biggest visible-trait driver in this sketch. "
        + (" ".join(modifier_line) if modifier_line else "No strong light-eye modifiers were seen in the smaller secondary set.")
    )

    return {
        "title": "Eye color sketch",
        "estimate": estimate,
        "confidence": confidence,
        "body": body,
        "links": _variant_links(lookup, ["rs12913832", "rs1800407", "rs12203592"]),
    }


def _skin_uv_card(lookup: Dict[str, Dict]) -> Optional[Dict]:
    relevant = [lookup.get("rs1426654"), lookup.get("rs16891982"), lookup.get("rs4911414")]
    if not any(relevant):
        return None

    pigmentation_score = (
        _count_allele(lookup, "rs1426654", "A")
        + _count_allele(lookup, "rs16891982", "C")
        + _count_allele(lookup, "rs4911414", "T")
    )
    mc1r_score = sum(
        _count_allele(lookup, rsid, allele)
        for rsid, allele in [
            ("rs1805007", "T"),
            ("rs1805008", "T"),
            ("rs885479", "T"),
            ("rs2228479", "A"),
            ("rs1110400", "T"),
        ]
    )

    if pigmentation_score >= 4 or mc1r_score >= 2:
        estimate = "Lighter pigmentation and easier burning are more plausible"
        confidence = "Limited-Moderate"
    elif pigmentation_score >= 2:
        estimate = "Intermediate pigmentation with some UV sensitivity is plausible"
        confidence = "Limited"
    else:
        estimate = "No strong lighter-pigmentation signal surfaced in this subset"
        confidence = "Limited"

    body = (
        "This sketch combines SLC24A5, SLC45A2, and ASIP pigmentation markers with "
        "MC1R-related UV-sensitivity variants. It is useful for rough direction only, "
        "not a reliable skin-tone prediction."
    )

    return {
        "title": "Skin tone / UV sensitivity sketch",
        "estimate": estimate,
        "confidence": confidence,
        "body": body,
        "links": _variant_links(lookup, ["rs1426654", "rs16891982", "rs4911414", "rs1805007", "rs1805008"]),
    }


def _hair_freckle_card(lookup: Dict[str, Dict]) -> Optional[Dict]:
    mc1r_score = sum(
        _count_allele(lookup, rsid, allele)
        for rsid, allele in [
            ("rs1805007", "T"),
            ("rs1805008", "T"),
            ("rs885479", "T"),
            ("rs2228479", "A"),
            ("rs1110400", "T"),
        ]
    )
    freckle_score = _count_allele(lookup, "rs2153271", "T")
    if mc1r_score == 0 and freckle_score == 0:
        return None

    if mc1r_score >= 2:
        estimate = "Freckling and red or auburn undertones are more plausible"
        confidence = "Limited-Moderate"
    elif mc1r_score == 1 or freckle_score:
        estimate = "Some freckling or lighter hair undertones are plausible"
        confidence = "Limited"
    else:
        estimate = "No strong hair-color signal surfaced from this subset"
        confidence = "Limited"

    body = (
        "MC1R drives most of the red-hair / freckling signal here, while BNC2 adds "
        "a smaller freckle tendency. These variants are probabilistic, not deterministic."
    )

    return {
        "title": "Hair / freckling sketch",
        "estimate": estimate,
        "confidence": confidence,
        "body": body,
        "links": _variant_links(lookup, ["rs1805007", "rs1805008", "rs885479", "rs2153271"]),
    }


def _hair_texture_card(lookup: Dict[str, Dict]) -> Optional[Dict]:
    edar = _count_allele(lookup, "rs3827760", "A")
    abcc11 = _count_allele(lookup, "rs17822931", "A")
    if not edar and not abcc11:
        return None

    parts = []
    if edar:
        parts.append("EDAR points toward thicker, straighter hair shafts")
    if abcc11:
        parts.append("ABCC11 is the classic dry-earwax / lower underarm-odor marker")

    return {
        "title": "Hair texture / body trait notes",
        "estimate": "A couple of classic visible-trait markers were present",
        "confidence": "Limited",
        "body": ". ".join(parts) + ". These are fun observational traits, not health-risk interpretations.",
        "links": _variant_links(lookup, ["rs3827760", "rs17822931"]),
    }


def build_fun_phenotype_cards(dna_results: Dict) -> Dict:
    lookup = _flatten_lookup(dna_results)
    cards = []
    for builder in (_eye_color_card, _skin_uv_card, _hair_freckle_card, _hair_texture_card):
        card = builder(lookup)
        if card:
            cards.append(card)

    return {
        "disclaimer": PHENOTYPE_DISCLAIMER,
        "cards": cards[:4],
    }


POPULATION_RULES = [
    {
        "rsid": "rs4988235",
        "allele": "T",
        "title": "Lactase-persistence signal",
        "body": "The lactase-persistence allele is well known in Northern European datasets and some pastoral populations elsewhere. It says something about one digestion-related allele, not your ancestry as a whole.",
    },
    {
        "rsid": "rs671",
        "allele": "A",
        "title": "ALDH2 population-enrichment note",
        "body": "The ALDH2 flush allele is most strongly enriched in East Asian populations and is uncommon in most European and African reference datasets.",
    },
    {
        "rsid": "rs1229984",
        "allele": "A",
        "title": "ADH1B alcohol-metabolism note",
        "body": "This fast-metabolizing ADH1B allele is commonly discussed in East Asian and some Middle Eastern / Ashkenazi Jewish datasets. It is a population-enrichment note, not an ancestry call.",
    },
    {
        "rsid": "rs3827760",
        "allele": "A",
        "title": "EDAR population-enrichment note",
        "body": "The EDAR V370A allele is famously enriched in East Asian and Native American reference populations and is often used in visible-trait studies.",
    },
    {
        "rsid": "rs17822931",
        "allele": "A",
        "title": "ABCC11 population-enrichment note",
        "body": "The dry-earwax ABCC11 allele is strongly enriched in East Asian reference datasets and much less common in many European and African datasets.",
    },
    {
        "rsid": "rs1426654",
        "allele": "A",
        "title": "SLC24A5 pigmentation note",
        "body": "This light-pigmentation allele is strongly enriched in West Eurasian reference populations and appears in many pigmentation studies.",
    },
    {
        "rsid": "rs12913832",
        "allele": "A",
        "title": "HERC2 pigmentation note",
        "body": "The classic light-eye allele is especially common in Northern and Eastern European reference populations, but it should not be used as a proxy for ancestry on its own.",
    },
]


def build_population_notes(dna_results: Dict) -> Dict:
    lookup = _flatten_lookup(dna_results)
    cards = []

    for rule in POPULATION_RULES:
        if _count_allele(lookup, rule["rsid"], rule["allele"]) <= 0:
            continue
        variant = lookup.get(rule["rsid"], {})
        cards.append({
            "title": rule["title"],
            "estimate": f"{rule['rsid']} is present",
            "confidence": "For fun",
            "body": rule["body"],
            "links": _variant_links(lookup, [rule["rsid"]]),
        })

    return {
        "disclaimer": POPULATION_DISCLAIMER,
        "cards": cards[:6],
    }
