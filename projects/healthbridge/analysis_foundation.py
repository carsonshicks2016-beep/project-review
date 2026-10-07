"""
HealthBridge AI - Structured report foundation

Builds investor/demo-friendly report data with explicit provenance,
uncertainty, trust markers, and next-step guidance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clinical_evidence import get_biomarker_reliability_report, get_clinical_context
from genomic_fun import build_fun_phenotype_cards, build_population_notes
from report_evidence import get_biomarker_evidence_links, get_snp_reference_links
from snp_processor import SNP_PANEL


ENGINE_VERSION = "HB-Foundations-2026.04"

METHODOLOGY_SECTIONS = [
    {
        "title": "Deterministic Analysis First",
        "body": "Reference ranges, optimal targets, pattern matches, and follow-up prompts are computed from explicit code before the report is rendered.",
    },
    {
        "title": "Evidence-Graded Interpretation",
        "body": "Each major signal is paired with either biomarker reliability data, curated SNP evidence, or both. Stronger evidence is shown more prominently.",
    },
    {
        "title": "Uncertainty Stays Visible",
        "body": "Collection conditions, biomarker variability, missing companion markers, and single-timepoint limitations are shown directly in the report.",
    },
    {
        "title": "Clinical Guardrails",
        "body": "The system separates observations, interpretation, and next-step suggestions. It does not claim to diagnose disease from a single upload.",
    },
]

TRUST_MARKERS = [
    {
        "title": "Traceable Inputs",
        "body": "Every highlighted signal can be traced back to source biomarkers, variants, or collection-context notes.",
    },
    {
        "title": "Confidence Scored",
        "body": "HealthBridge shows a confidence band instead of presenting every conclusion with the same certainty.",
    },
    {
        "title": "Next-Test Guidance",
        "body": "Recommendations focus on clarifying ambiguous signals, not just producing more narrative.",
    },
    {
        "title": "Built For Review",
        "body": "Results are organized so a user, clinician, or investor can quickly see what was observed, why it matters, and what would change the conclusion.",
    },
]

BIOMARKER_NEXT_TESTS = {
    "glucose": [("Fasting insulin", "Helps distinguish isolated glucose elevation from early insulin resistance."), ("HbA1c", "Shows whether the signal is sustained across the last 8-12 weeks.")],
    "hba1c": [("Fasting glucose", "Separates average glucose exposure from a single fasting snapshot."), ("Fasting insulin", "Clarifies whether glycemic pressure is being compensated for with excess insulin.")],
    "insulin": [("HOMA-IR", "Quantifies insulin resistance using fasting glucose plus insulin."), ("ApoB", "Useful when insulin resistance may also be affecting particle burden.")],
    "ldl": [("ApoB", "Better estimates total atherogenic particle burden than LDL-C alone."), ("Coronary artery calcium", "Useful if lipid risk needs anatomical confirmation.")],
    "apob": [("Lipoprotein(a)", "Helps separate baseline particle burden from inherited cardiovascular risk."), ("Coronary artery calcium", "Can clarify whether elevated particle burden has translated into plaque.")],
    "lpa": [("ApoB", "Complements inherited risk with a modifiable particle-burden marker."), ("Coronary artery calcium", "Useful when inherited risk may be clinically significant.")],
    "triglycerides": [("Fasting insulin", "High triglycerides often travel with insulin resistance."), ("ApoB", "Adds cardiovascular context when triglycerides are elevated.")],
    "hscrp": [("Repeat hs-CRP", "This marker is variable and is more credible when elevated twice."), ("Ferritin", "Can help distinguish inflammation from isolated lipid/metabolic issues.")],
    "homocysteine": [("Vitamin B12", "Supports methylation interpretation."), ("Folate", "Clarifies whether homocysteine elevation may be nutrition-related.")],
    "vitamin_d": [("Repeat 25-OH vitamin D", "Confirms whether supplementation or sun exposure changed the signal."), ("Calcium", "Helpful when interpreting vitamin D status in a broader bone-health context.")],
    "tsh": [("Free T4", "Adds thyroid hormone output context."), ("Free T3", "Helps distinguish conversion issues from gland output issues.")],
    "testosterone_total": [("Free testosterone", "More closely reflects bioavailable androgen signal."), ("SHBG", "Needed to interpret whether normal total testosterone is actually available to tissues.")],
    "testosterone_free": [("SHBG", "Binding protein levels can explain low free testosterone despite normal totals."), ("Estradiol", "Adds context to sex-hormone balance.")],
    "shbg": [("Free testosterone", "Clarifies whether binding-protein levels are reducing hormone availability."), ("Fasting insulin", "SHBG often shifts with metabolic health.")],
    "cortisol": [("Repeat AM cortisol", "Stress-axis markers are more reliable when repeated under consistent timing."), ("Sleep duration / wearable recovery", "Helpful when cortisol may be a recovery signal rather than a chronic pathology signal.")],
}

GENE_FOLLOW_UP = {
    "APOE": [("ApoB", "Most actionable lipid marker to pair with APOE risk."), ("Coronary artery calcium", "Objective plaque screen for higher inherited lipid risk.")],
    "TCF7L2": [("Fasting insulin", "Clarifies whether the genotype is already expressing as insulin resistance."), ("HbA1c", "Adds time-averaged glucose context to fasting markers.")],
    "MTNR1B": [("Fasting glucose", "Most direct phenotype readout for circadian glucose-risk variants."), ("Sleep timing", "Useful when melatonin signaling variants may be clinically relevant.")],
    "FADS1": [("Omega-3 Index", "Confirms whether omega-3 conversion limitations are showing up in tissue levels."), ("Triglycerides", "A practical blood readout for lipid response.")],
    "BDNF": [("Sleep consistency", "Recovery and sleep quality are meaningful modifiers of cognitive vulnerability."), ("Exercise regularity", "Physical activity is one of the strongest known BDNF modifiers.")],
    "MTHFR": [("Homocysteine", "Functional methylation readout."), ("Vitamin B12", "Clarifies whether methylation support is limited by cofactor availability.")],
}

CATEGORY_EXPLAINERS = {
    "Metabolic": "Metabolic markers help estimate glycemic control, insulin sensitivity, and cardiometabolic strain.",
    "Cardiovascular": "Cardiovascular markers and variants are most believable when particle burden, inflammation, and inherited risk point in the same direction.",
    "Hormonal": "Hormone signals are sensitive to sleep, stress, timing, and binding proteins, so context matters as much as the raw number.",
    "Vitamins": "Micronutrient markers are more useful when paired with symptoms, repeat testing, or companion markers.",
    "Mental Health & Cognition": "These variants describe population-level predisposition, not destiny. Sleep, exercise, and cardiometabolic health can materially change expression.",
    "Sleep & Circadian": "Circadian variants matter most when they line up with glucose, recovery, or timing-related symptoms.",
}


def _mode_label(mode: str) -> str:
    return {
        "dna": "DNA Analysis",
        "blood": "Blood Work Analysis",
        "combined": "Combined Analysis",
    }.get(mode, "Health Analysis")


def _confidence_label(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 55:
        return "Moderate"
    return "Limited"


def _theme_for_status(status: str) -> str:
    mapping = {
        "critical": "critical",
        "high": "critical",
        "warning": "warning",
        "moderate": "warning",
        "abnormal": "critical",
        "genomic_recontextualized": "warning",
        "suboptimal": "watch",
        "optimal": "strength",
        "insight": "strength",
    }
    return mapping.get(status, "neutral")


def _sort_key_for_flag(flag: str) -> int:
    order = {
        "abnormal": 0,
        "genomic_recontextualized": 1,
        "suboptimal": 2,
        "optimal": 3,
    }
    return order.get(flag, 4)


def _flatten_risk_variants(dna_results: Dict) -> List[Dict]:
    """Extract only risk-direction variants that show at least one risk allele."""
    variants = []
    for category, snps in dna_results.items():
        for snp in snps:
            if not snp.get("claim_ready", True):
                continue
            if snp.get("trait_direction") == "protective":
                continue
            if snp.get("zygosity") not in {"heterozygous", "homozygous_risk"}:
                continue
            item = dict(snp)
            item["category"] = category
            variants.append(item)

    def sort_key(item: Dict) -> Tuple[int, str, str]:
        return (
            0 if item.get("zygosity") == "homozygous_risk" else 1,
            item.get("category", ""),
            item.get("gene", ""),
        )

    return sorted(variants, key=sort_key)


def _flatten_protective_variants(dna_results: Dict) -> List[Dict]:
    """
    Extract protective-direction variants where the user carries the
    advantageous allele (risk_allele_count > 0 for protective entries).
    """
    variants = []
    for category, snps in dna_results.items():
        for snp in snps:
            if not snp.get("claim_ready", True):
                continue
            if snp.get("trait_direction") != "protective":
                continue
            # For protective entries, carrying the 'risk_allele' IS the advantage
            if snp.get("risk_allele_count", 0) == 0:
                continue
            item = dict(snp)
            item["category"] = category
            # Remap zygosity labels for positive framing
            if snp.get("zygosity") == "homozygous_risk":
                item["advantage_level"] = "full_advantage"
                item["advantage_label"] = "Full Genetic Advantage"
            elif snp.get("zygosity") == "heterozygous":
                item["advantage_level"] = "partial_advantage"
                item["advantage_label"] = "Partial Genetic Advantage"
            else:
                item["advantage_level"] = "carrier"
                item["advantage_label"] = "Carrier"
            variants.append(item)

    def sort_key(item: Dict) -> Tuple[int, str, str]:
        return (
            0 if item.get("advantage_level") == "full_advantage" else 1,
            item.get("category", ""),
            item.get("gene", ""),
        )

    return sorted(variants, key=sort_key)


def _build_metric_cards(mode: str, dna_results: Dict, blood_results: Dict, confidence_score: float, next_tests: List[Dict], cross_source_count: int) -> List[Dict]:
    dna_variants = sum(len(v) for v in dna_results.values()) if dna_results else 0
    biomarkers = blood_results.get("total_biomarkers", 0) if isinstance(blood_results, dict) else 0
    sources = 0
    if dna_results:
        sources += 1
    if blood_results:
        sources += 1

    return [
        {"label": "Sources Reviewed", "value": str(sources), "caption": "DNA, biomarkers, or both depending on mode."},
        {"label": "Confidence", "value": f"{round(confidence_score)} / 100", "caption": f"{_confidence_label(confidence_score)} confidence overall."},
        {"label": "Priority Signals", "value": str(cross_source_count or _estimate_priority_signals(blood_results)), "caption": "Items most worth clarifying or following up."},
        {"label": "Suggested Next Tests", "value": str(len(next_tests)), "caption": "Companion tests that would sharpen interpretation."},
        {"label": "Genomic Variants Reviewed", "value": str(dna_variants), "caption": "Clinically relevant panel matches found in the upload." if dna_results else "Not included in this mode."},
        {"label": "Biomarkers Parsed", "value": str(biomarkers), "caption": "Measured biomarkers successfully normalized into the internal schema." if blood_results else "Not included in this mode."},
    ]


def _estimate_priority_signals(blood_results: Dict) -> int:
    if not isinstance(blood_results, dict):
        return 0
    flagged = blood_results.get("flagged", {})
    return len(flagged.get("abnormal", [])) + len(flagged.get("genomic_recontextualized", []))


def _build_source_overview(mode: str, dna_results: Dict, blood_results: Dict) -> List[Dict]:
    cards = []
    if dna_results:
        variants = sum(len(v) for v in dna_results.values())
        risk_variants = len(_flatten_risk_variants(dna_results))
        cards.append({
            "title": "Genomic Input",
            "body": f"{variants} curated variants were reviewed across {len(dna_results)} categories, with {risk_variants} risk variants currently expressed in the upload.",
        })
    if blood_results:
        flagged = blood_results.get("flagged", {})
        cards.append({
            "title": "Biomarker Input",
            "body": (
                f"{blood_results.get('total_biomarkers', 0)} biomarkers were normalized. "
                f"{len(flagged.get('abnormal', []))} are outside standard range and "
                f"{len(flagged.get('suboptimal', []))} are suboptimal against tighter targets."
            ),
        })
    if mode != "combined":
        cards.append({
            "title": "Cross-Source Scope",
            "body": "This run only used one data source, so any causal interpretation should be treated more cautiously than a combined analysis.",
        })
    return cards


def _build_data_quality(mode: str, conditions, clinical_confidence: Optional[Dict], dna_results: Dict, blood_results: Dict) -> Dict:
    strengths = []
    limitations = []

    if dna_results:
        strengths.append("Risk variants were matched against a curated SNP panel rather than using free-text interpretation alone.")
    if blood_results:
        strengths.append("Biomarkers were normalized into named fields with standard and optimal ranges before report generation.")
    if mode == "combined":
        strengths.append("Two independent data sources were available, which improves the credibility of converging signals.")
    else:
        limitations.append("This is a single-source analysis, so cross-source confirmation was not available.")

    if conditions:
        if conditions.fasting_hours >= 8:
            strengths.append("Collection context indicates a standard fasting window, which improves interpretation quality for metabolic markers.")
        else:
            limitations.append("Collection context suggests a nonstandard fasting window, which can shift glucose, insulin, and triglycerides.")

        if conditions.sleep_hours_prior < 6:
            limitations.append("Short sleep before collection can temporarily shift metabolic and hormonal markers.")
    else:
        limitations.append("Pre-test conditions were not captured, so context-sensitive biomarkers should be interpreted more cautiously.")

    if not blood_results:
        limitations.append("No biomarker trend data was available, so this remains a point-in-time snapshot.")
    else:
        limitations.append("Even reliable biomarkers are stronger when repeated over time instead of interpreted from a single draw.")

    if not dna_results:
        limitations.append("No genotype context was available for recontextualizing borderline biomarker values.")

    score = clinical_confidence.get("combined_confidence", 58) if clinical_confidence else (72 if mode == "blood" else 55)
    return {
        "score": score,
        "label": _confidence_label(score),
        "strengths": strengths[:4],
        "limitations": limitations[:4],
    }


def _avg(values: List[float], default: float = 50.0) -> float:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return default
    return round(sum(clean) / len(clean), 1)


def _flatten_all_variants(dna_results: Dict) -> List[Dict]:
    variants = []
    for category, snps in dna_results.items():
        for snp in snps:
            item = dict(snp)
            item["category"] = category
            variants.append(item)

    def sort_key(item: Dict) -> Tuple[str, str, str]:
        return (
            item.get("category", ""),
            item.get("gene", ""),
            item.get("rsid", ""),
        )

    return sorted(variants, key=sort_key)


def _build_biomarker_dossiers(blood_results: Dict, adjusted_biomarkers: Dict) -> Tuple[List[Dict], List[Dict]]:
    if not isinstance(blood_results, dict):
        return [], []

    evaluated = list(blood_results.get("evaluated", {}).values())
    evaluated.sort(key=lambda item: (_sort_key_for_flag(item.get("flag")), item.get("category", ""), item.get("name", "")))

    dossiers = []
    matrix = []

    for result in evaluated:
        db_key = result.get("db_key")
        reliability = get_biomarker_reliability_report(db_key) if db_key else None
        adjustment = adjusted_biomarkers.get(db_key or "", {})

        certainty = _avg([
            reliability.get("confidence_score") if reliability else None,
            adjustment.get("confidence_score"),
        ], default=60.0)

        caveats = []
        if result.get("optimal_status") != "optimal" and result.get("standard_status") == "within_standard":
            caveats.append("This signal is outside a tighter optimization target, not necessarily outside conventional lab reference range.")
        if reliability and reliability.get("confidence_score", 0) < 60:
            caveats.append(reliability.get("interpretation"))
        caveats.extend(adjustment.get("caveats", []))
        if adjustment.get("adjustment_factors"):
            caveats.append("Collection-condition adjustment factors were applied: " + ", ".join(adjustment["adjustment_factors"]))

        next_tests = [
            {"name": name, "reason": reason}
            for name, reason in BIOMARKER_NEXT_TESTS.get(db_key, [])
        ]
        if result.get("genomic_actions"):
            for action in result["genomic_actions"][:2]:
                next_tests.append({"name": "Contextual follow-up", "reason": action})

        summary = f"{result['name']} measured {result['value']} {result['unit']}."
        if result.get("flag") == "abnormal":
            summary += " This is outside the standard reference range."
        elif result.get("flag") == "genomic_recontextualized":
            summary += " Standard reference interpretation is softened because genotype suggests the threshold should be stricter."
        elif result.get("flag") == "suboptimal":
            summary += " This is within common lab range but outside the tighter range used for optimization-focused interpretation."
        else:
            summary += " This currently sits in the preferred range."

        provenance = [
            {"label": "Measured value", "value": f"{result['value']} {result['unit']}"},
            {"label": "Standard range", "value": result.get("standard_range", "N/A")},
            {"label": "Optimal range", "value": result.get("optimal_range", "N/A")},
            {"label": "Lab reference", "value": result.get("reference_range_lab", "N/A")},
        ]
        if reliability:
            provenance.append({"label": "Reliability score", "value": f"{reliability['confidence_score']} / 100"})

        evidence_links = get_biomarker_evidence_links(
            db_key or result["name"],
            clinical_name=result.get("name"),
            category=result.get("category"),
        )

        dossiers.append({
            "id": db_key,
            "title": result["name"],
            "category": result.get("category", "General"),
            "theme": _theme_for_status(result.get("flag")),
            "status": result.get("flag", "unknown").replace("_", " ").title(),
            "certainty_score": certainty,
            "certainty_label": _confidence_label(certainty),
            "summary": summary,
            "interpretation": result.get("interpretation"),
            "context": result.get("genomic_context") or CATEGORY_EXPLAINERS.get(result.get("category"), "Interpretation depends on repeat testing, collection context, and companion markers."),
            "caveats": caveats[:4],
            "what_changes": [
                "A repeat test under consistent collection conditions would strengthen confidence.",
                "Companion biomarkers are often more informative than a single number in isolation.",
            ],
            "next_tests": next_tests[:4],
            "provenance": provenance,
            "evidence_links": evidence_links,
        })

        matrix.append({
            "name": result["name"],
            "category": result.get("category", "General"),
            "value": f"{result['value']} {result['unit']}",
            "flag": result.get("flag", "unknown"),
            "theme": _theme_for_status(result.get("flag")),
            "standard_range": result.get("standard_range", "N/A"),
            "optimal_range": result.get("optimal_range", "N/A"),
            "links": evidence_links[:2],
        })

    return dossiers[:8], matrix


def _build_genomic_dossiers(dna_results: Dict) -> List[Dict]:
    variants = _flatten_risk_variants(dna_results)
    dossiers = []

    for variant in variants[:8]:
        rsid = variant.get("rsid")
        context = get_clinical_context(rsid) if rsid else None
        certainty = context.get("confidence_score", 42) if context else 42
        next_tests = [
            {"name": name, "reason": reason}
            for name, reason in GENE_FOLLOW_UP.get(variant.get("gene"), [])
        ]

        evidence = []
        if context:
            for item in context.get("evidence", [])[:2]:
                evidence.append({
                    "label": f"{item['journal']} {item['year']} ({item['evidence_level']['code']})",
                    "title": item["title"],
                    "url": item.get("url"),
                })

        evidence_links = get_snp_reference_links(
            rsid,
            variant.get("gene"),
            variant.get("trait"),
            context.get("evidence", []) if context else None,
        )

        dossiers.append({
            "id": rsid,
            "title": f"{variant['gene']} {variant['trait']}",
            "category": variant.get("category", "Genomic"),
            "theme": _theme_for_status("critical" if variant.get("zygosity") == "homozygous_risk" else "warning"),
            "status": variant.get("zygosity", "").replace("_", " ").title(),
            "certainty_score": certainty,
            "certainty_label": _confidence_label(certainty),
            "summary": f"{variant['gene']} {rsid} is present as {variant['zygosity'].replace('_', ' ')} with genotype {variant['genotype']}.",
            "interpretation": variant.get("effect"),
            "context": context.get("confidence_description") if context else CATEGORY_EXPLAINERS.get(variant.get("category"), "This is a predisposition signal and should be paired with phenotype data before making strong claims."),
            "caveats": [
                "Genetic variants describe risk predisposition, not certainty.",
                "Penetrance depends on environment, behavior, and other biology.",
            ],
            "what_changes": [
                "This variant becomes more actionable when paired with its downstream biomarker phenotype.",
                "Repeat biomarker testing and symptom context are more important than genotype alone.",
            ],
            "next_tests": next_tests[:3],
            "provenance": [
                {"label": "Variant", "value": rsid},
                {"label": "Gene", "value": variant["gene"]},
                {"label": "Genotype", "value": variant["genotype"]},
                {"label": "Risk allele", "value": variant["risk_allele"]},
                {"label": "Evidence score", "value": f"{certainty} / 100" if context else "Curated annotation unavailable"},
            ],
            "evidence": evidence,
            "evidence_links": evidence_links,
        })

    return dossiers


def _build_genetic_advantages(dna_results: Dict) -> List[Dict]:
    """Build dossier cards for positive/protective trait variants."""
    variants = _flatten_protective_variants(dna_results)
    dossiers = []

    for variant in variants:
        rsid = variant.get("rsid")
        context = get_clinical_context(rsid) if rsid else None
        certainty = context.get("confidence_score", 42) if context else 42

        evidence_links = get_snp_reference_links(
            rsid,
            variant.get("gene"),
            variant.get("trait"),
            context.get("evidence", []) if context else None,
        )

        advantage_level = variant.get("advantage_level", "partial_advantage")
        advantage_label = variant.get("advantage_label", "Genetic Advantage")

        dossiers.append({
            "id": rsid,
            "title": f"{variant['gene']} {variant['trait']}",
            "category": variant.get("category", "Genetic Advantage"),
            "theme": "strength",
            "status": advantage_label,
            "advantage_level": advantage_level,
            "certainty_score": certainty,
            "certainty_label": _confidence_label(certainty),
            "summary": f"{variant['gene']} {rsid} — your genotype {variant['genotype']} carries the advantageous allele.",
            "interpretation": variant.get("effect"),
            "context": "This is a positive genetic signal — it indicates an inherent biological advantage that may support this aspect of your health.",
            "provenance": [
                {"label": "Variant", "value": rsid},
                {"label": "Gene", "value": variant["gene"]},
                {"label": "Genotype", "value": variant["genotype"]},
                {"label": "Advantageous allele", "value": variant["risk_allele"]},
                {"label": "Copies", "value": str(variant.get("risk_allele_count", 0))},
            ],
            "evidence_links": evidence_links,
        })

    return dossiers


def _build_cross_source_dossiers(scored_findings: List[Dict], pattern_results: Optional[Dict], blood_results: Dict) -> List[Dict]:
    if not scored_findings:
        return []

    pattern_map = {}
    for item in (pattern_results or {}).get("hardcoded_patterns", []):
        pattern = item.get("pattern", {})
        pattern_map[pattern.get("id")] = item

    evaluated = blood_results.get("evaluated", {}) if isinstance(blood_results, dict) else {}
    dossiers = []

    for finding in scored_findings:
        match = pattern_map.get(finding.get("finding_id"), {})
        pattern = match.get("pattern", {})
        required_biomarkers = list(pattern.get("required_biomarkers", {}).keys())
        provenance = []

        for rsid in match.get("triggered_snps", [])[:3]:
            provenance.append({"label": "Triggered SNP", "value": rsid})
        for biomarker in required_biomarkers[:3]:
            measured = next(
                (item for item in evaluated.values() if item.get("db_key") == biomarker),
                None,
            )
            if measured:
                provenance.append({
                    "label": measured["name"],
                    "value": f"{measured['value']} {measured['unit']} ({measured['flag'].replace('_', ' ')})",
                })

        caveats = []
        if not match.get("has_biomarker_confirmation"):
            caveats.append("This signal is more genotype-led than phenotype-confirmed.")
        if match and match.get("total_optional", 0) > 0 and match.get("optional_matches", 0) == 0:
            caveats.append("Optional reinforcing markers were not present, so this should be interpreted as an early signal rather than a settled conclusion.")
        if not caveats:
            caveats.append("This pattern is stronger than a single marker because multiple inputs pointed in the same direction.")

        next_tests = []
        for biomarker in pattern.get("optional_biomarkers", {}).keys():
            if not any(item.get("db_key") == biomarker for item in evaluated.values()):
                pretty = biomarker.replace("_", " ").title()
                next_tests.append({"name": pretty, "reason": "This optional companion marker would strengthen or soften the cross-source interpretation."})

        evidence_links = []
        for rsid in match.get("triggered_snps", [])[:2]:
            evidence_links.extend(get_snp_reference_links(rsid))
        for biomarker in required_biomarkers[:2]:
            measured = next(
                (item for item in evaluated.values() if item.get("db_key") == biomarker),
                None,
            )
            if measured:
                evidence_links.extend(
                    get_biomarker_evidence_links(
                        biomarker,
                        clinical_name=measured.get("name"),
                        category=measured.get("category"),
                    )[:2]
                )

        deduped_links = []
        seen_urls = set()
        for link in evidence_links:
            url = link.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped_links.append(link)

        dossiers.append({
            "id": finding.get("finding_id"),
            "title": finding.get("name"),
            "category": finding.get("category"),
            "theme": _theme_for_status(finding.get("severity")),
            "status": finding.get("action_priority", {}).get("level", "Watch"),
            "certainty_score": finding.get("confidence", 55),
            "certainty_label": _confidence_label(finding.get("confidence", 55)),
            "summary": finding.get("description"),
            "interpretation": finding.get("implications"),
            "context": f"Composite risk score {finding.get('composite_risk_score', 'N/A')} / 100. {match.get('strength', 'Cross-source review completed.')}" if match else f"Composite risk score {finding.get('composite_risk_score', 'N/A')} / 100.",
            "caveats": caveats[:3],
            "what_changes": [
                "Confidence increases when trend data or companion biomarkers confirm the same direction of signal.",
                "Signals that disappear on repeat testing should be downgraded rather than narrated more confidently.",
            ],
            "next_tests": next_tests[:4],
            "provenance": provenance[:6],
            "actions": finding.get("actions", [])[:4],
            "evidence_links": deduped_links[:6],
        })

    return dossiers


def _build_variant_matrix(dna_results: Dict) -> List[Dict]:
    rows = []
    for variant in _flatten_all_variants(dna_results):
        rsid = variant.get("rsid")
        context = get_clinical_context(rsid) if rsid else None
        score = context.get("confidence_score", 42) if context else 42

        if variant.get("zygosity") == "homozygous_risk":
            theme = _theme_for_status("critical")
        elif variant.get("zygosity") == "heterozygous":
            theme = _theme_for_status("warning")
        elif variant.get("claim_ready"):
            theme = "neutral"
        else:
            theme = "watch"

        rows.append({
            "rsid": rsid,
            "gene": variant.get("gene"),
            "trait": variant.get("trait"),
            "category": variant.get("category", "Genomic"),
            "genotype": variant.get("genotype"),
            "zygosity": variant.get("zygosity", "").replace("_", " "),
            "claim_ready": variant.get("claim_ready", False),
            "validation_status": variant.get("validation_status", "unknown").replace("_", " "),
            "validation_notes": variant.get("validation_notes", []),
            "evidence_score": score,
            "theme": theme,
            "links": get_snp_reference_links(
                rsid,
                variant.get("gene"),
                variant.get("trait"),
                context.get("evidence", []) if context else None,
            )[:3],
        })

    return rows


def _build_genomic_panel_summary(dna_results: Dict) -> Dict:
    all_variants = _flatten_all_variants(dna_results)
    matched = len(all_variants)
    risk_variants = len(_flatten_risk_variants(dna_results))
    protective_variants = len(_flatten_protective_variants(dna_results))
    return {
        "panel_size": len(SNP_PANEL),
        "matched_variants": matched,
        "risk_variants": risk_variants,
        "protective_variants": protective_variants,
        "note": (
            "HealthBridge now reviews a broad curated SNP panel, but it still should not be described as "
            "exhaustive. Clinical genetics evolves constantly, and associations vary in strength."
        ),
    }


def _build_dna_quality_context(dna_qc: Optional[Dict], dna_results: Dict) -> Optional[Dict]:
    if not dna_qc:
        return None

    parse = dna_qc.get("source_parse_summary", {})
    claim_ready = dna_qc.get("claim_ready_variants", 0)
    suppressed = dna_qc.get("suppressed_variants", 0)
    matched = dna_qc.get("matched_panel_variants", 0)

    strengths = []
    limitations = []

    if claim_ready:
        strengths.append(f"{claim_ready} matched panel variants survived strict dbSNP-backed validation.")
    if dna_qc.get("validated_complemented_calls"):
        strengths.append(
            f"{dna_qc['validated_complemented_calls']} matched calls were reconciled through non-ambiguous reverse-complement handling."
        )
    if dna_qc.get("risk_allele_complemented"):
        strengths.append(
            f"{dna_qc['risk_allele_complemented']} curated risk alleles were corrected to the dbSNP forward orientation before interpretation."
        )

    if dna_qc.get("palindromic_suppressed"):
        limitations.append(
            f"{dna_qc['palindromic_suppressed']} A/T or C/G variants were suppressed because strand cannot be proven from raw alleles alone."
        )
    if dna_qc.get("missing_authoritative_metadata"):
        limitations.append(
            f"{dna_qc['missing_authoritative_metadata']} matched variants lacked sufficient authoritative metadata and were not used for strong claims."
        )
    if dna_qc.get("validation_failures"):
        limitations.append(
            f"{dna_qc['validation_failures']} matched variants were rejected because the raw alleles did not reconcile cleanly with dbSNP."
        )

    return {
        "headline": "DNA quality controls",
        "summary": (
            f"{matched} panel matches were found in the upload. "
            f"{claim_ready} are claim-ready under strict validation and {suppressed} were suppressed."
        ),
        "parse_metrics": [
            {"label": "Retained panel rsIDs", "value": f"{parse.get('unique_rsids', 0):,}"},
            {"label": "Genotype rows scanned", "value": f"{parse.get('parsed_rows', 0):,}"},
            {"label": "Filtered non-panel rows", "value": f"{parse.get('filtered_non_panel_rows', 0):,}"},
            {"label": "No-calls skipped", "value": f"{parse.get('no_call_rows', 0):,}"},
            {"label": "Malformed rows", "value": f"{parse.get('malformed_rows', 0):,}"},
        ],
        "strengths": strengths[:4],
        "limitations": limitations[:4],
        "strict_mode": dna_qc.get("strict_mode", True),
        "source_hint": parse.get("source_hint", "unknown"),
    }


def _build_next_tests(*collections: List[Dict]) -> List[Dict]:
    deduped = {}
    for collection in collections:
        for item in collection:
            for next_test in item.get("next_tests", []):
                name = next_test.get("name")
                if not name:
                    continue
                if name not in deduped:
                    deduped[name] = next_test
    return list(deduped.values())[:10]


def _build_key_takeaways(cross_source_dossiers: List[Dict], biomarker_dossiers: List[Dict], genomic_dossiers: List[Dict], data_quality: Dict) -> List[Dict]:
    takeaways = []

    if cross_source_dossiers:
        top = cross_source_dossiers[0]
        takeaways.append({
            "title": top["title"],
            "body": top["summary"],
            "theme": top["theme"],
        })
    if biomarker_dossiers:
        top = biomarker_dossiers[0]
        takeaways.append({
            "title": top["title"],
            "body": top["interpretation"],
            "theme": top["theme"],
        })
    if genomic_dossiers:
        top = genomic_dossiers[0]
        takeaways.append({
            "title": top["title"],
            "body": top["context"],
            "theme": top["theme"],
        })

    takeaways.append({
        "title": "Data Quality Readout",
        "body": f"{data_quality['label']} confidence overall. " + " ".join(data_quality["limitations"][:1]),
        "theme": "neutral",
    })

    return takeaways[:4]


def _build_report_summary(mode: str, cross_source_dossiers: List[Dict], biomarker_dossiers: List[Dict], genomic_dossiers: List[Dict], data_quality: Dict) -> Dict:
    if cross_source_dossiers:
        headline = f"{len(cross_source_dossiers)} converging signals surfaced across your uploaded data."
        subheadline = "The strongest outputs are the patterns where biomarkers and genotype point in the same direction."
    elif biomarker_dossiers:
        headline = f"{len(biomarker_dossiers)} biomarker findings were prioritized for review."
        subheadline = "The report is separating standard out-of-range values from tighter optimization targets."
    elif genomic_dossiers:
        headline = f"{len(genomic_dossiers)} risk variants were prioritized for review."
        subheadline = "These are predisposition signals and should be paired with phenotype data before making strong conclusions."
    else:
        headline = "Your uploaded data was processed successfully."
        subheadline = "No major structured findings rose above the current reporting threshold."

    return {
        "eyebrow": _mode_label(mode),
        "headline": headline,
        "subheadline": subheadline,
        "confidence_line": f"Overall report confidence: {data_quality['score']} / 100 ({data_quality['label']}).",
    }


def build_report_context(
    mode: str,
    dna_results: Optional[Dict] = None,
    blood_results: Optional[Dict] = None,
    dna_qc: Optional[Dict] = None,
    conditions=None,
    clinical_confidence: Optional[Dict] = None,
    adjusted_biomarkers: Optional[Dict] = None,
    scored_findings: Optional[List[Dict]] = None,
    pattern_results: Optional[Dict] = None,
    wearable_metrics: Optional[Dict] = None,
) -> Dict:
    dna_results = dna_results or {}
    blood_results = blood_results or {}
    adjusted_biomarkers = adjusted_biomarkers or {}
    scored_findings = scored_findings or []

    wearable_context = build_wearable_context(wearable_metrics)

    data_quality = _build_data_quality(mode, conditions, clinical_confidence, dna_results, blood_results)
    biomarker_dossiers, biomarker_matrix = _build_biomarker_dossiers(blood_results, adjusted_biomarkers)
    genomic_dossiers = _build_genomic_dossiers(dna_results)
    genetic_advantages = _build_genetic_advantages(dna_results)
    cross_source_dossiers = _build_cross_source_dossiers(scored_findings, pattern_results, blood_results)
    variant_matrix = _build_variant_matrix(dna_results)
    genomic_panel_summary = _build_genomic_panel_summary(dna_results)
    dna_quality = _build_dna_quality_context(dna_qc, dna_results)
    phenotype_sketch = build_fun_phenotype_cards(dna_results)
    population_notes = build_population_notes(dna_results)
    next_tests = _build_next_tests(biomarker_dossiers, genomic_dossiers, cross_source_dossiers)
    metric_cards = _build_metric_cards(mode, dna_results, blood_results, data_quality["score"], next_tests, len(cross_source_dossiers))

    return {
        "engine_version": ENGINE_VERSION,
        "mode_label": _mode_label(mode),
        "report_summary": _build_report_summary(mode, cross_source_dossiers, biomarker_dossiers, genomic_dossiers, data_quality),
        "metric_cards": metric_cards,
        "trust_markers": TRUST_MARKERS,
        "source_overview": _build_source_overview(mode, dna_results, blood_results),
        "data_quality": data_quality,
        "key_takeaways": _build_key_takeaways(cross_source_dossiers, biomarker_dossiers, genomic_dossiers, data_quality),
        "cross_source_dossiers": cross_source_dossiers,
        "biomarker_dossiers": biomarker_dossiers,
        "genomic_dossiers": genomic_dossiers,
        "genetic_advantages": genetic_advantages,
        "biomarker_matrix": biomarker_matrix,
        "variant_matrix": variant_matrix,
        "genomic_panel_summary": genomic_panel_summary,
        "dna_quality": dna_quality,
        "phenotype_sketch": phenotype_sketch,
        "population_notes": population_notes,
        "next_tests": next_tests,
        "methodology_sections": METHODOLOGY_SECTIONS,
        "wearable_context": wearable_context,
        "cross_source_empty_state": "No hardcoded cross-source pattern reached threshold in this run. That is useful information: it means the engine did not force a convergence story where the inputs did not support one.",
    }


def build_wearable_context(wearable_metrics: Optional[Dict]) -> Optional[Dict]:
    """
    Transform raw wearable_parser output into a rich template-ready context dict.
    Returns None if no wearable data was parsed.
    """
    if not wearable_metrics:
        return None

    platform = wearable_metrics.get("platform") or "Unknown"
    dr = wearable_metrics.get("date_range", {})
    sleep = wearable_metrics.get("sleep", {})
    hrv = wearable_metrics.get("hrv", {})
    hr = wearable_metrics.get("heart_rate", {})
    activity = wearable_metrics.get("activity", {})
    recovery = wearable_metrics.get("recovery", {})
    flags = wearable_metrics.get("flags", [])

    # Nothing useful was parsed
    has_data = any([
        sleep.get("nights_analyzed"),
        hrv.get("avg_rmssd"),
        hr.get("avg_resting_hr"),
        activity.get("avg_daily_steps"),
    ])
    if not has_data:
        return None

    date_range_label = ""
    if dr.get("start") and dr.get("end"):
        date_range_label = f"{dr['start']} — {dr['end']} ({dr.get('days', 0)} days)"

    # ── Sleep summary cards ────────────────────────────────────────────────────
    sleep_cards = []
    if sleep.get("avg_total_hours"):
        optimal = 7.5 <= sleep["avg_total_hours"] <= 9.0
        sleep_cards.append({
            "label": "Avg Total Sleep",
            "value": f"{sleep['avg_total_hours']}h",
            "optimal": optimal,
            "note": "Optimal: 7.5–9h" if not optimal else "Within optimal range",
        })
    if sleep.get("avg_deep_hours"):
        optimal = sleep["avg_deep_hours"] >= 1.5
        sleep_cards.append({
            "label": "Avg Deep Sleep",
            "value": f"{sleep['avg_deep_hours']}h",
            "optimal": optimal,
            "note": "Optimal: ≥1.5h" if not optimal else "Within optimal range",
        })
    if sleep.get("avg_rem_hours"):
        optimal = sleep["avg_rem_hours"] >= 1.5
        sleep_cards.append({
            "label": "Avg REM Sleep",
            "value": f"{sleep['avg_rem_hours']}h",
            "optimal": optimal,
            "note": "Optimal: ≥1.5h" if not optimal else "Within optimal range",
        })
    if sleep.get("avg_efficiency_pct"):
        optimal = sleep["avg_efficiency_pct"] >= 85
        sleep_cards.append({
            "label": "Sleep Efficiency",
            "value": f"{sleep['avg_efficiency_pct']}%",
            "optimal": optimal,
            "note": "Optimal: ≥85%" if not optimal else "Within optimal range",
        })
    if sleep.get("avg_latency_min"):
        optimal = 10 <= sleep["avg_latency_min"] <= 20
        sleep_cards.append({
            "label": "Sleep Latency",
            "value": f"{sleep['avg_latency_min']} min",
            "optimal": optimal,
            "note": "Optimal: 10–20 min" if not optimal else "Within optimal range",
        })

    # ── HRV + Heart rate cards ─────────────────────────────────────────────────
    biometric_cards = []
    if hrv.get("avg_rmssd"):
        optimal = hrv["avg_rmssd"] >= 40
        trend = hrv.get("hrv_trend", "stable")
        biometric_cards.append({
            "label": "Avg HRV (RMSSD)",
            "value": f"{hrv['avg_rmssd']} ms",
            "optimal": optimal,
            "note": f"Trend: {trend} · Higher is better; age-dependent",
        })
    if hr.get("avg_resting_hr"):
        optimal = hr["avg_resting_hr"] <= 60
        trend = hr.get("resting_hr_trend", "stable")
        biometric_cards.append({
            "label": "Avg Resting HR",
            "value": f"{hr['avg_resting_hr']} bpm",
            "optimal": optimal,
            "note": f"Trend: {trend} · Optimal: <60 bpm",
        })
    if hr.get("min_resting_hr"):
        biometric_cards.append({
            "label": "Lowest Resting HR",
            "value": f"{hr['min_resting_hr']} bpm",
            "optimal": hr["min_resting_hr"] <= 55,
            "note": "Lowest recorded overnight value",
        })

    # ── Activity cards ─────────────────────────────────────────────────────────
    activity_cards = []
    if activity.get("avg_daily_steps"):
        optimal = activity["avg_daily_steps"] >= 7500
        activity_cards.append({
            "label": "Avg Daily Steps",
            "value": f"{activity['avg_daily_steps']:,}",
            "optimal": optimal,
            "note": "Optimal: ≥7,500 steps/day",
        })
    if activity.get("avg_active_calories"):
        activity_cards.append({
            "label": "Avg Active Calories",
            "value": f"{activity['avg_active_calories']} kcal",
            "optimal": activity["avg_active_calories"] >= 400,
            "note": "Target varies by size and goals",
        })
    if activity.get("avg_activity_score"):
        optimal = activity["avg_activity_score"] >= 75
        activity_cards.append({
            "label": "Avg Activity Score",
            "value": str(activity["avg_activity_score"]),
            "optimal": optimal,
            "note": "Platform score (0–100)",
        })
    if activity.get("sedentary_days_pct") is not None:
        pct = activity["sedentary_days_pct"]
        activity_cards.append({
            "label": "Sedentary Days",
            "value": f"{pct}%",
            "optimal": pct <= 20,
            "note": "Days with <5,000 steps",
        })
    if activity.get("high_activity_days_pct") is not None:
        pct = activity["high_activity_days_pct"]
        activity_cards.append({
            "label": "High Activity Days",
            "value": f"{pct}%",
            "optimal": pct >= 30,
            "note": "Days with ≥10,000 steps",
        })

    # ── Recovery ───────────────────────────────────────────────────────────────
    recovery_cards = []
    readiness = recovery.get("avg_readiness_score") or recovery.get("avg_recovery_score")
    if readiness:
        optimal = readiness >= 75
        recovery_cards.append({
            "label": "Avg Recovery / Readiness",
            "value": str(readiness),
            "optimal": optimal,
            "note": "Platform score (0–100). Optimal: ≥75",
        })
    if recovery.get("low_recovery_days_pct") is not None:
        pct = recovery["low_recovery_days_pct"]
        recovery_cards.append({
            "label": "Low Recovery Days",
            "value": f"{pct}%",
            "optimal": pct <= 15,
            "note": "Days with readiness/recovery <60",
        })
    if recovery.get("high_recovery_days_pct") is not None:
        pct = recovery["high_recovery_days_pct"]
        recovery_cards.append({
            "label": "High Recovery Days",
            "value": f"{pct}%",
            "optimal": pct >= 40,
            "note": "Days with readiness/recovery ≥80",
        })

    # ── Overall wearable score (simple average of key metrics) ────────────────
    score_components = []
    if sleep.get("avg_total_hours"):
        h = sleep["avg_total_hours"]
        score_components.append(min(100, max(0, (h / 8.0) * 100)) if h <= 9 else 90)
    if hrv.get("avg_rmssd"):
        score_components.append(min(100, hrv["avg_rmssd"] * 2))
    if hr.get("avg_resting_hr"):
        rhr = hr["avg_resting_hr"]
        score_components.append(max(0, min(100, (80 - rhr) / 20 * 100)))
    if activity.get("avg_daily_steps"):
        score_components.append(min(100, activity["avg_daily_steps"] / 100))
    if readiness:
        score_components.append(readiness)

    overall_score = round(sum(score_components) / len(score_components)) if score_components else 0
    overall_score = max(0, min(100, overall_score))

    if overall_score >= 80:
        overall_status, overall_color = "Excellent", "#10b981"
    elif overall_score >= 65:
        overall_status, overall_color = "Good", "#3b82f6"
    elif overall_score >= 45:
        overall_status, overall_color = "Fair", "#f59e0b"
    else:
        overall_status, overall_color = "Needs Attention", "#dc2626"

    # ── Flags → insight cards ──────────────────────────────────────────────────
    warning_flags = [f for f in flags if f.get("type") == "warning"]
    insight_flags = [f for f in flags if f.get("type") == "insight"]

    return {
        "platform": platform,
        "date_range_label": date_range_label,
        "nights_analyzed": sleep.get("nights_analyzed", 0),
        "days_analyzed": activity.get("days_analyzed", 0),
        "overall_score": overall_score,
        "overall_status": overall_status,
        "overall_color": overall_color,
        "sleep_cards": sleep_cards,
        "biometric_cards": biometric_cards,
        "activity_cards": activity_cards,
        "recovery_cards": recovery_cards,
        "warning_flags": warning_flags[:6],
        "insight_flags": insight_flags[:4],
        "raw": wearable_metrics,
    }


def build_methodology_context() -> Dict:
    return {
        "engine_version": ENGINE_VERSION,
        "methodology_sections": METHODOLOGY_SECTIONS,
        "trust_markers": TRUST_MARKERS,
        "guardrails": [
            "Observations are separated from interpretation and next-step guidance.",
            "Single-point measurements are never presented as equivalent to trend data.",
            "Genetic findings are framed as predisposition signals, not diagnoses.",
            "Collection context and biomarker variability are treated as first-class uncertainty inputs.",
        ],
    }
