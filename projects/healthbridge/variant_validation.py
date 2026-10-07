"""
HealthBridge AI - Genotype parsing and authoritative SNP validation

This module makes consumer raw DNA interpretation more conservative by:
1. parsing large raw genotype files in a vendor-tolerant way,
2. validating matched SNPs against authoritative dbSNP RefSNP metadata,
3. suppressing claims when strand/orientation is ambiguous or unsupported.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


VALID_BASES = {"A", "C", "G", "T"}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
PALINDROMIC_ALLELE_SETS = {frozenset({"A", "T"}), frozenset({"C", "G"})}
NCBI_REFSNP_URL = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{rsid}"

# Sources that document plus-strand reporting — palindromic SNPs can be
# resolved as forward-strand for these vendors.
PLUS_STRAND_SOURCES: set[str] = {"ancestrydna", "23andme"}
CACHE_PATH = Path(__file__).resolve().parent / "variant_metadata_cache.json"


@dataclass
class GenotypeParseSummary:
    source_hint: str = "unknown"
    total_lines: int = 0
    comment_lines: int = 0
    header_lines: int = 0
    parsed_rows: int = 0
    malformed_rows: int = 0
    invalid_rows: int = 0
    no_call_rows: int = 0
    filtered_non_panel_rows: int = 0
    duplicate_rsids: int = 0
    conflicting_duplicates: int = 0
    unique_rsids: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def canonical_panel_rsid(panel_key: str) -> str:
    return panel_key.lower().split("_", 1)[0]


def canonical_rsid(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return ""
    if cleaned.startswith("rs"):
        return cleaned
    return f"rs{cleaned}"


def complement_base(base: str) -> str:
    return COMPLEMENT.get(base.upper(), base.upper())


def complement_set(values: Iterable[str]) -> set[str]:
    return {complement_base(value) for value in values}


def get_primary_allele_pair(authoritative_meta: Dict) -> set[str]:
    """Extract the clinically relevant ref + primary alt allele pair.

    Many dbSNP entries list 3 or 4 observed alleles because of rare
    multi-allelic observations.  Using the full set for validation
    lets clinically irrelevant alleles pass (e.g. rs6025 where A is
    listed but the Leiden mutation is C→T).

    This function returns only {ref, primary_alt} so downstream
    validation is constrained to the clinically studied allele pair.
    If the ref or primary alt cannot be determined, falls back to the
    full observed set.
    """
    ref = (authoritative_meta.get("reference_allele") or "").upper()
    observed = authoritative_meta.get("observed_alleles", [])

    if not ref or ref not in set(observed):
        # Cannot determine primary pair — fall back to all observed
        return set(observed)

    alts = [a for a in observed if a != ref]
    if len(alts) == 1:
        # Simple bi-allelic — return {ref, alt}
        return {ref, alts[0]}

    if len(alts) == 0:
        return set(observed)

    # Multi-allelic: pick the most clinically significant alt.
    # Heuristic: prefer the complement of the ref (transition),
    # because transitions (A↔G, C↔T) are far more common than
    # transversions in clinically studied SNPs.
    transitions = {"A": "G", "G": "A", "C": "T", "T": "C"}
    transition_alt = transitions.get(ref)
    if transition_alt and transition_alt in alts:
        return {ref, transition_alt}

    # No transition found — return all observed as fallback
    return set(observed)


def _normalize_base(base: str) -> str:
    value = (base or "").strip().upper()
    if value in {"", "0", "-", "--", "?", "N", "NA", "NC"}:
        return "-"
    return value


def _normalize_alleles(raw_alleles: List[str]) -> Tuple[Tuple[str, str], str]:
    alleles = [_normalize_base(value) for value in raw_alleles[:2]]
    while len(alleles) < 2:
        alleles.append("-")

    if all(value == "-" for value in alleles):
        return ("-", "-"), "no_call"

    if any(value not in VALID_BASES and value != "-" for value in alleles):
        return tuple(alleles), "invalid"

    called = [value for value in alleles if value in VALID_BASES]
    if len(called) == 1:
        called = [called[0], called[0]]
    elif len(called) == 0:
        return ("-", "-"), "no_call"

    return tuple(sorted(called)), "ok"


def _split_fields(line: str) -> List[str]:
    if "\t" in line:
        return [part.strip() for part in line.rstrip("\n").split("\t")]
    if "," in line:
        return next(csv.reader([line]))
    return [part.strip() for part in line.rstrip("\n").split()]


def _extract_raw_alleles(parts: List[str]) -> Optional[List[str]]:
    if len(parts) >= 5:
        return [parts[3], parts[4]]

    if len(parts) == 4:
        combined = parts[3].strip().upper()
        if combined in {"", "--", "00", "0", "NC", "NA"}:
            return ["-", "-"]
        letters = [char for char in combined if char.isalpha()]
        if len(letters) >= 2:
            return [letters[0], letters[1]]
        if len(letters) == 1:
            return [letters[0], letters[0]]
    return None


def parse_raw_genotype_file(filepath: str, keep_only_rsids: Optional[set[str]] = None) -> Dict:
    """
    Parse a raw genotype file conservatively.

    Supports simple tab/comma/space-delimited layouts with rsid, chromosome,
    position, and either allele1/allele2 or a combined genotype column.
    """
    summary = GenotypeParseSummary()
    records = {}
    comment_buffer = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            summary.total_lines += 1
            line = raw_line.strip()
            if not line:
                continue

            lowered = line.lower()
            if line.startswith("#"):
                summary.comment_lines += 1
                comment_buffer.append(lowered)
                if "ancestrydna" in lowered:
                    summary.source_hint = "ancestrydna"
                elif "23andme" in lowered:
                    summary.source_hint = "23andme"
                elif "myheritage" in lowered:
                    summary.source_hint = "myheritage"
                elif "ftdna" in lowered or "familytreedna" in lowered:
                    summary.source_hint = "familytreedna"
                continue

            parts = _split_fields(raw_line)
            if not parts:
                continue

            header_tokens = {token.lower() for token in parts[:5]}
            if "rsid" in header_tokens and ("allele1" in header_tokens or "result" in header_tokens or "genotype" in header_tokens):
                summary.header_lines += 1
                continue

            rsid = canonical_rsid(parts[0])
            if not rsid.startswith("rs"):
                summary.malformed_rows += 1
                continue

            chromosome = parts[1].strip() if len(parts) > 1 else ""
            position = parts[2].strip() if len(parts) > 2 else ""
            raw_alleles = _extract_raw_alleles(parts)
            if raw_alleles is None:
                summary.malformed_rows += 1
                continue

            alleles, status = _normalize_alleles(raw_alleles)
            if status == "invalid":
                summary.invalid_rows += 1
                continue
            if status == "no_call":
                summary.no_call_rows += 1
                continue

            if keep_only_rsids is not None and rsid not in keep_only_rsids:
                summary.filtered_non_panel_rows += 1
                summary.parsed_rows += 1
                continue

            if rsid in records:
                summary.duplicate_rsids += 1
                if records[rsid]["alleles"] != alleles:
                    summary.conflicting_duplicates += 1
                continue

            records[rsid] = {
                "rsid": rsid,
                "chromosome": chromosome,
                "position": position,
                "alleles": alleles,
                "line_number": line_number,
            }
            summary.parsed_rows += 1

    summary.unique_rsids = len(records)
    return {
        "genotypes": records,
        "summary": summary.to_dict(),
    }


def load_variant_metadata_cache() -> Dict[str, Dict]:
    if not CACHE_PATH.exists():
        return {}

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_variant_metadata_cache(cache: Dict[str, Dict]) -> None:
    with CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def fetch_variant_metadata(rsid: str, session: Optional[requests.Session] = None) -> Optional[Dict]:
    canonical = canonical_rsid(rsid)
    numeric_rsid = canonical.lstrip("rs")
    client = session or requests.Session()

    response = client.get(
        NCBI_REFSNP_URL.format(rsid=numeric_rsid),
        timeout=20,
        headers={"User-Agent": "HealthBridgeAI/2026.04"},
    )
    response.raise_for_status()
    payload = response.json()

    placement = None
    for item in payload.get("primary_snapshot_data", {}).get("placements_with_allele", []):
        traits = item.get("placement_annot", {}).get("seq_id_traits_by_assembly", [])
        if item.get("is_ptlp") and any(trait.get("is_top_level") for trait in traits):
            placement = item
            break

    if not placement:
        return None

    observed = []
    reference = None
    for allele_item in placement.get("alleles", []):
        spdi = allele_item.get("allele", {}).get("spdi", {})
        deleted = spdi.get("deleted_sequence", "")
        inserted = spdi.get("inserted_sequence", "")
        if len(deleted) != 1 or len(inserted) != 1:
            continue
        deleted = deleted.upper()
        inserted = inserted.upper()
        if deleted not in VALID_BASES or inserted not in VALID_BASES:
            continue
        if deleted == inserted:
            reference = deleted
        observed.append(inserted)

    observed_alleles = sorted(set(observed))
    if len(observed_alleles) < 2:
        return None

    clinvar_significance = []
    for annotation in payload.get("primary_snapshot_data", {}).get("allele_annotations", []):
        for item in annotation.get("clinical", []):
            description = item.get("clinical_significances", [])
            clinvar_significance.extend(description)

    chrom = None
    position = None
    for trait in placement.get("placement_annot", {}).get("seq_id_traits_by_assembly", []):
        if trait.get("is_top_level"):
            chrom = trait.get("assembly_name", "")
            break

    if placement.get("alleles"):
        spdi = placement["alleles"][0].get("allele", {}).get("spdi", {})
        position = spdi.get("position", 0) + 1 if spdi.get("position") is not None else None

    return {
        "rsid": canonical,
        "refsnp_id": payload.get("refsnp_id"),
        "last_update_date": payload.get("last_update_date"),
        "reference_allele": reference,
        "observed_alleles": observed_alleles,
        "is_palindromic": frozenset(observed_alleles) in PALINDROMIC_ALLELE_SETS,
        "citations": payload.get("citations", [])[:100],
        "citations_count": len(payload.get("citations", [])),
        "seq_id": placement.get("seq_id"),
        "position": position,
        "placement_is_aln_opposite_orientation": placement.get("placement_annot", {}).get("is_aln_opposite_orientation", False),
        "clinvar_significance": sorted(set(clinvar_significance))[:10],
        "assembly_name": chrom,
    }


def ensure_variant_metadata(rsids: Iterable[str]) -> Dict[str, Dict]:
    cache = load_variant_metadata_cache()
    missing = [canonical_rsid(rsid) for rsid in rsids if canonical_rsid(rsid) not in cache]
    if not missing:
        return cache

    session = requests.Session()
    updated = False
    for rsid in missing:
        try:
            metadata = fetch_variant_metadata(rsid, session=session)
        except Exception:
            metadata = None
        if metadata:
            cache[rsid] = metadata
            updated = True

    if updated:
        save_variant_metadata_cache(cache)
    return cache


def validate_variant_call(
    genotype_record: Dict,
    panel_meta: Dict,
    authoritative_meta: Optional[Dict],
    strict: bool = True,
    source_hint: str = "unknown",
) -> Dict:
    """Validate a genotype call against authoritative dbSNP metadata.

    Key improvements over the original implementation:
    1. Uses **primary allele pair** (ref + main alt) instead of the full
       multi-allelic observed set to prevent false-positive validation.
    2. Resolves palindromic (A/T, C/G) SNPs as forward-strand when the
       source is known to report on the plus strand (e.g. AncestryDNA).
    3. Cross-checks the curated risk allele against the primary pair
       before falling back to the full observed set.
    """
    result = {
        "claim_ready": False,
        "validation_status": "unvalidated",
        "validation_notes": [],
        "risk_allele_plus_strand": None,
        "risk_allele_in_input_orientation": None,
        "risk_allele_count": 0,
        "input_orientation": "unknown",
        "is_palindromic": False,
    }

    if not authoritative_meta:
        result["validation_status"] = "missing_dbsnp_metadata"
        result["validation_notes"].append("No authoritative dbSNP metadata was available for this rsID.")
        return result

    observed_authoritative = set(authoritative_meta.get("observed_alleles", []))
    primary_alleles = get_primary_allele_pair(authoritative_meta)
    is_palindromic = bool(authoritative_meta.get("is_palindromic"))
    result["is_palindromic"] = is_palindromic

    # For palindromic check, use the primary allele pair, not the full set
    if len(primary_alleles) >= 2:
        primary_is_palindromic = frozenset(primary_alleles) in PALINDROMIC_ALLELE_SETS
    else:
        primary_is_palindromic = is_palindromic
    result["is_palindromic"] = primary_is_palindromic

    if len(observed_authoritative) < 2:
        result["validation_status"] = "insufficient_authoritative_alleles"
        result["validation_notes"].append("dbSNP did not provide a usable single-base allele set.")
        return result

    # Determine whether this source can resolve palindromic ambiguity
    source_is_plus_strand = source_hint.lower() in PLUS_STRAND_SOURCES
    can_resolve_palindromic = source_is_plus_strand and primary_is_palindromic

    # --- Risk allele reconciliation ---
    # First try against the primary allele pair (stricter), then fall back
    # to the full observed set for multi-allelic sites.
    curated_risk = panel_meta.get("risk_allele", "").upper()

    if curated_risk in primary_alleles:
        risk_plus = curated_risk
    elif curated_risk in observed_authoritative:
        # Risk allele is in the broader observed set but not the primary pair.
        # This may indicate a rare multi-allelic allele or a strand issue.
        risk_plus = curated_risk
        result["validation_notes"].append(
            f"Risk allele {curated_risk} matched a rare/multi-allelic dbSNP observed allele "
            f"but is not in the primary ref/alt pair {sorted(primary_alleles)}."
        )
    else:
        complemented_risk = complement_base(curated_risk)
        if complemented_risk in primary_alleles and not primary_is_palindromic:
            risk_plus = complemented_risk
            result["validation_notes"].append(
                f"Curated risk allele {curated_risk} was complemented to {complemented_risk} "
                f"against dbSNP forward-strand primary alleles."
            )
        elif complemented_risk in observed_authoritative and not primary_is_palindromic:
            risk_plus = complemented_risk
            result["validation_notes"].append(
                f"Curated risk allele {curated_risk} was complemented to {complemented_risk} "
                f"against dbSNP forward alleles (multi-allelic match)."
            )
        else:
            result["validation_status"] = "curated_risk_allele_not_supported"
            result["validation_notes"].append(
                f"The curated risk allele {curated_risk} could not be reconciled with "
                f"dbSNP forward alleles (primary pair: {sorted(primary_alleles)})."
            )
            return result

    # --- Input orientation detection ---
    input_alleles = tuple(genotype_record.get("alleles", ()))
    observed_input = set(input_alleles)

    # Use primary alleles for orientation check, with fallback to full set
    primary_complement = complement_set(primary_alleles)
    auth_complement = complement_set(observed_authoritative)

    if observed_input <= primary_alleles:
        input_orientation = "forward"
        risk_in_input = risk_plus
    elif observed_input <= observed_authoritative:
        # Input alleles match the broader observed set
        input_orientation = "forward"
        risk_in_input = risk_plus
    elif observed_input <= primary_complement and not primary_is_palindromic:
        input_orientation = "reverse_complemented_input"
        risk_in_input = complement_base(risk_plus)
        result["validation_notes"].append(
            "Input genotype was reverse complemented to reconcile with dbSNP forward alleles."
        )
    elif observed_input <= auth_complement and not primary_is_palindromic:
        input_orientation = "reverse_complemented_input"
        risk_in_input = complement_base(risk_plus)
        result["validation_notes"].append(
            "Input genotype was reverse complemented against the broader dbSNP allele set."
        )
    else:
        # Palindromic or genuinely unresolvable
        if primary_is_palindromic:
            if can_resolve_palindromic:
                # Source is known plus-strand — treat input as forward
                input_orientation = "forward"
                risk_in_input = risk_plus
                result["validation_notes"].append(
                    f"Palindromic A/T or C/G SNP resolved as forward-strand because "
                    f"{source_hint} documents plus-strand allele reporting."
                )
            elif strict:
                result["validation_status"] = "palindromic_strand_ambiguous"
                result["validation_notes"].append(
                    "This rsID is an A/T or C/G SNP. Without probe-manifest context "
                    "or a known plus-strand source, strand resolution is ambiguous."
                )
                return result
            else:
                # Non-strict palindromic — allow with caveat
                input_orientation = "forward"
                risk_in_input = risk_plus
                result["validation_notes"].append(
                    "Palindromic SNP allowed under non-strict mode (strand assumed forward)."
                )
        else:
            result["validation_status"] = "input_alleles_not_supported"
            result["validation_notes"].append(
                f"Observed input alleles {sorted(observed_input)} did not reconcile "
                f"with authoritative dbSNP alleles {sorted(primary_alleles)}."
            )
            return result

    risk_count = sum(1 for allele in input_alleles if allele == risk_in_input)
    result.update({
        "claim_ready": True,
        "validation_status": "validated",
        "risk_allele_plus_strand": risk_plus,
        "risk_allele_in_input_orientation": risk_in_input,
        "risk_allele_count": risk_count,
        "input_orientation": input_orientation,
    })

    if primary_is_palindromic and can_resolve_palindromic:
        result["validation_notes"].append(
            f"Palindromic variant validated via {source_hint} plus-strand convention."
        )
    elif input_orientation == "forward":
        result["validation_notes"].append(
            "Observed genotype matched authoritative dbSNP forward-strand alleles directly."
        )

    return result


def panel_keep_set(panel_keys: Iterable[str]) -> set[str]:
    return {canonical_panel_rsid(key) for key in panel_keys}
