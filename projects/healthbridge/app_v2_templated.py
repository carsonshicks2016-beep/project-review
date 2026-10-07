"""
HealthBridge AI - Enhanced Web Application v2 with Jinja2 Templates
Integrates clinical evidence, action scoring, pre-test conditions, and visualizations
"""

import json
import os
import tempfile
import traceback
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our new modules
from snp_processor import analyze_dna_file, parse_ancestry_file, process_snps, SNP_PANEL
from blood_parser import process_blood_work, BIOMARKER_DB
from pattern_engine import run_pattern_engine
from wearable_parser import parse_wearable, build_wearable_summary

# Import new clinical modules
from clinical_evidence import (
    clinical_db, get_clinical_context, get_biomarker_reliability_report,
    ClinicalValidator
)
from action_scoring import ActionScoringEngine, ScoredFinding, ClinicalAction
from pre_test_conditions import (
    PreTestConditions, BiomarkerAdjustments, ExerciseTiming,
    ConditionsForm, MedicationEntry
)
from visualizations import (
    ChartGenerator, HTMLChartRenderer,
    BiomarkerPoint
)
from analysis_foundation import build_report_context, build_methodology_context, build_wearable_context

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "healthbridge_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize scoring engine
scoring_engine = ActionScoringEngine()


# =============================================================================
# TEMPLATE FILTERS
# =============================================================================

@app.template_filter('confidence_class')
def confidence_class_filter(score):
    """Convert confidence score to CSS class"""
    if score >= 70:
        return 'high'
    elif score >= 50:
        return 'medium'
    return 'low'


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_with_clinical_context(dna_results: dict, blood_results: dict,
                                   conditions: PreTestConditions,
                                   wearable_metrics: dict | None = None) -> dict:
    """
    Perform comprehensive analysis with clinical context
    """
    output = {
        "mode": "combined",
        "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "clinical_confidence": {},
        "adjusted_biomarkers": {},
        "scored_findings": [],
        "action_plan": {},
        "visualizations": {},
        "pattern_results": {},
        "wearable_metrics": wearable_metrics,
    }
    evaluated_biomarkers = blood_results.get("evaluated", {}) if isinstance(blood_results, dict) else {}
    bio_lookup = {}

    # Step 1: Calculate pattern confidence
    all_snps = []
    for category, snps in dna_results.items():
        for snp in snps:
            if not snp.get("claim_ready", True):
                continue
            all_snps.append(snp.get("rsid", "").lstrip("rs"))

    for marker_name, biomarker in evaluated_biomarkers.items():
        if not isinstance(biomarker, dict):
            continue
        db_key = biomarker.get("db_key") or marker_name
        value = biomarker.get("value")
        if value is None:
            continue
        bio_lookup[db_key] = value

    all_biomarkers = list(bio_lookup.keys())

    confidence = clinical_db.calculate_pattern_confidence(all_snps, all_biomarkers)
    output["clinical_confidence"] = confidence

    # Step 2: Adjust biomarkers for pre-test conditions
    validator = ClinicalValidator(clinical_db)
    for marker_name, biomarker in evaluated_biomarkers.items():
        if not isinstance(biomarker, dict):
            continue
        marker = biomarker.get("db_key") or marker_name
        val = biomarker.get("value")
        if val is None:
            continue

        adjusted, factors = BiomarkerAdjustments.calculate_adjusted_value(
            marker, val, conditions
        )

        validation = validator.validate_biomarker(marker, val, conditions.to_dict())

        output["adjusted_biomarkers"][marker] = {
            "measured": val,
            "adjusted": adjusted,
            "adjustment_factors": factors,
            "confidence_score": validation.get("confidence_score", 50),
            "caveats": validation.get("caveats", [])
        }

    # Step 3: Run pattern engine and score findings
    pattern_results = run_pattern_engine(dna_results, blood_results)
    patterns = pattern_results.get("hardcoded_patterns", [])
    output["pattern_results"] = pattern_results

    snp_lookup = {}
    for cat, snps in dna_results.items():
        for snp in snps:
            if not snp.get("claim_ready", True):
                continue
            snp_lookup[snp.get("rsid", "").lstrip("rs")] = snp

    scored_findings = []
    for pattern_match in patterns:
        pattern = pattern_match.get("pattern", {})
        scored = scoring_engine.score_pattern(
            pattern, snp_lookup, bio_lookup, confidence["combined_confidence"]
        )
        scored_findings.append(scored)

    output["scored_findings"] = [f.to_dict() for f in scored_findings]

    # Step 4: Generate action plan
    action_plan = scoring_engine.generate_action_plan(scored_findings)
    output["action_plan"] = action_plan

    # Step 5: Generate visualizations
    dashboard = ChartGenerator.generate_composite_dashboard(
        bio_lookup,
        {cat: len(snps) for cat, snps in dna_results.items()}
    )
    output["visualizations"]["dashboard"] = dashboard

    return output


def _mode_name(mode: str) -> str:
    return {
        "dna": "DNA Analysis",
        "blood": "Blood Work Analysis",
        "combined": "Combined Analysis",
    }.get(mode, "Health Analysis")


def _default_conditions() -> PreTestConditions:
    now = datetime.now()
    return PreTestConditions(
        collection_date=now,
        collection_time=now,
        fasting_hours=12,
        sleep_hours_prior=7,
        exercise_timing=ExerciseTiming.NONE,
        acute_illness=False,
        injury_recent=False,
        inflammation_present=False,
    )


def _source_error_message(source: str, exc: Exception) -> str:
    message = str(exc).strip()
    source = source.lower()

    if source == "blood":
        if "could not extract biomarkers" in message.lower():
            return message
        return (
            "The lab report could not be normalized from this PDF. "
            "Try a text-based export from the lab portal or a cleaner PDF with selectable text."
        )

    if source == "dna":
        if "required" in message.lower():
            return message
        return (
            "The DNA upload could not be parsed. "
            "Use the raw AncestryDNA text export rather than a screenshot, PDF, or spreadsheet."
        )

    return message or "The upload could not be processed for this run."


def _build_analysis_error_context(requested_mode: str, error_summary: str, source_errors: list[dict]) -> dict:
    recovery_steps = [
        "Return home and retry the upload with the original source files rather than edited exports or screenshots.",
        "If you are uploading a lab report, prefer a portal PDF with selectable text instead of a scan or phone photo.",
        "If you are uploading DNA, use the raw `AncestryDNA.txt` export without renaming it to another format.",
        "If the same files keep failing, retry each source separately to isolate whether the issue is DNA, blood work, or both.",
    ]

    return {
        "requested_mode_label": _mode_name(requested_mode),
        "error_title": f"{_mode_name(requested_mode)} could not be completed",
        "error_body": (
            "HealthBridge stopped before the report was rendered. "
            "This usually means one of the uploaded files could not be normalized cleanly enough for a trustworthy report."
        ),
        "error_summary": error_summary,
        "error_details": source_errors or [{"source": "Analysis engine", "message": error_summary}],
        "recovery_steps": recovery_steps,
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Home page with mode selection"""
    return render_template("index.html", **build_methodology_context())


@app.route("/methodology")
def methodology():
    """Explain how HealthBridge builds its report."""
    return render_template("methodology.html", **build_methodology_context())


@app.route("/analyze", methods=["POST"])
def analyze():
    """Process uploaded files and generate report"""
    requested_mode = request.form.get("mode", "combined")
    mode = requested_mode
    dna_path = None
    blood_path = None
    analysis_notice = None
    analysis_errors = []
    conditions = None
    dna_results = {}
    dna_qc = {}
    blood_results = {}
    wearable_metrics = None

    try:
        # Parse pre-test conditions if blood work is involved
        if requested_mode in ["blood", "combined"]:
            try:
                conditions = ConditionsForm.parse_form_data(request.form)
            except Exception as e:
                print(f"Error parsing conditions: {e}")
                traceback.print_exc()
                conditions = _default_conditions()
                analysis_notice = (
                    "Some collection-context fields could not be read, so HealthBridge used "
                    "standard default assumptions for this run."
                )

        # Process files
        if requested_mode in ["dna", "combined"]:
            dna_file = request.files.get("dna_file")
            if not dna_file or not dna_file.filename:
                if requested_mode == "dna":
                    raise RuntimeError("A DNA file is required for this analysis.")
            else:
                dna_path = os.path.join(UPLOAD_FOLDER, secure_filename(dna_file.filename))
                dna_file.save(dna_path)
                try:
                    dna_results, dna_qc = analyze_dna_file(dna_path, strict=True)
                except Exception as exc:
                    print("DNA processing failed:")
                    traceback.print_exc()
                    message = _source_error_message("dna", exc)
                    analysis_errors.append({"source": "DNA upload", "message": message})
                    dna_results = {}
                    dna_qc = {}
                    if requested_mode == "dna":
                        raise RuntimeError(message) from exc

        if requested_mode in ["blood", "combined"]:
            blood_file = request.files.get("blood_file")
            if not blood_file or not blood_file.filename:
                if requested_mode == "blood":
                    raise RuntimeError("A lab report PDF is required for this analysis.")
            else:
                blood_path = os.path.join(UPLOAD_FOLDER, secure_filename(blood_file.filename))
                blood_file.save(blood_path)
                try:
                    blood_results = process_blood_work(blood_path, dna_results or None)
                except Exception as exc:
                    print("Blood work processing failed:")
                    traceback.print_exc()
                    message = _source_error_message("blood", exc)
                    analysis_errors.append({"source": "Blood work upload", "message": message})
                    blood_results = {}
                    if requested_mode == "blood":
                        raise RuntimeError(message) from exc

        if requested_mode == "combined":
            if dna_results and not blood_results:
                mode = "dna"
                analysis_notice = (
                    "Combined analysis was partially completed. HealthBridge preserved the "
                    "usable DNA analysis and flagged the blood-work issue below for retry."
                )
            elif blood_results and not dna_results:
                mode = "blood"
                analysis_notice = (
                    "Combined analysis was partially completed. HealthBridge preserved the "
                    "usable blood-work analysis and flagged the DNA issue below for retry."
                )
            elif not dna_results and not blood_results:
                raise RuntimeError(
                    "HealthBridge could not extract usable data from either upload. "
                    "Try the raw AncestryDNA text export and a text-based lab PDF with selectable text."
                )

        # Parse wearable files if provided
        wearable_files_list = request.files.getlist("wearable_files")
        if wearable_files_list:
            wearable_file_map = {}
            for wf in wearable_files_list:
                if wf and wf.filename:
                    try:
                        content = wf.read().decode("utf-8", errors="ignore")
                        wearable_file_map[wf.filename] = content
                    except Exception:
                        pass
            if wearable_file_map:
                try:
                    wearable_metrics = parse_wearable(wearable_file_map)
                except Exception as exc:
                    print(f"Wearable parsing failed: {exc}")
                    traceback.print_exc()

        # Run enhanced analysis
        if mode in ["blood", "combined"] and conditions and blood_results:
            analysis = analyze_with_clinical_context(dna_results, blood_results, conditions, wearable_metrics)
            analysis["mode"] = mode
        else:
            # Simplified analysis for single-mode
            analysis = {
                "mode": mode,
                "dna_results": dna_results,
                "blood_results": blood_results,
                "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                "clinical_confidence": None,
                "scored_findings": [],
                "action_plan": {},
                "visualizations": {},
                "adjusted_biomarkers": {},
                "pattern_results": {},
                "wearable_metrics": wearable_metrics,
            }

        analysis.update(
            build_report_context(
                mode=mode,
                dna_results=dna_results,
                blood_results=blood_results,
                conditions=conditions,
                clinical_confidence=analysis.get("clinical_confidence"),
                adjusted_biomarkers=analysis.get("adjusted_biomarkers"),
                scored_findings=analysis.get("scored_findings"),
                pattern_results=analysis.get("pattern_results"),
                dna_qc=dna_qc,
                wearable_metrics=analysis.get("wearable_metrics"),
            )
        )
        analysis["analysis_notice"] = analysis_notice
        analysis["analysis_errors"] = analysis_errors
        analysis["requested_mode"] = requested_mode
        analysis["requested_mode_label"] = _mode_name(requested_mode)
        analysis["dna_qc"] = dna_qc
        analysis["is_demo"] = False

        return render_template("results.html", **analysis)
    except Exception as exc:
        print("Analysis request failed:")
        traceback.print_exc()
        context = _build_analysis_error_context(requested_mode, str(exc), analysis_errors)
        status_code = 422 if isinstance(exc, RuntimeError) else 500
        return render_template("analysis_error.html", **context, **build_methodology_context()), status_code
    finally:
        for path in [dna_path, blood_path]:
            if path and os.path.exists(path):
                os.remove(path)


@app.route("/demo")
def demo():
    """Pre-loaded demo report using synthetic DNA, blood, and wearable data."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── Load DNA ────────────────────────────────────────────────────────────────
    dna_path = os.path.join(BASE_DIR, "test_ancestry.txt")
    dna_results = {}
    dna_qc = {}
    try:
        dna_results, dna_qc = analyze_dna_file(dna_path, strict=True)
    except Exception as exc:
        print(f"Demo DNA load failed: {exc}")
        traceback.print_exc()

    # ── Load blood work ─────────────────────────────────────────────────────────
    blood_results = {}
    blood_path = os.path.join(BASE_DIR, "synthetic_blood_test.pdf")
    try:
        blood_results = process_blood_work(blood_path, dna_results or None)
    except Exception as exc:
        print(f"Demo blood load via PDF failed ({exc}), falling back to JSON sidecar.")
        json_path = os.path.join(BASE_DIR, "synthetic_blood_test.json")
        if os.path.exists(json_path):
            import json as _json
            try:
                with open(json_path) as fh:
                    raw = _json.load(fh)
                # Build a minimal blood_results structure from the JSON sidecar
                evaluated = {}
                for key, entry in raw.items():
                    evaluated[key] = {
                        "name": key.replace("_", " ").title(),
                        "db_key": key,
                        "value": entry["value"],
                        "unit": entry.get("unit", ""),
                        "reference_range_lab": entry.get("reference_range", ""),
                        "standard_range": entry.get("reference_range", ""),
                        "optimal_range": "",
                        "flag": "optimal",
                        "category": "General",
                        "interpretation": "",
                        "genomic_context": None,
                        "genomic_actions": [],
                    }
                blood_results = {
                    "evaluated": evaluated,
                    "total_biomarkers": len(evaluated),
                    "flagged": {"abnormal": [], "suboptimal": [], "genomic_recontextualized": [], "optimal": list(evaluated.keys())},
                }
            except Exception as je:
                print(f"Demo JSON sidecar also failed: {je}")

    # ── Load wearable data ──────────────────────────────────────────────────────
    wearable_metrics = None
    oura_files = {
        "oura_sleep.csv": None,
        "oura_readiness.csv": None,
        "oura_activity.csv": None,
    }
    for fname in list(oura_files.keys()):
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as fh:
                oura_files[fname] = fh.read()
        else:
            del oura_files[fname]

    if oura_files:
        try:
            wearable_metrics = parse_wearable(oura_files)
        except Exception as exc:
            print(f"Demo wearable load failed: {exc}")
            traceback.print_exc()

    # ── Build analysis ──────────────────────────────────────────────────────────
    conditions = _default_conditions()
    try:
        if blood_results:
            analysis = analyze_with_clinical_context(dna_results, blood_results, conditions, wearable_metrics)
        else:
            analysis = {
                "mode": "dna",
                "dna_results": dna_results,
                "blood_results": {},
                "generated_at": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                "clinical_confidence": None,
                "scored_findings": [],
                "action_plan": {},
                "visualizations": {},
                "adjusted_biomarkers": {},
                "pattern_results": {},
                "wearable_metrics": wearable_metrics,
            }

        mode = "combined" if (dna_results and blood_results) else ("dna" if dna_results else "blood")
        analysis["mode"] = mode

        analysis.update(
            build_report_context(
                mode=mode,
                dna_results=dna_results,
                blood_results=blood_results,
                conditions=conditions,
                clinical_confidence=analysis.get("clinical_confidence"),
                adjusted_biomarkers=analysis.get("adjusted_biomarkers"),
                scored_findings=analysis.get("scored_findings"),
                pattern_results=analysis.get("pattern_results"),
                dna_qc=dna_qc,
                wearable_metrics=analysis.get("wearable_metrics"),
            )
        )
        analysis["analysis_notice"] = None
        analysis["analysis_errors"] = []
        analysis["requested_mode"] = mode
        analysis["requested_mode_label"] = _mode_name(mode)
        analysis["dna_qc"] = dna_qc
        analysis["is_demo"] = True

        return render_template("results.html", **analysis)
    except Exception as exc:
        print("Demo route failed:")
        traceback.print_exc()
        context = _build_analysis_error_context("combined", str(exc), [])
        return render_template("analysis_error.html", **context, **build_methodology_context()), 500


@app.route("/api/biomarker/<name>/reliability")
def biomarker_reliability(name):
    """API endpoint for biomarker reliability data"""
    report = get_biomarker_reliability_report(name)
    if report:
        return jsonify(report)
    return jsonify({"error": "Biomarker not found"}), 404


@app.route("/api/snp/<rsid>/context")
def snp_context(rsid):
    """API endpoint for SNP clinical context"""
    context = get_clinical_context(rsid)
    if context:
        return jsonify(context)
    return jsonify({"error": "SNP not found"}), 404


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template("base.html", content="""
    <div class="hero">
        <h1>Page Not Found</h1>
        <p>The page you're looking for doesn't exist.</p>
        <a href="/" class="btn btn-primary mt-4">Return Home</a>
    </div>
    """), 404


@app.errorhandler(500)
def server_error(error):
    context = {
        "requested_mode_label": "HealthBridge",
        "error_title": "Server Error",
        "error_body": "Something went wrong while rendering this page. Please retry the analysis or return home and start a fresh run.",
        "error_summary": "Unexpected server error",
        "error_details": [{"source": "Application", "message": "The request could not be completed cleanly."}],
        "recovery_steps": [
            "Return home and rerun the analysis once.",
            "If the issue only happens with one file, retry the sources separately to isolate the problem.",
            "Prefer raw source exports over screenshots, scans, or reformatted documents.",
        ],
    }
    return render_template("analysis_error.html", **context, **build_methodology_context()), 500


if __name__ == "__main__":
    print("=" * 60)
    print("HealthBridge AI Enhanced (v2) - Jinja2 Template Edition")
    print("=" * 60)
    print("\nFeatures loaded:")
    print("  ✓ Clinical Evidence Module")
    print("  ✓ Action Scoring Engine")
    print("  ✓ Pre-Test Conditions Module")
    print("  ✓ Visualization Module")
    print("  ✓ Jinja2 Templates with external CSS/JS")
    print("\nTemplate folders:")
    print(f"  - Templates: {app.template_folder}")
    print(f"  - Static: {app.static_folder}")
    print("\nStarting server at http://localhost:5001")
    print("=" * 60)
    app.run(debug=True, port=5001)
