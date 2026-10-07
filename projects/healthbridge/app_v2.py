"""
HealthBridge AI - Enhanced Web Application v2
Integrates clinical evidence, action scoring, pre-test conditions, and visualizations
"""

import json
import os
import tempfile
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our new modules
from snp_processor import parse_ancestry_file, process_snps, SNP_PANEL
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "healthbridge_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize scoring engine
scoring_engine = ActionScoringEngine()

# =============================================================================
# ENHANCED HTML TEMPLATES
# =============================================================================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HealthBridge AI — Precision Health Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --color-primary: #0d3b5e;
            --color-success: #059669;
            --color-warning: #f59e0b;
            --color-danger: #dc2626;
            --color-info: #3b82f6;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f7f8fa;
            color: #111827;
            min-height: 100vh;
            line-height: 1.6;
        }

        /* Navigation */
        nav {
            background: #fff;
            border-bottom: 1px solid #e5e7eb;
            height: 62px;
            display: flex;
            align-items: center;
            padding: 0 48px;
            position: sticky;
            top: 0;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .brand { font-size: 1.08em; font-weight: 700; color: var(--color-primary); letter-spacing: -0.4px; }
        .brand em { color: var(--color-success); font-style: normal; }
        .brand-tag {
            margin-left: 14px;
            font-size: 0.67em;
            color: #9ca3af;
            font-weight: 500;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            border-left: 1px solid #e5e7eb;
            padding-left: 14px;
        }

        /* Hero */
        .hero {
            max-width: 1160px;
            margin: 0 auto;
            padding: 68px 48px 52px;
        }
        .hero-eyebrow {
            font-size: 0.71em;
            font-weight: 600;
            letter-spacing: 2.2px;
            text-transform: uppercase;
            color: var(--color-success);
            margin-bottom: 18px;
        }
        .hero h1 {
            font-size: 2.9em;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.13;
            letter-spacing: -1.3px;
            margin-bottom: 20px;
            max-width: 640px;
        }
        .hero p {
            font-size: 1.02em;
            color: #6b7280;
            line-height: 1.78;
            max-width: 540px;
        }

        /* Mode Cards */
        .modes {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            max-width: 1160px;
            margin: 48px auto 0;
            padding: 0 48px;
        }
        .mode-card {
            background: #fff;
            border: 2px solid #e5e7eb;
            border-radius: 16px;
            padding: 30px 26px;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }
        .mode-card:hover { border-color: var(--color-primary); box-shadow: 0 8px 32px rgba(13,59,94,0.1); transform: translateY(-2px); }
        .mode-card.selected {
            border-color: var(--color-primary);
            box-shadow: 0 0 0 4px rgba(13,59,94,0.07), 0 8px 32px rgba(13,59,94,0.1);
        }
        .mode-badge {
            display: inline-flex;
            align-items: center;
            font-size: 0.64em;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            padding: 4px 11px;
            border-radius: 20px;
            margin-bottom: 16px;
        }
        .badge-dna { background: #eff6ff; color: #1d4ed8; }
        .badge-blood { background: #fff1f2; color: #be123c; }
        .badge-combined { background: #f0fdf4; color: var(--color-success); }
        .mode-title { font-size: 1.05em; font-weight: 700; color: #0f172a; margin-bottom: 10px; }
        .mode-desc { font-size: 0.8em; color: #6b7280; line-height: 1.68; }

        /* Upload Section */
        .upload-section {
            display: none;
            max-width: 1160px;
            margin: 24px auto 0;
            padding: 0 48px;
        }
        .upload-section.visible { display: block; }
        .upload-card {
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 42px 44px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .upload-grid { display: grid; gap: 24px; margin-bottom: 28px; }
        .upload-field label {
            display: block;
            font-size: 0.82em;
            font-weight: 600;
            color: #374151;
            margin-bottom: 9px;
        }
        .upload-field input[type="file"] {
            display: block;
            width: 100%;
            padding: 18px 20px;
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            font-size: 0.84em;
            color: #6b7280;
            background: #fafafa;
            cursor: pointer;
            transition: all 0.2s;
        }
        .upload-field input[type="file"]:hover { border-color: var(--color-primary); background: #f0f6ff; }
        .submit-btn {
            display: block;
            width: 100%;
            padding: 17px;
            background: var(--color-primary);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: 0.2px;
            transition: all 0.2s;
        }
        .submit-btn:hover { background: #0a2e4a; box-shadow: 0 6px 20px rgba(13,59,94,0.28); }

        /* Results Styling */
        .results-container { max-width: 1160px; margin: 40px auto; padding: 0 48px; }

        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
        }
        .confidence-high { background: #dcfce7; color: #166534; }
        .confidence-medium { background: #fef3c7; color: #92400e; }
        .confidence-low { background: #fee2e2; color: #991b1b; }

        .priority-banner {
            padding: 16px 20px;
            border-radius: 12px;
            margin: 20px 0;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }
        .priority-critical { background: #fee2e2; border: 1px solid #fecaca; }
        .priority-high { background: #ffedd5; border: 1px solid #fed7aa; }
        .priority-moderate { background: #fef3c7; border: 1px solid #fde68a; }

        .evidence-panel {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 16px 0;
        }
        .evidence-panel h4 {
            font-size: 0.85em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .evidence-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9em;
        }
        .evidence-item:last-child { border-bottom: none; }

        /* Disclaimer */
        .disclaimer {
            max-width: 1160px;
            margin: 52px auto;
            padding: 0 48px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        .disclaimer-icon { font-size: 0.88em; flex-shrink: 0; color: #9ca3af; margin-top: 2px; }
        .disclaimer-text { font-size: 0.74em; color: #9ca3af; line-height: 1.68; }

        /* Loading */
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.95);
            z-index: 100;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .loading-overlay.visible { display: flex; }
        .spinner {
            width: 50px; height: 50px;
            border: 4px solid #e5e7eb;
            border-top-color: var(--color-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Chart containers */
        .chart-container {
            background: #fff;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    {{content|safe}}

    <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
        <p style="margin-top: 20px; color: #6b7280;">Analyzing your health data...</p>
    </div>

    <script>
        function selectMode(mode) {
            document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
            document.getElementById('card-' + mode).classList.add('selected');
            document.querySelectorAll('.upload-section').forEach(s => s.classList.remove('visible'));
            document.getElementById('section-' + mode).classList.add('visible');
        }

        function showLoading() {
            document.getElementById('loading').classList.add('visible');
        }
    </script>
</body>
</html>
"""

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_with_clinical_context(dna_results: dict, blood_results: dict,
                                   conditions: PreTestConditions) -> dict:
    """
    Perform comprehensive analysis with clinical context
    """
    output = {
        "generated_at": datetime.now().isoformat(),
        "clinical_confidence": {},
        "adjusted_biomarkers": {},
        "scored_findings": [],
        "action_plan": {},
        "visualizations": {}
    }
    evaluated_biomarkers = blood_results.get("evaluated", {}) if isinstance(blood_results, dict) else {}
    bio_lookup = {}

    # Step 1: Calculate pattern confidence
    all_snps = []
    for category, snps in dna_results.items():
        for snp in snps:
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

    snp_lookup = {}
    for cat, snps in dna_results.items():
        for snp in snps:
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

    # Generate gauge for each biomarker
    gauges = {}
    for marker, value in bio_lookup.items():
        optimal = ChartGenerator.OPTIMAL_RANGES.get(marker)
        standard = ChartGenerator.STANDARD_RANGES.get(marker)
        if optimal and standard:
            gauges[marker] = ChartGenerator.generate_biomarker_gauge(
                marker, value, (optimal[0], optimal[1]), (standard[0], standard[1])
            )
    output["visualizations"]["gauges"] = gauges

    return output


def render_enhanced_report(analysis: dict) -> str:
    """Render the enhanced analysis report"""

    # Confidence badge
    confidence = analysis.get("clinical_confidence", {})
    conf_score = confidence.get("combined_confidence", 50)
    if conf_score >= 70:
        conf_class = "confidence-high"
        conf_label = f"High Confidence ({conf_score}/100)"
    elif conf_score >= 50:
        conf_class = "confidence-medium"
        conf_label = f"Moderate Confidence ({conf_score}/100)"
    else:
        conf_class = "confidence-low"
        conf_label = f"Limited Confidence ({conf_score}/100)"

    # Dashboard
    dashboard = analysis.get("visualizations", {}).get("dashboard", {})
    dashboard_html = HTMLChartRenderer.render_dashboard_html(dashboard)

    # Top priorities
    priorities = analysis.get("action_plan", {}).get("top_3_priorities", [])
    priorities_html = ""
    for p in priorities:
        priority_class = f"priority-{p.get('action_priority', {}).get('level', 'moderate').lower()}"
        priorities_html += f"""
        <div class="priority-banner {priority_class}">
            <div style="font-size: 1.5em;">⚠️</div>
            <div>
                <div style="font-weight: 700; color: #7c2d12; margin-bottom: 4px;">
                    {p.get('name', 'Unknown')}
                </div>
                <div style="font-size: 0.9em; color: #92400e;">
                    {p.get('description', '')[:200]}...
                </div>
                <div style="margin-top: 8px; font-size: 0.8em; color: #b45309;">
                    Risk Score: {p.get('composite_risk_score', 0)}/100 |
                    Action: {p.get('time_to_intervention', 'Address as needed')}
                </div>
            </div>
        </div>
        """

    # Evidence panel
    evidence_html = f"""
    <div class="evidence-panel">
        <h4>Clinical Evidence Summary</h4>
        <div class="evidence-item">
            <span>Genetic Evidence Confidence</span>
            <span class="confidence-badge {conf_class}">{confidence.get('snp_confidence', 0)}/100</span>
        </div>
        <div class="evidence-item">
            <span>Biomarker Reliability</span>
            <span class="confidence-badge {conf_class}">{confidence.get('biomarker_confidence', 0)}/100</span>
        </div>
        <div class="evidence-item">
            <span>Overall Reliability Grade</span>
            <span style="font-weight: 600;">{confidence.get('reliability_grade', 'Unknown')}</span>
        </div>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 0.85em; color: #64748b;">
            {confidence.get('recommendation', '')}
        </div>
    </div>
    """

    content = f"""
    <nav>
        <div class="brand">Health<em>Bridge</em> AI</div>
        <div class="brand-tag">Enhanced Clinical Report</div>
    </nav>

    <div class="results-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <h1>Your Health Analysis Report</h1>
            <span class="confidence-badge {conf_class}">{conf_label}</span>
        </div>

        {dashboard_html}

        <h2 style="margin: 40px 0 20px; color: #1f2937;">Priority Findings</h2>
        {priorities_html}

        {evidence_html}

        <h2 style="margin: 40px 0 20px; color: #1f2937;">Action Timeline</h2>

        <div style="background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #dc2626; margin-bottom: 16px;">⚠️ Immediate (Within 1 Week)</h3>
            <div id="immediate-actions">
                <!-- Populated by JavaScript -->
            </div>
        </div>

        <div style="background: #fff; border-radius: 12px; padding: 24px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: #f59e0b; margin-bottom: 16px;">📋 Short-term (Within 1 Month)</h3>
            <div id="short-term-actions">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const analysisData = {json.dumps(analysis, indent=2)};
        console.log('Analysis data loaded:', analysisData);
    </script>
    """

    return render_template_string(BASE_TEMPLATE, content=content)


# =============================================================================
# ROUTES
# =============================================================================

UPLOAD_PAGE = """
<nav>
    <div class="brand">Health<em>Bridge</em> AI</div>
    <div class="brand-tag">Health Intelligence Platform</div>
</nav>

<div class="hero">
    <div class="hero-eyebrow">Precision Health Analysis</div>
    <h1>Your biology,<br>fully decoded.</h1>
    <p>Upload your genetic data, lab work, and wearable exports. HealthBridge AI cross-references all three sources with clinical evidence to build a personalized health intelligence report.</p>
</div>

<div class="modes">
    <div class="mode-card" onclick="selectMode('dna')" id="card-dna">
        <div class="mode-badge badge-dna">Genomic</div>
        <div class="mode-title">DNA Analysis</div>
        <div class="mode-desc">Analyzes ~1,000 clinically relevant SNPs across 14 health categories from your AncestryDNA raw data file.</div>
    </div>
    <div class="mode-card" onclick="selectMode('blood')" id="card-blood">
        <div class="mode-badge badge-blood">Biomarker</div>
        <div class="mode-title">Blood Work Analysis</div>
        <div class="mode-desc">Evaluates your lab report with confidence scoring and pre-test condition adjustments.</div>
    </div>
    <div class="mode-card" onclick="selectMode('combined')" id="card-combined">
        <div class="mode-badge badge-combined">★ Full Analysis</div>
        <div class="mode-title">Cross-Source Analysis</div>
        <div class="mode-desc">Combines DNA and blood work with clinical evidence linking for the most comprehensive picture.</div>
    </div>
</div>

<div class="upload-section" id="section-dna">
    <div class="upload-card">
        <form action="/analyze" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <input type="hidden" name="mode" value="dna">
            <div class="upload-grid">
                <div class="upload-field">
                    <label>AncestryDNA Raw Data File <em>(.txt)</em></label>
                    <input type="file" name="dna_file" accept=".txt" required>
                </div>
            </div>
            <button class="submit-btn" type="submit">Generate Genetic Report →</button>
        </form>
    </div>
</div>

<div class="upload-section" id="section-blood">
    <div class="upload-card">
        <form action="/analyze" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <input type="hidden" name="mode" value="blood">
            <div class="upload-grid">
                <div class="upload-field">
                    <label>Lab Report PDF</label>
                    <input type="file" name="blood_file" accept=".pdf" required>
                </div>
            </div>
            {conditions_form}
            <button class="submit-btn" type="submit">Generate Blood Work Report →</button>
        </form>
    </div>
</div>

<div class="upload-section" id="section-combined">
    <div class="upload-card">
        <form action="/analyze" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <input type="hidden" name="mode" value="combined">
            <div class="upload-grid">
                <div class="upload-field">
                    <label>AncestryDNA Raw Data File <em>(.txt)</em></label>
                    <input type="file" name="dna_file" accept=".txt" required>
                </div>
                <div class="upload-field">
                    <label>Lab Report PDF</label>
                    <input type="file" name="blood_file" accept=".pdf" required>
                </div>
            </div>
            {conditions_form}
            <button class="submit-btn" type="submit">Generate Full Analysis →</button>
        </form>
    </div>
</div>

<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-text">
        HealthBridge AI is for informational and lifestyle optimization purposes only and does not constitute medical advice, diagnosis, or treatment. All uploaded files are deleted immediately after report generation. Consult a qualified healthcare provider before making changes to your health regimen.
    </div>
</div>
"""

@app.route("/")
def index():
    """Home page with mode selection"""
    conditions_form_html = ConditionsForm.get_html_form()
    content = UPLOAD_PAGE.format(conditions_form=conditions_form_html)
    return render_template_string(BASE_TEMPLATE, content=content)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Process uploaded files and generate report"""
    mode = request.form.get("mode", "combined")
    dna_path = None
    blood_path = None

    try:
        # Parse pre-test conditions if blood work is involved
        conditions = None
        if mode in ["blood", "combined"]:
            try:
                conditions = ConditionsForm.parse_form_data(request.form)
            except Exception as e:
                print(f"Error parsing conditions: {e}")
                conditions = PreTestConditions(
                    collection_date=datetime.now(),
                    collection_time=datetime.now(),
                    fasting_hours=12,
                    sleep_hours_prior=7,
                    exercise_timing=ExerciseTiming.NONE,
                    acute_illness=False
                )

        # Process files
        dna_results = {}
        blood_results = {}

        if mode in ["dna", "combined"]:
            if "dna_file" in request.files:
                dna_file = request.files["dna_file"]
                if dna_file.filename:
                    dna_path = os.path.join(UPLOAD_FOLDER, secure_filename(dna_file.filename))
                    dna_file.save(dna_path)
                    genotypes = parse_ancestry_file(dna_path)
                    dna_results = process_snps(genotypes)

        if mode in ["blood", "combined"]:
            if "blood_file" in request.files:
                blood_file = request.files["blood_file"]
                if blood_file.filename:
                    blood_path = os.path.join(UPLOAD_FOLDER, secure_filename(blood_file.filename))
                    blood_file.save(blood_path)
                    blood_results = process_blood_work(blood_path, dna_results or None)

        # Run enhanced analysis
        if mode == "combined" and conditions:
            analysis = analyze_with_clinical_context(dna_results, blood_results, conditions)
        else:
            # Simplified analysis for single-mode
            analysis = {
                "mode": mode,
                "dna_results": dna_results,
                "blood_results": blood_results,
                "generated_at": datetime.now().isoformat()
            }

        return render_enhanced_report(analysis)
    finally:
        for path in [dna_path, blood_path]:
            if path and os.path.exists(path):
                os.remove(path)


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


if __name__ == "__main__":
    print("Starting HealthBridge AI Enhanced (v2)...")
    print("Clinical Evidence Module: Loaded")
    print("Action Scoring Engine: Loaded")
    print("Pre-Test Conditions Module: Loaded")
    print("Visualization Module: Loaded")
    app.run(debug=True, port=5000)
