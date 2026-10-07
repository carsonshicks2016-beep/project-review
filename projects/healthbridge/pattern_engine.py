"""
HealthBridge AI - Cross-Source Pattern Recognition Engine
Detects clinically meaningful patterns across genomic + biomarker data.

Architecture:
  Layer 1: Hardcoded pattern library — high-confidence, curated multi-variable patterns
  Layer 2: AI-driven scan — finds emergent patterns not in the library

Each pattern has:
  - id, name, severity (critical/warning/insight)
  - required_snps: rsids that must be present with risk allele (any match = trigger)
  - required_biomarkers: {biomarker_key: condition} — conditions checked against evaluated results
  - description: plain-language explanation
  - implications: what this combined pattern means clinically
  - actions: specific actionable recommendations
"""

import json
import boto3

# =============================================================================
# SEVERITY LEVELS
# critical  = compounding risks requiring prompt attention
# warning   = suboptimal pattern worth addressing
# insight   = interesting cross-source finding, may be protective or neutral
# =============================================================================

def check_wearable_condition(metric_path: str, condition_fn, wearable_metrics: dict) -> bool:
    """
    Check a wearable metric condition using dot-path notation.
    e.g. metric_path = "sleep.avg_total_hours"
    """
    if not wearable_metrics:
        return False
    try:
        parts = metric_path.split(".")
        val = wearable_metrics
        for p in parts:
            val = val.get(p) if isinstance(val, dict) else None
            if val is None:
                return False
        return condition_fn(float(val))
    except (TypeError, ValueError):
        return False

PATTERN_LIBRARY = [

    # =========================================================================
    # METABOLIC / INSULIN RESISTANCE PATTERNS
    # =========================================================================
    {
        "id": "insulin_resistance_triad",
        "name": "Emerging Insulin Resistance Triad",
        "severity": "critical",
        "category": "Metabolic",
        "required_snps": ["rs7903146", "rs12255372", "rs5219", "rs1044498"],
        "required_biomarkers": {
            "glucose": lambda v: v > 88,
            "insulin": lambda v: v > 7,
        },
        "optional_biomarkers": {
            "hba1c": lambda v: v > 5.2,
            "triglycerides": lambda v: v > 100,
        },
        "description": "Your genetics include variants that impair beta cell insulin secretion and reduce insulin sensitivity. Combined with fasting glucose and insulin levels trending above optimal, this pattern reflects early insulin resistance — often years before HbA1c or glucose cross clinical thresholds.",
        "implications": "Standard lab reference ranges would classify these values as 'normal,' but the combination of genetic predisposition and trending biomarkers indicates the metabolic stress is already present. This pattern predicts progression to pre-diabetes without intervention.",
        "references": [
            {"title": "Variant of transcription factor 7-like 2 (TCF7L2) gene confers risk of type 2 diabetes", "journal": "Nature Genetics", "year": 2006, "pmid": "16415884"},
            {"title": "The TCF7L2 gene and type 2 diabetes in the Diabetes Prevention Program", "journal": "Diabetes", "year": 2006, "pmid": "16567547"}
        ],
        "actions": [
            "Target fasting glucose below 85 mg/dL and fasting insulin below 6 uIU/mL as personal goals — stricter than standard reference ranges",
            "Prioritize time-restricted eating (8-10 hour window) — shown to improve insulin sensitivity independently of caloric intake",
            "Eliminate liquid calories (juice, soda, alcohol) and refined carbohydrates — these directly drive the glucose/insulin cycle",
            "Add 10-minute post-meal walks — reduces postprandial glucose spikes by 20-30%",
            "Test HOMA-IR to quantify insulin resistance degree; retest every 6 months",
            "Consider continuous glucose monitor (CGM) trial to identify personal glucose spike triggers"
        ]
    },
    {
        "id": "mthfr_b12_homocysteine",
        "name": "Active Methylation Insufficiency",
        "severity": "critical",
        "category": "Methylation",
        "required_snps": ["rs1801133", "rs1801131", "rs1801394", "rs602662"],
        "required_biomarkers": {
            "homocysteine": lambda v: v > 9,
        },
        "optional_biomarkers": {
            "vitamin_b12": lambda v: v < 600,
            "folate": lambda v: v < 12,
        },
        "description": "MTHFR and/or MTRR genetic variants are impairing your body's ability to convert folate and B12 into their active methylated forms. Your homocysteine level confirms this methylation bottleneck is active — not just a genetic predisposition but a current functional problem.",
        "implications": "Elevated homocysteine is an independent risk factor for cardiovascular disease, cognitive decline, and stroke. It also impairs DNA methylation, neurotransmitter synthesis, and glutathione production. Standard B12 and folate levels can appear 'normal' while methylation is functionally compromised because the standard forms of these nutrients are poorly utilized with MTHFR variants.",
        "references": [
            {"title": "A candidate genetic risk factor for vascular disease: a common mutation in methylenetetrahydrofolate reductase", "journal": "Nature Genetics", "year": 1995, "pmid": "7647779"},
            {"title": "MTHFR C677T polymorphism and cardiovascular disease: a scientific statement from the American Heart Association", "journal": "Circulation", "year": 2015, "pmid": "25666504"}
        ],
        "actions": [
            "Switch immediately to methylfolate (L-5-MTHF, 400-800 mcg/day) — folic acid is poorly converted with MTHFR variants and may actually compete with methylfolate",
            "Use methylcobalamin or adenosylcobalamin B12 (1000 mcg/day sublingually) — not cyanocobalamin",
            "Add P5P (pyridoxal-5-phosphate, active B6, 25-50 mg/day) — cofactor in homocysteine remethylation",
            "Retest homocysteine in 8-12 weeks after starting methylated B vitamins — target below 8 umol/L",
            "Test RBC folate (not serum folate) for more accurate intracellular status",
            "Minimize alcohol — directly depletes folate and raises homocysteine"
        ]
    },
    {
        "id": "vitamin_d_transport_bottleneck",
        "name": "Vitamin D Transport & Activation Bottleneck",
        "severity": "warning",
        "category": "Vitamins",
        "required_snps": ["rs10741657", "rs2282679", "rs12785878", "rs2228570", "rs731236", "rs1544410"],
        "required_biomarkers": {
            "vitamin_d": lambda v: v < 60,
        },
        "optional_biomarkers": {},
        "description": "You carry variants affecting multiple steps of vitamin D metabolism — synthesis in the skin (DHCR7), transport in the blood (GC/vitamin D binding protein), conversion to active form (CYP2R1), and cellular response (VDR). Your serum 25-OH vitamin D level, even if technically 'sufficient' by standard ranges, may represent functional insufficiency for your genotype.",
        "implications": "With stacking variants in the vitamin D pathway, the standard sufficiency threshold of 30 ng/mL is inadequate. Research on VDR and GC variants suggests these individuals require higher serum levels to achieve equivalent tissue-level vitamin D activity. Functional vitamin D insufficiency affects immune function, bone density, muscle strength, cardiovascular health, and mood.",
        "actions": [
            "Target serum 25-OH vitamin D of 60-80 ng/mL — not merely 'sufficient' (>30) by standard ranges",
            "Start vitamin D3 at 5,000-8,000 IU/day with vitamin K2 (MK-7, 100-200 mcg) — K2 directs calcium properly and prevents arterial calcification",
            "Take vitamin D with the largest meal of the day — it is fat-soluble and absorption increases significantly with dietary fat",
            "Consider testing 1,25-OH vitamin D (active form) to assess conversion efficiency",
            "Recheck 25-OH vitamin D after 90 days of supplementation to calibrate dose",
            "Daily midday sun exposure (15-30 min, arms and legs exposed) provides 5,000-10,000 IU equivalent but is limited by latitude, season, and skin tone"
        ]
    },
    {
        "id": "omega3_conversion_deficiency",
        "name": "Omega-3 Conversion Deficiency",
        "severity": "warning",
        "category": "Nutrition",
        "required_snps": ["rs174537", "rs1535", "rs174575"],
        "required_biomarkers": {
            "triglycerides": lambda v: v > 100,
            "hdl": lambda v: v < 60,
        },
        "optional_biomarkers": {
            "hscrp": lambda v: v > 1.0,
        },
        "description": "FADS1/FADS2 variants severely limit your ability to convert plant-based omega-3 (ALA from flaxseed, chia, walnuts) into the biologically active forms EPA and DHA. Your lipid panel shows a pattern consistent with omega-3 insufficiency: elevated triglycerides and suboptimal HDL.",
        "implications": "Plant-based omega-3 sources are essentially ineffective for individuals with these variants — conversion rates may be as low as 1-5% of normal. Without adequate EPA and DHA, inflammatory resolution is impaired, triglyceride clearance is reduced, and HDL production is suboptimal. Standard dietary advice to 'eat more plant omega-3s' would be insufficient for this genotype.",
        "actions": [
            "Supplement directly with marine-source EPA+DHA (fish oil or algae oil) — 2-4g EPA+DHA per day is the target range for this genotype",
            "Consider Omega-3 Index testing (a blood test measuring EPA+DHA in red blood cell membranes) — target 8-12%",
            "Choose triglyceride-form fish oil over ethyl ester form — superior absorption (look for labels specifying 'natural triglyceride form')",
            "Eat oily fish 3-4x per week (salmon, sardines, mackerel, herring) in addition to supplementation",
            "Retest triglycerides and HDL in 90 days after consistent supplementation",
            "Take fish oil with meals containing fat to maximize absorption"
        ]
    },

    # =========================================================================
    # CARDIOVASCULAR PATTERNS
    # =========================================================================
    {
        "id": "apoe4_lipid_risk",
        "name": "APOE4 Cardiovascular Risk Profile",
        "severity": "critical",
        "category": "Cardiovascular",
        "required_snps": ["rs429358"],
        "required_biomarkers": {
            "ldl": lambda v: v > 90,
        },
        "optional_biomarkers": {
            "hscrp": lambda v: v > 1.0,
            "apob": lambda v: v > 80,
            "lpa": lambda v: v > 30,
        },
        "description": "You carry the APOE4 allele — the most significant common genetic risk factor for both cardiovascular disease and late-onset Alzheimer's disease. APOE4 impairs LDL clearance from the bloodstream and promotes inflammatory lipid metabolism. Your LDL level, even at 'borderline' values, carries substantially higher cardiovascular risk for your genotype than it would for someone without APOE4.",
        "implications": "For APOE4 carriers, the optimal LDL target is below 70 mg/dL rather than the standard <100 mg/dL. LDL-C can also be misleading in APOE4 carriers — ApoB is a more accurate measure of atherogenic particle burden. APOE4 also increases sensitivity to dietary saturated fat, meaning saturated fat raises LDL more dramatically in this genotype than in APOE3/3 individuals.",
        "actions": [
            "Set a personal LDL target of <70 mg/dL — discuss with your physician; this may warrant lipid-lowering therapy",
            "Request ApoB testing — this is the most important lipid metric for APOE4 carriers; target <70 mg/dL",
            "Dramatically reduce saturated fat (red meat, dairy fat, coconut oil) — APOE4 carriers show larger LDL increases from saturated fat than other genotypes",
            "Replace saturated fat with monounsaturated fats (olive oil, avocado, nuts) — shown to reduce LDL in APOE4 carriers",
            "Prioritize aerobic exercise 150+ minutes per week — directly improves LDL clearance",
            "Consider requesting a coronary artery calcium (CAC) score — a non-invasive measure of subclinical atherosclerosis that is highly informative for APOE4 carriers",
            "Monitor cognitive health proactively — APOE4 increases late-onset Alzheimer's risk 3-4x (heterozygous) or 8-12x (homozygous)"
        ]
    },
    {
        "id": "thrombosis_risk_compounded",
        "name": "Compounded Thrombosis Risk",
        "severity": "critical",
        "category": "Cardiovascular",
        "required_snps": ["rs6025", "rs1799963"],
        "required_biomarkers": {},
        "optional_biomarkers": {
            "hscrp": lambda v: v > 1.5,
            "homocysteine": lambda v: v > 10,
            "platelets": lambda v: v > 300,
        },
        "description": "Factor V Leiden (rs6025) and/or Prothrombin G20210A (rs1799963) are among the most significant inherited thrombophilia variants. These variants cause hypercoagulability — an increased tendency for blood to clot.",
        "implications": "Factor V Leiden heterozygosity increases venous thromboembolism (DVT/PE) risk 5-10x; homozygosity increases it 50-100x. When combined with elevated homocysteine or inflammatory markers, thrombotic risk compounds significantly. Long periods of immobility, oral contraceptives, pregnancy, and surgery are particularly high-risk situations.",
        "actions": [
            "Inform all healthcare providers about this variant — it affects surgical planning, contraceptive choices, and pregnancy management",
            "Avoid prolonged immobility — get up and move every 60-90 minutes during long travel or desk work",
            "Stay well hydrated — dehydration increases blood viscosity and clot risk",
            "Discuss aspirin therapy with your physician — low-dose aspirin may be appropriate depending on overall risk profile",
            "Women with this variant should discuss contraceptive options with their OB/GYN — combined oral contraceptives containing estrogen increase clot risk substantially",
            "Compression stockings during long flights or car rides",
            "Consider a hematology consultation for comprehensive thrombophilia workup"
        ]
    },
    {
        "id": "inflammation_cardiovascular_convergence",
        "name": "Inflammation-Cardiovascular Convergence",
        "severity": "warning",
        "category": "Cardiovascular",
        "required_snps": ["rs1800629", "rs1800795", "rs1800871"],
        "required_biomarkers": {
            "hscrp": lambda v: v > 1.5,
            "ldl": lambda v: v > 100,
        },
        "optional_biomarkers": {
            "homocysteine": lambda v: v > 10,
        },
        "description": "Pro-inflammatory genetic variants (TNF-alpha, IL-6, IL-10) combined with elevated hs-CRP and borderline LDL create a compounding cardiovascular risk profile. Inflammation accelerates LDL oxidation and plaque formation — meaning the cardiovascular risk from borderline LDL is substantially higher in a pro-inflammatory genetic context.",
        "implications": "The JUPITER trial demonstrated that even with normal LDL, elevated hs-CRP independently predicts cardiovascular events. With both elevated LDL and elevated hs-CRP driven partly by genetics, the combined risk exceeds what either factor alone would suggest. Oxidized LDL (not total LDL) is the actual atherogenic particle.",
        "actions": [
            "Target hs-CRP below 1.0 mg/L — treat this as aggressively as LDL",
            "Prioritize anti-inflammatory diet: Mediterranean pattern, high omega-3, minimize refined carbohydrates and seed oils",
            "Optimize sleep to 7.5-9 hours — sleep deprivation is one of the most potent drivers of systemic inflammation",
            "Address gut health — intestinal permeability is a major driver of systemic TNF-alpha elevation",
            "Request PON1 activity testing or oxidized LDL — more relevant than total LDL for this inflammatory genotype",
            "Consider requesting a Cardio IQ advanced lipid panel (LDL particle number, LDL-P) rather than relying on standard LDL-C"
        ]
    },

    # =========================================================================
    # HORMONAL PATTERNS
    # =========================================================================
    {
        "id": "testosterone_shbg_trap",
        "name": "Testosterone-SHBG Binding Trap",
        "severity": "warning",
        "category": "Hormonal",
        "required_snps": ["rs1799941", "rs6257", "rs6152"],
        "required_biomarkers": {
            "testosterone_total": lambda v: v < 700,
            "shbg": lambda v: v > 35,
        },
        "optional_biomarkers": {
            "testosterone_free": lambda v: v < 12,
            "insulin": lambda v: v > 8,
        },
        "description": "SHBG genetic variants combined with elevated SHBG and total testosterone below optimal create a 'binding trap' — a significant proportion of circulating testosterone is bound and biologically inactive. Total testosterone appearing 'normal' can mask functionally low free testosterone.",
        "implications": "The androgen receptor (AR) variant further affects how testosterone signals in target tissues. Free testosterone, not total, determines actual androgen activity. High SHBG is commonly driven by low insulin (which is metabolically healthy but hormonally counterproductive in this context), thyroid dysfunction, or genetic predisposition.",
        "actions": [
            "Test free testosterone and SHBG if not already done — these are more informative than total testosterone alone",
            "Review thyroid function — elevated TSH reduces SHBG clearance",
            "Resistance training 3-4x per week transiently lowers SHBG and raises free testosterone",
            "Adequate dietary fat intake (especially saturated and monounsaturated fat) supports testosterone synthesis — very low-fat diets suppress testosterone production",
            "Optimize zinc status (target 80-120 ug/dL) — zinc is a cofactor in testosterone synthesis and SHBG regulation",
            "Ensure sleep is 7.5-9 hours — 70% of testosterone production occurs during sleep, primarily in REM"
        ]
    },
    {
        "id": "estrogen_clearance_slow",
        "name": "Impaired Estrogen Clearance",
        "severity": "warning",
        "category": "Hormonal",
        "required_snps": ["rs4680", "rs10046", "rs1056836", "rs700518"],
        "required_biomarkers": {
            "estradiol": lambda v: v > 30,
        },
        "optional_biomarkers": {
            "shbg": lambda v: v < 25,
        },
        "description": "COMT Val158Met (slow variant), CYP19A1 (aromatase), and CYP1B1 variants combine to create impaired estrogen catabolism. Slow COMT reduces breakdown of catechol estrogens; CYP1B1 variants increase production of genotoxic 4-OH estrogen metabolites; aromatase variants affect testosterone-to-estradiol conversion rate.",
        "implications": "Elevated estrogen with impaired clearance leads to accumulation of estrogen and its metabolites. In men, this increases estrogen-related side effects and may suppress testosterone. In women, this pattern is associated with estrogen dominance symptoms and increased breast cancer risk over time. The 4-OH estrogen metabolites produced by CYP1B1 are directly genotoxic.",
        "actions": [
            "Eat cruciferous vegetables daily (broccoli, cauliflower, Brussels sprouts, kale) — contain DIM and I3C which support healthy estrogen metabolism toward protective 2-OH pathway",
            "Consider DIM (diindolylmethane) supplementation 100-200 mg/day — promotes 2-OH estrogen metabolite production",
            "Calcium-D-glucarate (500-1000 mg/day) inhibits beta-glucuronidase, supporting estrogen conjugation and excretion",
            "Support methylation (methylfolate, methylcobalamin) — COMT requires SAMe as methyl donor for estrogen breakdown",
            "Minimize alcohol — directly inhibits estrogen metabolism and raises estradiol",
            "Reduce body fat if elevated — adipose tissue is a major aromatase site, converting testosterone to estrogen",
            "Test urinary estrogen metabolites (DUTCH test) to assess 2-OH vs 4-OH vs 16-OH estrogen ratio"
        ]
    },

    # =========================================================================
    # DETOX / OXIDATIVE STRESS PATTERNS
    # =========================================================================
    {
        "id": "detox_impairment_stack",
        "name": "Phase I/II Detoxification Impairment Stack",
        "severity": "warning",
        "category": "Detoxification",
        "required_snps": ["rs1695", "rs1800566", "rs1801280", "rs4880"],
        "required_biomarkers": {
            "ggtp": lambda v: v > 20,
        },
        "optional_biomarkers": {
            "alt": lambda v: v > 25,
            "hscrp": lambda v: v > 1.0,
        },
        "description": "Multiple detoxification enzyme variants are stacking: GSTP1 (glutathione S-transferase, Phase II), NQO1 (quinone oxidoreductase), NAT2 (acetylation), and SOD2 (mitochondrial antioxidant). Combined with elevated GGT — a sensitive marker of glutathione demand and oxidative stress — this indicates your detoxification capacity is under load.",
        "implications": "GGT elevation in the context of impaired phase II detox reflects increased glutathione turnover — your body is using glutathione faster than it can produce it. This leaves you more vulnerable to environmental toxins, carcinogens, and oxidative damage. Slow NAT2 acetylation means certain drugs and aromatic amines (found in overcooked meat) are processed more slowly and accumulate.",
        "actions": [
            "N-acetylcysteine (NAC) 600-900 mg/day — direct glutathione precursor, supports both phase I and II detox",
            "Alpha-lipoic acid 300-600 mg/day — regenerates glutathione and other antioxidants",
            "Minimize environmental toxin exposure: filter drinking water, choose organic produce for high-pesticide items, avoid plastic food containers",
            "Avoid well-done or charred meat — produces aromatic amines that are poorly cleared with NAT2 slow variants",
            "Riboflavin (B2) 25-50 mg/day — cofactor for NQO1 and glutathione reductase",
            "Selenium 100-200 mcg/day — essential cofactor for glutathione peroxidase",
            "Consider periodic liver function testing (ALT, GGT) to monitor detox load"
        ]
    },

    # =========================================================================
    # SLEEP / METABOLIC INTERACTION PATTERNS
    # =========================================================================
    {
        "id": "circadian_glucose_disruption",
        "name": "Circadian-Glucose Metabolic Disruption",
        "severity": "warning",
        "category": "Metabolic/Sleep",
        "required_snps": ["rs10830963", "rs1801260", "rs11121022"],
        "required_biomarkers": {
            "glucose": lambda v: v > 88,
        },
        "optional_biomarkers": {
            "insulin": lambda v: v > 7,
            "hba1c": lambda v: v > 5.2,
        },
        "description": "MTNR1B (melatonin receptor) variants impair the normal suppression of insulin secretion during nighttime melatonin signaling, directly elevating fasting glucose. CLOCK gene variants cause circadian phase delay (eveningness tendency) which disrupts cortisol and glucose rhythms. Combined with glucose trending above optimal, this pattern indicates circadian biology is actively worsening metabolic health.",
        "implications": "Eating late at night is particularly harmful for MTNR1B variants — melatonin is elevated in the evening and directly antagonizes insulin secretion via the same receptor, causing glucose to remain elevated longer after late meals. Shift work or irregular sleep schedules are especially damaging for this genotype.",
        "actions": [
            "Establish a strict eating cutoff 3-4 hours before bedtime — this is more important for your genotype than for average individuals",
            "Front-load calories earlier in the day — breakfast and lunch larger than dinner",
            "Consistent sleep and wake times daily (including weekends) — circadian disruption directly worsens MTNR1B-related glucose dysregulation",
            "Morning light exposure within 30 minutes of waking — anchors circadian rhythm and improves glucose metabolism throughout the day",
            "Avoid late-night snacking entirely — even small carbohydrate loads cause exaggerated glucose elevation when melatonin is high",
            "Consider CGM to visualize your personal glucose response to meal timing"
        ]
    },

    # =========================================================================
    # THYROID / AUTOIMMUNE PATTERNS
    # =========================================================================
    {
        "id": "hashimotos_early_pattern",
        "name": "Early Hashimoto's Risk Pattern",
        "severity": "warning",
        "category": "Autoimmune/Thyroid",
        "required_snps": ["rs2476601", "rs3087243"],
        "required_biomarkers": {
            "tsh": lambda v: v > 2.0,
        },
        "optional_biomarkers": {
            "free_t3": lambda v: v < 3.2,
            "hscrp": lambda v: v > 1.0,
        },
        "description": "PTPN22 and CTLA4 autoimmune susceptibility variants combined with TSH trending above 2.0 mIU/L creates a pattern consistent with early or subclinical Hashimoto's thyroiditis — the most common cause of hypothyroidism, which proceeds through a silent autoimmune phase for years before TSH becomes overtly abnormal.",
        "implications": "Standard TSH reference range extends to 4.5 mIU/L, but functional medicine and longevity-oriented practitioners target TSH below 2.0 as the optimal range. TSH trending from 1.5 toward 2.5 over years, in someone with autoimmune genetic variants, often reflects subclinical Hashimoto's destroying thyroid tissue gradually. Identifying this pattern early allows intervention before overt hypothyroidism develops.",
        "actions": [
            "Request thyroid antibody testing: TPO antibodies (anti-thyroid peroxidase) and Tg antibodies (anti-thyroglobulin) — these confirm autoimmune thyroid disease before TSH becomes overtly elevated",
            "Optimize selenium: 100-200 mcg/day — selenium is essential for thyroid hormone production and has been shown to reduce TPO antibody levels",
            "Test and optimize vitamin D to 60-80 ng/mL — vitamin D deficiency is strongly associated with autoimmune thyroid disease",
            "Consider gluten elimination trial — gluten molecular mimicry with thyroid tissue is well-documented in Hashimoto's; 3-month trial with antibody retest",
            "Address gut permeability — autoimmune conditions are strongly linked to intestinal hyperpermeability; L-glutamine, zinc carnosine, and butyrate support gut barrier",
            "Track TSH, Free T4, and Free T3 annually — longitudinal trend is more informative than any single measurement"
        ]
    },

    # =========================================================================
    # COGNITIVE / NEUROLOGICAL PATTERNS
    # =========================================================================
    {
        "id": "bdnf_cognitive_vulnerability",
        "name": "BDNF-Dependent Cognitive Vulnerability",
        "severity": "warning",
        "category": "Cognitive",
        "required_snps": ["rs6265"],
        "required_biomarkers": {
            "vitamin_d": lambda v: v < 50,
            "hscrp": lambda v: v > 1.5,
        },
        "optional_biomarkers": {
            "homocysteine": lambda v: v > 10,
        },
        "description": "The BDNF Val66Met variant (Met allele) reduces activity-dependent BDNF secretion — the brain's primary neuroplasticity factor. Combined with low vitamin D (which regulates BDNF gene expression) and elevated inflammation (which suppresses BDNF), this pattern indicates neuroplasticity is being actively impaired by modifiable factors.",
        "implications": "BDNF deficiency is linked to depression, anxiety, cognitive decline, and reduced learning capacity. The Met variant makes BDNF levels more sensitive to lifestyle factors — both more vulnerable to depletion by poor lifestyle and more responsive to improvement through targeted interventions. This pattern is especially actionable because the modifiable drivers (vitamin D, inflammation) can be addressed.",
        "actions": [
            "Optimize vitamin D to 60-80 ng/mL — vitamin D directly upregulates BDNF gene expression",
            "Aerobic exercise is the most potent BDNF stimulator known — aim for 30+ minutes of elevated heart rate exercise at least 4x per week",
            "Reduce hs-CRP below 1.0 mg/L — inflammation directly suppresses BDNF synthesis",
            "Intermittent fasting (16:8 or equivalent) — shown to increase BDNF by 50-400% in animal models, human evidence emerging",
            "Omega-3 fatty acids (EPA+DHA 2g+/day) — support BDNF signaling and neuroinflammation resolution",
            "Address homocysteine if elevated — hyperhomocysteinemia is directly neurotoxic and impairs BDNF signaling"
        ]
    },

    # =========================================================================
    # BONE & JOINT PATTERNS
    # =========================================================================
    {
        "id": "bone_density_triple_threat",
        "name": "Bone Density Triple Vulnerability",
        "severity": "warning",
        "category": "Bone Health",
        "required_snps": ["rs2234693", "rs2228570", "rs1800012"],
        "required_biomarkers": {
            "vitamin_d": lambda v: v < 50,
        },
        "optional_biomarkers": {
            "calcium": lambda v: v < 9.2,
            "magnesium": lambda v: v < 2.0,
        },
        "description": "ESR1 (estrogen receptor), VDR (vitamin D receptor), and COL1A1 (collagen I) variants combine to create compounding bone density vulnerability. Estrogen receptor variants reduce bone protection from estrogen; VDR variants impair calcium absorption and bone mineralization signaling; COL1A1 variants reduce collagen matrix quality. Combined with suboptimal vitamin D, active bone loss is likely occurring.",
        "implications": "Each variant independently increases fracture risk; together they create a significantly elevated risk profile that begins in the 20s-30s as peak bone mass is established. This is the window during which intervention has the highest impact — bone density built now determines fracture risk decades later.",
        "actions": [
            "Achieve 25-OH vitamin D of 60-80 ng/mL immediately — this is the most critical intervention for VDR variant individuals",
            "Ensure adequate calcium intake: 1000-1200 mg/day from food sources (dairy, fortified foods, leafy greens) — avoid calcium supplements without vitamin D and K2",
            "Vitamin K2 (MK-7) 100-200 mcg/day — activates osteocalcin for bone matrix mineralization and directs calcium to bone rather than arteries",
            "Magnesium glycinate 300-400 mg/day — required for vitamin D activation and bone matrix formation",
            "Weight-bearing exercise and resistance training — the only proven stimulus for osteoblast activity and new bone formation",
            "Collagen peptides 10-15g/day — support COL1A1-variant collagen matrix quality; take with vitamin C",
            "Consider baseline DEXA scan — establishes bone density baseline for longitudinal monitoring given this genetic profile"
        ]
    },

    # =========================================================================
    # PROTECTIVE / POSITIVE PATTERNS
    # =========================================================================
    {
        "id": "longevity_foxo3_protective",
        "name": "FOXO3 Longevity Advantage",
        "severity": "insight",
        "category": "Longevity",
        "required_snps": ["rs2802292", "rs13217795"],
        "required_biomarkers": {
            "hscrp": lambda v: v < 1.0,
            "glucose": lambda v: v < 90,
        },
        "optional_biomarkers": {},
        "description": "FOXO3 longevity variants combined with low inflammation and optimal glucose create a profile associated with exceptional longevity in population studies. FOXO3 encodes a transcription factor that promotes cellular stress resistance, autophagy, and DNA repair — consistently associated with centenarian status across multiple populations.",
        "implications": "This is a genuinely protective pattern. The lifestyle behaviors that activate FOXO3 — caloric moderation, exercise, low inflammation, ketosis — are the same behaviors maintaining your other biomarkers in optimal range. This suggests your lifestyle choices are aligned with maximizing your genetic longevity potential.",
        "actions": [
            "Continue and maintain the metabolic behaviors that keep glucose and inflammation optimal — these directly activate FOXO3 pathways",
            "Periodic fasting or caloric restriction — FOXO3 is strongly activated by energy stress and AMPK signaling",
            "High-intensity interval training periodically — acute metabolic stress is a potent FOXO3 activator",
            "Avoid chronic overnutrition — sustained caloric excess suppresses FOXO3 via insulin/IGF-1 signaling",
            "This genetic advantage is conditional — it requires the lifestyle context to express. Maintain current trajectory."
        ]
    },
    {
        "id": "gilbert_syndrome_antioxidant",
        "name": "Gilbert's Syndrome — Antioxidant Advantage",
        "severity": "insight",
        "category": "Hepatic",
        "required_snps": ["rs4148323", "rs8175347"],
        "required_biomarkers": {
            "bilirubin_total": lambda v: v > 1.2,
            "alt": lambda v: v < 30,
            "ast": lambda v: v < 30,
        },
        "optional_biomarkers": {},
        "description": "UGT1A1 variant combined with mildly elevated total bilirubin and normal liver enzymes is the classic presentation of Gilbert's syndrome — a benign hereditary condition affecting bilirubin conjugation. This is not liver disease.",
        "implications": "Gilbert's syndrome is associated with lower rates of cardiovascular disease and certain cancers. Bilirubin is a potent antioxidant — mildly elevated levels provide ongoing antioxidant protection. The elevated bilirubin in isolation with normal ALT/AST requires no treatment. However, certain medications (including some commonly prescribed drugs) require dose adjustment in UGT1A1 variants — this includes irinotecan (cancer treatment) and some HIV medications.",
        "actions": [
            "No treatment required — this is a benign condition with likely protective effects",
            "Inform healthcare providers of UGT1A1 variant status — important for medication dosing especially if irinotecan chemotherapy is ever considered",
            "Avoid prolonged fasting, which can temporarily raise bilirubin further in Gilbert's syndrome",
            "Minimize alcohol — while Gilbert's itself is benign, alcohol strains the same UGT1A1 enzyme system"
        ]
    },

    # =========================================================================
    # WEARABLE + GENOMIC + BIOMARKER PATTERNS
    # =========================================================================
    {
        "id": "hrv_inflammation_genomic",
        "name": "HRV Suppression from Genetic Inflammation Load",
        "severity": "critical",
        "category": "Recovery/Inflammation",
        "required_snps": ["rs1800629", "rs1800795", "rs1800871", "rs1143634"],
        "required_biomarkers": {},
        "required_wearable": {"hrv.avg_rmssd": lambda v: v < 55},
        "optional_biomarkers": {"hscrp": lambda v: v > 1.0},
        "optional_wearable": {
            "hrv.hrv_trend": lambda v: v == "declining",
            "recovery.avg_readiness_score": lambda v: v < 70,
        },
        "description": "Pro-inflammatory genetic variants (TNF-alpha, IL-6, IL-10, IL-1B) are chronically suppressing your autonomic nervous system recovery capacity, reflected in below-optimal HRV. This is not just poor lifestyle — your baseline inflammatory setpoint is genetically elevated, creating a persistent drag on HRV that standard sleep and stress advice will not fully resolve.",
        "implications": "Chronically suppressed HRV from genetic inflammation is one of the most underrecognized compounding risk patterns. Low HRV independently predicts cardiovascular events, impaired glucose regulation, poor stress resilience, and faster biological aging. The genetic driver means the primary intervention must be anti-inflammatory — not just 'sleep more.'",
        "actions": [
            "Target hs-CRP below 0.5 mg/L as your primary HRV optimization goal — inflammation is your primary HRV suppressor",
            "Omega-3 supplementation (EPA+DHA 3-4g/day) — the most evidence-backed anti-inflammatory intervention for HRV improvement",
            "Cold exposure (cold shower or ice bath 3-4x/week) — directly increases vagal tone and HRV within 4-6 weeks",
            "Eliminate seed oils (canola, soybean, corn oil) — linoleic acid drives arachidonic acid inflammation cascade",
            "Time-restricted eating (16:8) — reduces inflammatory cytokines and improves HRV in 8-12 weeks",
            "Magnesium glycinate 400mg before bed — reduces NF-kB inflammatory signaling and improves HRV during sleep",
            "Track HRV trend weekly — expect 4-8 weeks to see measurable improvement after implementing anti-inflammatory protocol"
        ]
    },
    {
        "id": "sleep_mthfr_cognitive_risk",
        "name": "Sleep-Methylation Cognitive Risk Convergence",
        "severity": "critical",
        "category": "Cognitive/Sleep",
        "required_snps": ["rs1801133", "rs6265"],
        "required_biomarkers": {},
        "required_wearable": {"sleep.avg_rem_hours": lambda v: v < 1.5},
        "optional_biomarkers": {
            "homocysteine": lambda v: v > 9,
            "vitamin_b12": lambda v: v < 600,
        },
        "optional_wearable": {"sleep.avg_total_hours": lambda v: v < 7.5},
        "description": "MTHFR-impaired methylation combined with BDNF Val66Met and chronically low REM sleep creates a triple threat to cognitive health. REM sleep is when BDNF is produced and memory consolidation occurs — the Met variant already reduces baseline BDNF secretion, and poor REM compounds this further. Elevated homocysteine from MTHFR is directly neurotoxic to the hippocampus over time.",
        "implications": "Each factor independently reduces cognitive resilience; together they compound in a self-reinforcing loop. Poor REM → less BDNF → weaker memory consolidation → more cognitive stress → worse sleep. The window for intervention is now — this pattern predicts accelerated cognitive aging.",
        "actions": [
            "Prioritize REM sleep above all else — your genetics make REM disproportionately valuable; target 1.8-2.0h nightly",
            "Eliminate alcohol entirely — even one drink reduces REM by 20-25%, especially damaging for this genotype",
            "Start methylfolate 400-800mcg + methylcobalamin 1000mcg daily immediately — reduces homocysteine and supports BDNF methylation",
            "Aerobic exercise 30+ min 4x/week is the most potent BDNF stimulator available — non-negotiable for this genotype",
            "Keep bedroom at 65-68°F — lower ambient temperature significantly increases REM sleep duration",
            "Consistent bedtime within 30 minutes — REM occurs predominantly in the second half of sleep and is first disrupted by irregular schedules",
            "Test homocysteine every 6 months — your primary cognitive aging biomarker given this profile"
        ]
    },
    {
        "id": "sleep_glucose_genetic_compounding",
        "name": "Sleep Deprivation Amplifying Genetic Glucose Risk",
        "severity": "critical",
        "category": "Metabolic/Sleep",
        "required_snps": ["rs7903146", "rs10830963"],
        "required_biomarkers": {"glucose": lambda v: v > 88},
        "required_wearable": {"sleep.avg_total_hours": lambda v: v < 7.0},
        "optional_biomarkers": {"insulin": lambda v: v > 7},
        "optional_wearable": {"hrv.hrv_trend": lambda v: v == "declining"},
        "description": "TCF7L2 and MTNR1B variants impair glucose regulation genetically, and your wearable data shows average sleep below 7 hours — which independently raises fasting glucose 10-15% and increases insulin resistance up to 30%. You are experiencing genetic AND behavioral glucose dysregulation simultaneously, confirmed by your blood work.",
        "implications": "Sleep deprivation is one of the most potent and underrecognized drivers of insulin resistance. For individuals with TCF7L2 and MTNR1B variants, this effect is amplified. Your fasting glucose is likely meaningfully worse on short-sleep nights. This pattern will worsen progressively if sleep remains chronically shortened.",
        "actions": [
            "Treat sleep extension as a primary metabolic drug — going from 6.5h to 7.5h will improve fasting glucose comparably to early metformin",
            "Track fasting glucose on good vs poor sleep nights — the correlation will be obvious within 2 weeks",
            "Hard sleep schedule: in bed by 10:30pm, no screens after 9pm, consistent even on weekends",
            "Magnesium glycinate 400mg + glycine 3g before bed — improve sleep depth and independently improve insulin sensitivity",
            "No eating within 3 hours of bedtime — critical for MTNR1B variants where melatonin blocks insulin secretion",
            "Morning exercise before breakfast — depletes glycogen and improves insulin sensitivity for the entire day"
        ]
    },
    {
        "id": "low_activity_apoe4_risk",
        "name": "Physical Inactivity Amplifying APOE4 Risk",
        "severity": "critical",
        "category": "Cardiovascular/Cognitive",
        "required_snps": ["rs429358"],
        "required_biomarkers": {},
        "required_wearable": {"activity.avg_daily_steps": lambda v: v < 7500},
        "optional_biomarkers": {
            "ldl": lambda v: v > 90,
            "hscrp": lambda v: v > 1.0,
        },
        "optional_wearable": {
            "activity.sedentary_days_pct": lambda v: v > 25,
            "heart_rate.avg_resting_hr": lambda v: v > 65,
        },
        "description": "APOE4 carriers who are physically inactive represent one of the highest-risk modifiable patterns in precision health. Aerobic exercise is one of the few interventions proven to reduce APOE4-associated Alzheimer's risk — it upregulates BDNF, improves amyloid clearance via glymphatic function, and directly lowers LDL and inflammation. Your step count indicates insufficient aerobic activity to counteract these genetic risks.",
        "implications": "For APOE4 carriers, physical inactivity is not just a general cardiovascular risk — it is a primary driver of Alzheimer's risk accumulation. APOE4 carriers who exercise regularly have Alzheimer's risk comparable to non-APOE4 individuals who are sedentary. Exercise is essentially the most powerful gene-modifying intervention available for this genotype.",
        "actions": [
            "150+ minutes of zone 2 aerobic exercise per week is non-negotiable for APOE4 carriers — this is the threshold where Alzheimer's risk reduction becomes significant",
            "Zone 2 specifically (conversational pace, 60-70% max HR) — maximizes mitochondrial biogenesis and BDNF production",
            "Target 8,500+ daily steps as your minimum floor — park farther, take stairs, walk after meals",
            "Add 2x/week resistance training — independently improves LDL clearance and insulin sensitivity",
            "Track resting heart rate trend — target below 60 bpm as your cardiovascular fitness marker",
            "Morning exercise preferred — greater BDNF elevation and metabolic benefits than evening for this genotype"
        ]
    },
    {
        "id": "poor_recovery_hormonal_suppression",
        "name": "Chronic Recovery Deficit Driving Hormonal Suppression",
        "severity": "warning",
        "category": "Hormonal/Recovery",
        "required_snps": ["rs4680", "rs743572", "rs1799941"],
        "required_biomarkers": {},
        "required_wearable": {"recovery.avg_readiness_score": lambda v: v < 65},
        "optional_biomarkers": {
            "testosterone_total": lambda v: v < 600,
            "cortisol": lambda v: v > 16,
            "dhea_s": lambda v: v < 300,
        },
        "optional_wearable": {
            "hrv.hrv_trend": lambda v: v == "declining",
            "sleep.avg_deep_hours": lambda v: v < 1.5,
        },
        "description": "COMT, CYP17A1, and SHBG variants affect stress hormone metabolism and sex hormone bioavailability. Chronically low readiness scores indicate your body is not recovering between days — a pattern that elevates cortisol and suppresses testosterone over time. Slow COMT means stress hormones and catechol estrogens clear more slowly, prolonging the cortisol response.",
        "implications": "Chronic under-recovery creates a hormonal environment where cortisol remains elevated while anabolic hormones decline. This is self-reinforcing — poor recovery leads to worse sleep, which leads to even poorer recovery and further hormonal suppression.",
        "actions": [
            "Identify the primary recovery thief: sleep quality, overtraining, or chronic psychological stress — address the root cause first",
            "Deep sleep is where 70% of testosterone is produced — prioritize it with consistent bedtime, cool room, and magnesium glycinate",
            "Add one complete rest day per week — walk only, no structured exercise",
            "Ashwagandha KSM-66 (600mg/day) — clinically shown to reduce cortisol 25-30% and improve testosterone in chronically stressed individuals",
            "Track readiness score trend over 4 weeks after implementing recovery protocol — target consistent scores above 70",
            "Salivary cortisol testing (4-point: AM/noon/PM/bedtime) — characterizes your cortisol curve and is highly informative for COMT variants"
        ]
    },
    {
        "id": "wearable_metabolic_trifecta",
        "name": "Three-Way Metabolic Stress Convergence",
        "severity": "critical",
        "category": "Metabolic",
        "required_snps": ["rs7903146", "rs1260326"],
        "required_biomarkers": {"triglycerides": lambda v: v > 120},
        "required_wearable": {
            "sleep.avg_total_hours": lambda v: v < 7.5,
            "activity.avg_daily_steps": lambda v: v < 8000,
        },
        "optional_biomarkers": {
            "glucose": lambda v: v > 88,
            "hdl": lambda v: v < 55,
        },
        "optional_wearable": {"recovery.avg_readiness_score": lambda v: v < 70},
        "description": "TCF7L2 and GCKR genetic variants predispose to elevated triglycerides and impaired glucose regulation. Your blood work confirms elevated triglycerides, and your wearable data identifies the two most potent behavioral drivers: insufficient sleep and insufficient movement. All three data sources — genetics, blood work, and daily behavior — are converging on the same metabolic stress pattern.",
        "implications": "This is a three-way confirmation of metabolic stress. Genetics loaded the gun, insufficient sleep and inactivity pulled the trigger, and blood work shows the result. The wearable data tells us exactly where the behavioral leverage is — you do not need more tests, you need to fix sleep and movement.",
        "actions": [
            "Extend sleep to 7.5+ hours — sleep deprivation elevates triglycerides 20-30% through cortisol-driven hepatic lipogenesis",
            "Add 2,000+ steps per day to current average — even modest NEAT increases meaningfully reduce triglycerides",
            "Eliminate refined carbohydrates and liquid sugars — the most direct dietary driver for GCKR variants",
            "Post-meal walks 10-15 min after dinner — reduces triglyceride spikes by accelerating lipoprotein lipase clearance",
            "Target triglycerides below 100 mg/dL — stricter than the standard <150 given your genetic profile",
            "Retest lipid panel in 90 days after fixing sleep and activity — expect 20-40% triglyceride reduction from behavioral changes alone"
        ]
    },
]


# =============================================================================
# PATTERN MATCHING ENGINE
# =============================================================================

def check_biomarker_condition(biomarker_key, condition_fn, evaluated_biomarkers):
    """Check if a biomarker meets a condition. Returns True if met."""
    # Search by db_key
    for name, data in evaluated_biomarkers.items():
        if data.get("db_key") == biomarker_key and data.get("in_db"):
            try:
                return condition_fn(float(data["value"]))
            except (ValueError, TypeError):
                return False
    return False


def detect_hardcoded_patterns(snp_results: dict, blood_results: dict = None, wearable_metrics: dict = None) -> list:
    """
    Run the hardcoded pattern library against available data.
    Returns list of matched patterns with match details.
    """
    # Flatten SNP results to a set of risk rsids
    risk_rsids = set()
    for category_snps in snp_results.values():
        for snp in category_snps:
            if not snp.get("claim_ready", True):
                continue
            if snp["zygosity"] in ["heterozygous", "homozygous_risk"]:
                risk_rsids.add(snp["rsid"].lower())

    evaluated_biomarkers = blood_results["evaluated"] if blood_results else {}

    matched_patterns = []

    for pattern in PATTERN_LIBRARY:
        # Check required SNPs — at least one must match
        snp_match = any(rsid.lower() in risk_rsids for rsid in pattern["required_snps"])
        if not snp_match:
            continue

        # Check required biomarkers — ALL must match if blood_results provided
        required_bio = pattern.get("required_biomarkers", {})
        if required_bio and blood_results:
            bio_match = all(
                check_biomarker_condition(key, fn, evaluated_biomarkers)
                for key, fn in required_bio.items()
            )
            if not bio_match:
                continue
        elif required_bio and not blood_results:
            continue

        # Check required wearable conditions — ALL must match if wearable provided
        required_wear = pattern.get("required_wearable", {})
        if required_wear and wearable_metrics:
            wear_match = all(
                check_wearable_condition(key, fn, wearable_metrics)
                for key, fn in required_wear.items()
            )
            if not wear_match:
                continue
        elif required_wear and not wearable_metrics:
            # Pattern requires wearable data but none provided — skip
            continue

        # Count optional matches (biomarker + wearable)
        optional_bio = pattern.get("optional_biomarkers", {})
        optional_matches = 0
        if optional_bio and blood_results:
            optional_matches += sum(
                1 for key, fn in optional_bio.items()
                if check_biomarker_condition(key, fn, evaluated_biomarkers)
            )
        optional_wear = pattern.get("optional_wearable", {})
        if optional_wear and wearable_metrics:
            optional_matches += sum(
                1 for key, fn in optional_wear.items()
                if check_wearable_condition(key, fn, wearable_metrics)
            )

        triggered_snps = [rsid for rsid in pattern["required_snps"] if rsid.lower() in risk_rsids]

        # Determine confirmation strength
        has_bio = bool(required_bio and blood_results)
        has_wear = bool(required_wear and wearable_metrics)
        if has_bio and has_wear:
            strength = "confirmed by blood work + wearable"
        elif has_bio:
            strength = "confirmed by blood work"
        elif has_wear:
            strength = "confirmed by wearable data"
        else:
            strength = "genetic only"

        matched_patterns.append({
            "pattern": pattern,
            "triggered_snps": triggered_snps,
            "optional_matches": optional_matches,
            "total_optional": len(optional_bio) + len(optional_wear),
            "has_biomarker_confirmation": has_bio,
            "has_wearable_confirmation": has_wear,
            "strength": strength
        })

    severity_order = {"critical": 0, "warning": 1, "insight": 2}
    matched_patterns.sort(key=lambda x: (
        severity_order.get(x["pattern"]["severity"], 3),
        -x["optional_matches"]
    ))

    return matched_patterns


def detect_ai_patterns(snp_results: dict, blood_results: dict, hardcoded_patterns: list, client, wearable_metrics: dict = None) -> tuple:
    """
    AI-driven pattern scan — finds emergent patterns across all three data sources.
    Returns (ai_patterns_text, tokens_used)
    """
    already_found = [p["pattern"]["name"] for p in hardcoded_patterns]

    # Risk SNPs
    risk_snp_lines = []
    for category, snps in snp_results.items():
        for snp in snps:
            if snp["zygosity"] in ["heterozygous", "homozygous_risk"]:
                risk_snp_lines.append(
                    f"{snp['rsid']} ({snp['gene']}) | {snp['trait']} | "
                    f"{snp['zygosity']} | {snp['effect']}"
                )

    # Flagged biomarkers
    bio_lines = []
    if blood_results:
        for name, b in blood_results["evaluated"].items():
            if b.get("in_db") and b.get("flag") in ["abnormal", "suboptimal", "genomic_recontextualized"]:
                bio_lines.append(
                    f"{b['name']}: {b['value']} {b['unit']} "
                    f"[optimal: {b.get('optimal_range', 'N/A')}] — {b['flag'].upper()}"
                )

    # Wearable summary
    wearable_lines = []
    if wearable_metrics:
        s = wearable_metrics.get("sleep", {})
        h = wearable_metrics.get("hrv", {})
        hr = wearable_metrics.get("heart_rate", {})
        a = wearable_metrics.get("activity", {})
        r = wearable_metrics.get("recovery", {})

        if s.get("avg_total_hours"):
            wearable_lines.append(f"Sleep: {s['avg_total_hours']}h avg total, {s.get('avg_deep_hours','?')}h deep, {s.get('avg_rem_hours','?')}h REM, {s.get('avg_efficiency_pct','?')}% efficiency")
        if h.get("avg_rmssd"):
            wearable_lines.append(f"HRV: {h['avg_rmssd']} ms RMSSD avg, trend: {h.get('hrv_trend', 'unknown')}")
        if hr.get("avg_resting_hr"):
            wearable_lines.append(f"Resting HR: {hr['avg_resting_hr']} bpm avg, trend: {hr.get('resting_hr_trend', 'unknown')}")
        if a.get("avg_daily_steps"):
            wearable_lines.append(f"Activity: {a['avg_daily_steps']:,} steps/day avg, {a.get('sedentary_days_pct','?')}% sedentary days")
        readiness = r.get("avg_readiness_score") or r.get("avg_recovery_score")
        if readiness:
            wearable_lines.append(f"Recovery/Readiness: {readiness} avg score, {r.get('low_recovery_days_pct','?')}% low recovery days")
        for flag in wearable_metrics.get("flags", []):
            wearable_lines.append(f"FLAG: {flag['metric']} — {flag['message']}")

    system = (
        "You are a precision health pattern recognition specialist analyzing three simultaneous data sources: "
        "genetic variants, blood biomarkers, and physiological wearable data. "
        "Your job is to identify clinically meaningful patterns where the combination across sources "
        "tells a story that no single source alone would reveal. "
        "Focus on actionable intersections. Do not repeat patterns already identified."
    )

    already_str = "\n".join(f"- {p}" for p in already_found) if already_found else "None"
    snp_str = "\n".join(risk_snp_lines[:40])
    bio_str = "\n".join(bio_lines) if bio_lines else "No blood work provided."
    wear_str = "\n".join(wearable_lines) if wearable_lines else "No wearable data provided."

    user = (
        f"Identify 2-4 cross-source health patterns NOT already listed below.\n\n"
        f"ALREADY IDENTIFIED (do not repeat):\n{already_str}\n\n"
        f"GENETIC RISK VARIANTS:\n{snp_str}\n\n"
        f"FLAGGED BIOMARKERS:\n{bio_str}\n\n"
        f"WEARABLE PHYSIOLOGICAL DATA:\n{wear_str}\n\n"
        f"For each additional pattern found:\n"
        f"1. Name it clearly (5-8 words)\n"
        f"2. Explain what the multi-source combination means in plain language (2-3 sentences)\n"
        f"3. Give 2-3 specific, named, actionable recommendations\n\n"
        f"Only identify patterns where the combination across sources is genuinely more significant "
        f"than any single source alone. If no additional meaningful patterns exist, say so."
    )

    body = json.dumps({
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": 1500, "temperature": 0.4}
    })

    response = client.invoke_model(
        modelId="us.amazon.nova-pro-v1:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    text = result["output"]["message"]["content"][0]["text"]
    usage = result.get("usage", {})
    tokens = usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
    return text, tokens


def run_pattern_engine(snp_results: dict, blood_results: dict = None, client=None, wearable_metrics: dict = None) -> dict:
    """
    Main entry point. Runs both pattern detection layers across all available data sources.
    """
    sources = ["genomic"]
    if blood_results:
        sources.append("blood work")
    if wearable_metrics:
        sources.append("wearable")
    print(f"Running cross-source pattern engine ({', '.join(sources)})...")

    # Layer 1: Hardcoded patterns
    hardcoded = detect_hardcoded_patterns(snp_results, blood_results, wearable_metrics)
    print(f"  Layer 1: {len(hardcoded)} hardcoded patterns matched")
    for p in hardcoded:
        print(f"    [{p['pattern']['severity'].upper()}] {p['pattern']['name']} ({p['strength']})")

    # Layer 2: AI-driven scan
    ai_patterns_text = None
    ai_tokens = 0
    if client and snp_results:
        print("  Layer 2: Running AI pattern scan...")
        ai_patterns_text, ai_tokens = detect_ai_patterns(
            snp_results, blood_results or {}, hardcoded, client, wearable_metrics
        )

    return {
        "hardcoded_patterns": hardcoded,
        "ai_patterns_text": ai_patterns_text,
        "ai_tokens": ai_tokens,
        "total_patterns": len(hardcoded),
        "critical_count": sum(1 for p in hardcoded if p["pattern"]["severity"] == "critical"),
        "warning_count": sum(1 for p in hardcoded if p["pattern"]["severity"] == "warning"),
        "insight_count": sum(1 for p in hardcoded if p["pattern"]["severity"] == "insight"),
        "sources_used": sources,
    }
