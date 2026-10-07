"""
HealthBridge AI - Blood Work Parser & Biomarker Reference Database
Supports: Quest, Labcorp, Function Health, Boston Heart, any generic lab PDF
Evaluates against both standard reference ranges and longevity-optimal ranges
Cross-references with genomic data when available
"""

import json
import re
import boto3
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional local fallback
    PdfReader = None

# =============================================================================
# BIOMARKER REFERENCE DATABASE
# Each entry: standard_low, standard_high, optimal_low, optimal_high, unit,
#             category, clinical_name, genomic_snps (SNPs that recontextualize
#             interpretation), genomic_note (what to say when SNPs are present)
# =============================================================================

BIOMARKER_DB = {

    # =========================================================================
    # METABOLIC / GLUCOSE
    # =========================================================================
    "glucose": {
        "aliases": ["fasting glucose", "blood glucose", "serum glucose", "glucose, serum", "glucose, plasma"],
        "unit": "mg/dL", "category": "Metabolic",
        "standard_low": 70, "standard_high": 99,
        "optimal_low": 72, "optimal_high": 85,
        "clinical_name": "Fasting Glucose",
        "interpretation": {
            "optimal": "Fasting glucose is in the longevity-optimal range.",
            "normal": "Fasting glucose is within standard reference range but above longevity-optimal target.",
            "high": "Fasting glucose is elevated — pre-diabetic or diabetic range warrants clinical evaluation.",
            "low": "Fasting glucose is below standard range — hypoglycemia risk."
        },
        "genomic_snps": ["rs7903146", "rs12255372", "rs5219", "rs1260326", "rs10830963", "rs1801133"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TCF7L2, KCNJ11, or MTNR1B variants impair how your beta cells secrete insulin and how your cells respond to it. This means a fasting glucose of 90-99 mg/dL — which a doctor would call completely normal — carries the same long-term risk for you as a glucose of 105+ would for someone without these variants.",
            "implications": "Left unaddressed, this pattern progresses silently. HbA1c stays 'normal' while insulin resistance builds for years. By the time your glucose crosses the pre-diabetic threshold of 100, you will likely have had elevated insulin for 5-10 years already — meaning significant cardiovascular and metabolic damage has already occurred. Pre-diabetes doubles cardiovascular disease risk and increases dementia risk by 50%.",
            "actions": [
                "Target fasting glucose below 85 mg/dL as your personal goal — not the standard 99 cutoff",
                "Test fasting insulin alongside glucose every 6 months — insulin rises years before glucose does, and this is your early warning signal",
                "10-minute walk after every meal — shown to reduce post-meal glucose spikes by 20-30% and directly addresses the beta cell insufficiency these variants cause",
                "Eliminate liquid calories (juice, soda, sports drinks) — these bypass the insulin response buffering that your TCF7L2 variants already compromise",
                "Time-restricted eating in an 8-10 hour window — reduces total daily insulin demand by 25-30%, directly compensating for your reduced beta cell capacity"
            ]
        }
    },
    "hba1c": {
        "aliases": ["hemoglobin a1c", "hba1c", "glycated hemoglobin", "glycohemoglobin", "a1c"],
        "unit": "%", "category": "Metabolic",
        "standard_low": 0, "standard_high": 5.6,
        "optimal_low": 4.6, "optimal_high": 5.3,
        "clinical_name": "HbA1c",
        "interpretation": {
            "optimal": "HbA1c reflects excellent 3-month glucose control.",
            "normal": "HbA1c is within normal range but above longevity-optimal target.",
            "high": "HbA1c is elevated — pre-diabetic (5.7-6.4%) or diabetic (≥6.5%) range.",
            "low": "HbA1c is below standard range."
        },
        "genomic_snps": ["rs7903146", "rs5219", "rs10830963"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TCF7L2 and KCNJ11 variants mean your beta cells are working harder than average to keep glucose controlled. An HbA1c of 5.4-5.6% — technically 'normal' — reflects compensated insulin resistance in someone with your genetic profile, not true metabolic health.",
            "implications": "HbA1c above 5.3% in someone with these variants predicts progression to pre-diabetes within 3-5 years without intervention. Each 0.1% increase in HbA1c above 5.0% is associated with a 6% increase in cardiovascular event risk. At 5.6%, you are 60% of the way to the pre-diabetic threshold — but the metabolic damage accumulates linearly, not just when you cross the cutoff.",
            "actions": [
                "Your personal HbA1c target is below 5.3%, not the standard 5.6% — this is achievable with dietary carbohydrate reduction",
                "Reduce refined carbohydrate intake to below 100g/day — the single most effective dietary intervention for lowering HbA1c",
                "Resistance training 3x/week — increases GLUT4 transporters in muscle, improving glucose uptake independently of insulin and directly compensating for your genetic beta cell limitation",
                "Retest HbA1c in 90 days after implementing dietary changes — expect 0.2-0.4% reduction with consistent effort"
            ]
        }
    },
    "insulin": {
        "aliases": ["fasting insulin", "insulin, fasting", "insulin fasting", "serum insulin"],
        "unit": "uIU/mL", "category": "Metabolic",
        "standard_low": 2, "standard_high": 25,
        "optimal_low": 2, "optimal_high": 6,
        "clinical_name": "Fasting Insulin",
        "interpretation": {
            "optimal": "Fasting insulin is in the longevity-optimal range — excellent insulin sensitivity.",
            "normal": "Fasting insulin is within standard range but elevated relative to optimal, suggesting early insulin resistance.",
            "high": "Fasting insulin is elevated — significant insulin resistance likely.",
            "low": "Fasting insulin is below standard range."
        },
        "genomic_snps": ["rs7903146", "rs1801282", "rs1044498", "rs5219"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your PPARG, TCF7L2, or ENPP1 variants reduce insulin sensitivity at the cellular level. When your body compensates by producing more insulin to overcome this resistance, fasting insulin rises while glucose stays 'normal' — making glucose and HbA1c misleadingly reassuring. Fasting insulin above 7 uIU/mL with these variants indicates meaningful insulin resistance regardless of glucose.",
            "implications": "Chronic compensatory hyperinsulinemia is the root driver of metabolic syndrome, PCOS, visceral fat accumulation, accelerated arterial aging, and elevated cancer risk (insulin is a growth factor). The average person with T2D had elevated fasting insulin for 10-15 years before their glucose crossed the diagnostic threshold. You are looking at the early stage of that trajectory.",
            "actions": [
                "Your target is fasting insulin below 6 uIU/mL — retest in 90 days after dietary changes",
                "Intermittent fasting (16:8) is the most powerful single intervention for reducing fasting insulin — each hour of fasting depletes glycogen and forces insulin down",
                "Eliminate snacking between meals — every eating occasion spikes insulin; your variants mean these spikes are higher and last longer than average",
                "Berberine 500mg twice daily with meals — clinically shown to lower fasting insulin and improve insulin sensitivity through AMPK activation, comparable to metformin in some studies",
                "Visceral fat reduction is the structural fix — even 5-7% body weight loss reduces fasting insulin by 30-40% in insulin-resistant individuals"
            ]
        }
    },
    "homa_ir": {
        "aliases": ["homa-ir", "homa ir", "insulin resistance index"],
        "unit": "index", "category": "Metabolic",
        "standard_low": 0, "standard_high": 2.9,
        "optimal_low": 0, "optimal_high": 1.0,
        "clinical_name": "HOMA-IR (Insulin Resistance Index)",
        "interpretation": {
            "optimal": "HOMA-IR indicates excellent insulin sensitivity.",
            "normal": "HOMA-IR is within normal range but suggests developing insulin resistance.",
            "high": "HOMA-IR is elevated — significant insulin resistance.",
            "low": "HOMA-IR is at the lower end — excellent insulin sensitivity."
        },
        "genomic_snps": ["rs7903146", "rs1801282", "rs1799883"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TCF7L2, PPARG, or related insulin metabolism variants impair both insulin secretion and cellular insulin signaling. HOMA-IR is calculated from fasting glucose and fasting insulin, which means your genetic tendency to have elevated fasting insulin (even with 'normal' glucose) inflates your HOMA-IR. A HOMA-IR between 1.0 and 2.9 is considered 'normal' by population standards — but for your genotype, any value above 1.0 reflects a meaningful degree of insulin resistance.",
            "implications": "HOMA-IR above 1.0 with your variant profile indicates that your pancreas is already compensating — producing more insulin than average to maintain glucose in the normal range. This compensatory hyperinsulinemia is a precursor to metabolic syndrome, visceral fat accumulation, type 2 diabetes, and cardiovascular disease. The window between HOMA-IR of 1.0 and 2.9 is not 'safe' for your genotype — it is the period during which meaningful intervention is most effective.",
            "actions": [
                "Your personal HOMA-IR target is below 1.0 — treat anything above that threshold as a signal to act, not monitor",
                "Time-restricted eating (16:8 or 14:10 window) is the most effective lifestyle intervention for reducing HOMA-IR — it directly lowers fasting insulin by extending the overnight low-insulin period",
                "Resistance training 3x/week increases GLUT4 transporter density in muscle, improving insulin-independent glucose uptake and directly lowering insulin demand",
                "Myo-inositol (2g) combined with D-chiro-inositol (50mg) daily — acts as a secondary insulin messenger and is clinically validated to reduce HOMA-IR in insulin-resistant individuals",
                "Retest fasting insulin and HOMA-IR every 90 days while implementing these changes — the trend over time is more important than any single measurement"
            ]
        }
    },

    # =========================================================================
    # LIPID PANEL
    # =========================================================================
    "total_cholesterol": {
        "aliases": ["total cholesterol", "cholesterol, total", "cholesterol"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 0, "standard_high": 199,
        "optimal_low": 150, "optimal_high": 180,
        "clinical_name": "Total Cholesterol",
        "interpretation": {
            "optimal": "Total cholesterol is in the longevity-optimal range.",
            "normal": "Total cholesterol is within borderline-acceptable range.",
            "high": "Total cholesterol is elevated — cardiovascular risk assessment warranted.",
            "low": "Total cholesterol is below standard range — may affect hormone synthesis and neurological function."
        },
        "genomic_snps": ["rs429358", "rs7412", "rs11206510", "rs2479409", "rs688", "rs2228671"],
        "genomic_note": {
            "direction": "worse",
            "context": "For APOE4 carriers, total cholesterol is a particularly misleading metric. What matters is not the total amount of cholesterol but the number of atherogenic particles (LDL particles) and their time in circulation — both of which are worsened by APOE4. Your total cholesterol near the standard 'borderline' threshold carries meaningfully higher risk for your genotype.",
            "implications": "APOE4 carriers with 'borderline' total cholesterol (190-210 mg/dL) have cardiovascular event rates comparable to non-carriers with 'high' cholesterol (>240 mg/dL). The standard risk stratification tools were built on population averages that do not adequately weight APOE4 status.",
            "actions": [
                "Shift focus from total cholesterol to ApoB and LDL particle number — these are the metrics that actually predict risk for your genotype",
                "Request a comprehensive cardiovascular risk panel including ApoB, Lp(a), and LDL-P rather than relying on standard lipid panel alone",
                "Saturated fat restriction is the most impactful dietary change for your genotype — replace with olive oil and fatty fish"
            ]
        }
    },
    "ldl": {
        "aliases": ["ldl cholesterol", "ldl-c", "ldl", "low density lipoprotein", "ldl chol"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 0, "standard_high": 99,
        "optimal_low": 40, "optimal_high": 70,
        "clinical_name": "LDL Cholesterol",
        "interpretation": {
            "optimal": "LDL is in the longevity-optimal range.",
            "normal": "LDL is within borderline range — optimization warranted given cardiovascular risk context.",
            "high": "LDL is elevated — cardiovascular risk assessment and intervention warranted.",
            "low": "LDL is very low — monitor for hormonal and neurological implications."
        },
        "genomic_snps": ["rs429358", "rs7412", "rs11206510", "rs688", "rs2228671", "rs4149056"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your APOE4 allele changes how your liver processes LDL — APOE4 carriers clear LDL from the bloodstream more slowly, meaning LDL particles spend more time in circulation and have more opportunity to penetrate arterial walls and oxidize. At the same LDL level, APOE4 carriers accumulate arterial plaque approximately 30-40% faster than non-carriers. Additionally, APOE4 makes you significantly more sensitive to dietary saturated fat — the same intake that minimally affects most people can raise your LDL by 15-25 mg/dL.",
            "implications": "APOE4 is the single strongest common genetic risk factor for cardiovascular disease and late-onset Alzheimer's disease. One copy increases Alzheimer's risk 3-4x; two copies increases it 8-12x. The cardiovascular and neurological risks share the same root — impaired lipid clearance and increased oxidative stress. Your LDL at current levels is already creating risk above what standard guidelines suggest. Waiting until your LDL crosses the 'high' threshold means years of additional plaque accumulation that could have been prevented.",
            "actions": [
                "Your personal LDL target is below 70 mg/dL — not the standard 'optimal' of 100 mg/dL; discuss with your physician",
                "Request ApoB testing — for APOE4 carriers, ApoB is more predictive than LDL-C; target ApoB below 70 mg/dL",
                "Dramatically reduce saturated fat (red meat, full-fat dairy, coconut oil) — APOE4 carriers show 2-3x larger LDL increases from saturated fat than APOE3/3 individuals",
                "Replace saturated fat with monounsaturated fats (olive oil, avocado, almonds) — shown to reduce LDL specifically in APOE4 carriers",
                "Request a coronary artery calcium (CAC) score — directly measures arterial plaque burden; highly recommended as a baseline for APOE4 carriers",
                "If SLCO1B1 variant is present, inform your prescribing physician before starting statins — this variant increases statin-induced muscle damage risk"
            ]
        }
    },
    "hdl": {
        "aliases": ["hdl cholesterol", "hdl-c", "hdl", "high density lipoprotein", "hdl chol"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 40, "standard_high": 999,
        "optimal_low": 60, "optimal_high": 90,
        "clinical_name": "HDL Cholesterol",
        "interpretation": {
            "optimal": "HDL is in the longevity-optimal range — excellent cardiovascular protection.",
            "normal": "HDL is at acceptable levels but below longevity-optimal target.",
            "high": "HDL is very high — generally favorable but extremely high HDL (>100) may paradoxically increase risk.",
            "low": "HDL is low — significant cardiovascular risk factor."
        },
        "genomic_snps": ["rs1800775", "rs708272", "rs1800588", "rs328"],
        "genomic_note": {
            "direction": "neutral",
            "context": "Your CETP or LIPC variants affect how HDL particles are formed and cleared. Some individuals with these variants have genetically high HDL numbers that look great on paper but are functionally impaired — the particles are larger and less efficient at pulling cholesterol out of arterial walls (reduced cholesterol efflux capacity). Your HDL number alone does not tell the full story.",
            "implications": "Paradoxically high HDL from CETP variants (above 90-100 mg/dL) has been associated with increased rather than decreased cardiovascular risk in some studies, because the particles are dysfunctional. If your HDL is in this range, it is worth understanding whether it reflects genuine protective function or genetically altered CETP activity.",
            "actions": [
                "Focus on HDL function markers rather than just HDL-C number — ask about cholesterol efflux capacity testing if available through your provider",
                "Aerobic exercise is the most reliable way to improve functional HDL (not just HDL-C) — aim for 150+ minutes zone 2 weekly",
                "Omega-3 supplementation improves HDL particle quality and function independently of HDL-C levels"
            ]
        }
    },
    "triglycerides": {
        "aliases": ["triglycerides", "trigs", "trig", "triglyceride"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 0, "standard_high": 149,
        "optimal_low": 40, "optimal_high": 80,
        "clinical_name": "Triglycerides",
        "interpretation": {
            "optimal": "Triglycerides are in the longevity-optimal range — excellent metabolic health indicator.",
            "normal": "Triglycerides are within normal range but above optimal — refined carbohydrate and alcohol intake should be reviewed.",
            "high": "Triglycerides are elevated — metabolic risk, insulin resistance, and cardiovascular risk.",
            "low": "Triglycerides are very low — generally favorable."
        },
        "genomic_snps": ["rs1260326", "rs780094", "rs328", "rs268", "rs174537"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your GCKR variant causes your liver to produce more triglycerides from carbohydrates than average — a process called de novo lipogenesis. This means your triglycerides are partially hardwired by genetics, and the standard dietary advice to 'reduce fat' will not address your primary driver. Your LPL variant additionally reduces the rate at which triglycerides are cleared from your bloodstream after meals.",
            "implications": "Elevated triglycerides with your variant profile indicate your liver is converting carbohydrates to fat at an accelerated rate, and your body clears those fats from your blood more slowly. This increases VLDL production, reduces HDL, and drives small dense LDL particle formation — the most atherogenic LDL subtype. Triglycerides above 150 double your risk of metabolic syndrome and are independently associated with cardiovascular events.",
            "actions": [
                "Your primary target is carbohydrate reduction, not fat reduction — specifically refined carbohydrates and sugars, which are the substrate for your GCKR-driven de novo lipogenesis",
                "Omega-3 supplementation (EPA+DHA 3-4g/day) directly upregulates LPL activity, compensating for your LPL variant and improving triglyceride clearance",
                "Aerobic exercise 4-5x/week activates LPL in muscle tissue, your most effective tool for improving triglyceride clearance given your genetic profile",
                "Your personal target is triglycerides below 100 mg/dL — not the standard <150 threshold",
                "Alcohol restriction is especially important for you — alcohol directly inhibits hepatic LPL and compounds your GCKR-driven triglyceride overproduction"
            ]
        }
    },
    "vldl": {
        "aliases": ["vldl", "vldl cholesterol", "very low density lipoprotein"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 0, "standard_high": 29,
        "optimal_low": 0, "optimal_high": 18,
        "clinical_name": "VLDL Cholesterol",
        "interpretation": {
            "optimal": "VLDL is in the optimal range.",
            "normal": "VLDL is within standard range but above optimal.",
            "high": "VLDL is elevated — triglyceride-rich lipoprotein burden is increased.",
            "low": "VLDL is very low — favorable."
        },
        "genomic_snps": ["rs1260326", "rs174537"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your GCKR variant causes your liver to convert dietary carbohydrates into VLDL triglyceride particles at an accelerated rate. FADS variants additionally alter how your liver processes polyunsaturated fatty acids, further affecting VLDL composition. VLDL is the direct precursor to LDL — elevated VLDL production means your liver is continuously flooding your bloodstream with triglyceride-rich particles that mature into atherogenic LDL.",
            "implications": "Chronically elevated VLDL with your GCKR variant signals that your liver is in a state of ongoing de novo lipogenesis, converting carbohydrates to fat at a rate that exceeds your ability to clear them. This drives small dense LDL particle formation (the most atherogenic LDL subtype), suppresses HDL, and contributes to hepatic fat accumulation. VLDL above 18 mg/dL combined with GCKR variants places you at meaningfully elevated cardiometabolic risk.",
            "actions": [
                "Dietary carbohydrate reduction is the primary lever — specifically reducing refined starches and sugars, which are the substrate for your GCKR-driven VLDL overproduction",
                "Omega-3 supplementation (EPA+DHA 3-4g/day) directly reduces hepatic VLDL secretion and activates LPL to clear existing VLDL particles faster",
                "Eliminate alcohol and fructose — both directly stimulate hepatic de novo lipogenesis and compound GCKR-driven VLDL overproduction",
                "Aerobic exercise 150+ minutes weekly activates LPL in peripheral tissues, the primary enzyme responsible for clearing VLDL from your bloodstream",
                "Track VLDL longitudinally alongside triglycerides — they move together, and your personal target is VLDL below 18 mg/dL"
            ]
        }
    },
    "apob": {
        "aliases": ["apolipoprotein b", "apob", "apo b", "apo-b"],
        "unit": "mg/dL", "category": "Lipids",
        "standard_low": 0, "standard_high": 109,
        "optimal_low": 40, "optimal_high": 80,
        "clinical_name": "Apolipoprotein B (ApoB)",
        "interpretation": {
            "optimal": "ApoB is in the longevity-optimal range — excellent atherogenic particle burden.",
            "normal": "ApoB is within standard range but above longevity-optimal target.",
            "high": "ApoB is elevated — direct measure of atherogenic particle burden, more predictive than LDL-C.",
            "low": "ApoB is very low — favorable."
        },
        "genomic_snps": ["rs429358", "rs7412", "rs11206510", "rs688"],
        "genomic_note": {
            "direction": "worse",
            "context": "ApoB is the single protein present on every atherogenic particle (LDL, VLDL, IDL, Lp(a)) — one molecule per particle. For APOE4 carriers and those with PCSK9 or LDLR variants, ApoB is the most accurate measure of cardiovascular risk on a standard blood panel. Your variants reduce the rate at which your liver clears LDL particles from circulation, meaning LDL spends more time in your bloodstream — time during which each particle can penetrate arterial walls, oxidize, and trigger plaque formation.",
            "implications": "ApoB above 80 mg/dL in the context of APOE4 or LDLR/PCSK9 variants carries cardiovascular risk that is significantly underestimated by LDL-C alone. The discordance pattern — where LDL-C appears acceptable but ApoB is elevated — is common in your genetic profile and is systematically missed by standard lipid panels. Every 10 mg/dL increase in ApoB above 70 is associated with approximately 10-15% increased cardiovascular event risk. With impaired particle clearance, particles spend more time in circulation than the number alone suggests.",
            "actions": [
                "Your ApoB target is below 70 mg/dL — not the standard reference range cutoff of 109 mg/dL",
                "Request ApoB testing at every cardiovascular checkup — this should replace or supplement LDL-C as your primary lipid tracking metric given your genotype",
                "Saturated fat restriction is your highest-yield dietary intervention — APOE4 carriers show 2-3x larger ApoB increases from saturated fat than non-carriers",
                "If ApoB remains above 80 mg/dL despite lifestyle optimization, discuss statin therapy with your physician — APOE4 carriers show exceptional risk reduction from statin therapy due to their baseline impaired LDL clearance",
                "Also monitor Lp(a) — it contributes to ApoB particle count and is largely lifestyle-resistant, requiring separate risk stratification"
            ]
        }
    },
    "lpa": {
        "aliases": ["lipoprotein a", "lp(a)", "lpa", "lipoprotein(a)"],
        "unit": "nmol/L", "category": "Lipids",
        "standard_low": 0, "standard_high": 75,
        "optimal_low": 0, "optimal_high": 30,
        "clinical_name": "Lipoprotein(a) [Lp(a)]",
        "interpretation": {
            "optimal": "Lp(a) is in the optimal range — low atherothrombotic risk from this source.",
            "normal": "Lp(a) is borderline — warrants monitoring especially with other cardiovascular risks.",
            "high": "Lp(a) is elevated — significant independent cardiovascular and stroke risk factor.",
            "low": "Lp(a) is very low — favorable."
        },
        "genomic_snps": ["rs3798220", "rs10455872"],
        "genomic_note": {
            "direction": "worse",
            "context": "LPA gene variants set your Lp(a) level at birth, and it changes little over your lifetime regardless of diet or exercise. Lp(a) is a modified LDL particle with an additional apo(a) protein that makes it particularly dangerous — it promotes both atherosclerosis (like LDL) and thrombosis (like a clotting factor) simultaneously. With LPA risk variants, your elevated Lp(a) is a fixed genetic risk exposure that has been present since birth and requires active cardiovascular risk management.",
            "implications": "Lp(a) above 30 nmol/L doubles cardiovascular event risk independent of LDL, blood pressure, and all other risk factors. Above 75 nmol/L, it triples risk and is associated with premature heart attack and aortic stenosis. Unlike LDL, Lp(a) is largely unresponsive to statins (which can paradoxically slightly raise Lp(a)), lifestyle modification, or dietary changes. This is not a problem you can solve with better habits — it is a fixed risk multiplier that must be factored into your overall cardiovascular strategy.",
            "actions": [
                "Ensure your physician is aware of your Lp(a) level and factors it into your overall cardiovascular risk assessment — it changes the threshold at which other risk factors require treatment",
                "If Lp(a) is above 75 nmol/L, consider a coronary artery calcium (CAC) score to assess existing arterial burden — this guides how aggressively to treat your other modifiable risk factors",
                "Niacin (extended release, 1-2g/day) is the most established supplement for reducing Lp(a) by 15-30% — discuss with your physician as niacin has its own side effect profile",
                "Emerging therapies: PCSK9 inhibitors (evolocumab, alirocumab) reduce Lp(a) by 20-30%; RNA-targeted therapies (pelacarsen) in late-stage trials show 80%+ reduction — ask your cardiologist about eligibility",
                "Compensate aggressively on all other modifiable risk factors — since Lp(a) cannot be easily lowered, your LDL, ApoB, hs-CRP, and blood pressure targets should be more aggressive than standard guidelines suggest"
            ]
        }
    },

    # =========================================================================
    # INFLAMMATION
    # =========================================================================
    "hscrp": {
        "aliases": ["hs-crp", "hscrp", "high sensitivity crp", "c-reactive protein", "crp", "c reactive protein hs"],
        "unit": "mg/L", "category": "Inflammation",
        "standard_low": 0, "standard_high": 3.0,
        "optimal_low": 0, "optimal_high": 0.5,
        "clinical_name": "High-Sensitivity CRP (hs-CRP)",
        "interpretation": {
            "optimal": "hs-CRP is in the optimal range — low systemic inflammation.",
            "normal": "hs-CRP is within normal range but elevated relative to longevity-optimal — subclinical inflammation present.",
            "high": "hs-CRP is elevated — significant systemic inflammation, elevated cardiovascular and all-cause mortality risk.",
            "low": "hs-CRP is at optimal low levels."
        },
        "genomic_snps": ["rs1800629", "rs1800795", "rs1800871", "rs1800896", "rs1143634"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TNF-alpha, IL-6, and/or IL-10 variants mean your immune system operates at a higher baseline inflammatory setpoint than average. Even with a perfect diet and lifestyle, you will tend toward higher hs-CRP than someone without these variants. The flip side: your inflammatory response to poor sleep, gut dysbiosis, or omega-3 deficiency is also amplified — lifestyle factors hit you harder than they hit someone with anti-inflammatory genetics.",
            "implications": "Chronically elevated hs-CRP above 1.0 mg/L is one of the strongest independent predictors of cardiovascular events — the JUPITER trial showed hs-CRP is as predictive as LDL. It also predicts insulin resistance, depression, cognitive decline, and accelerated biological aging. For you specifically, the genetic inflammatory load means standard dietary interventions alone are unlikely to get your hs-CRP below 1.0 mg/L — you need a multi-pronged approach targeting your specific inflammatory pathways.",
            "actions": [
                "Omega-3 supplementation (EPA+DHA 3-4g/day) directly inhibits the NF-kB pathway that your TNF-alpha variant upregulates — this is not generic anti-inflammatory advice, it specifically targets your genetic mechanism",
                "Sleep below 7 hours raises your hs-CRP by more than it would for someone with anti-inflammatory genetics — prioritize sleep as an anti-inflammatory intervention",
                "Gut health is your highest-leverage target: intestinal permeability drives systemic TNF-alpha elevation, which is exactly the variant you carry — L-glutamine, butyrate, and fermented foods specifically support the gut barrier that controls your inflammatory load",
                "Curcumin (as theracurmin or phospholipid complex for absorption) 500-1000mg/day — shown to reduce TNF-alpha and IL-6 specifically, targeting your variant pathways",
                "Your hs-CRP target is below 0.5 mg/L — not the standard 'low risk' cutoff of 1.0 mg/L"
            ]
        }
    },
    "homocysteine": {
        "aliases": ["homocysteine", "homocystiene", "hcy"],
        "unit": "umol/L", "category": "Inflammation",
        "standard_low": 0, "standard_high": 15,
        "optimal_low": 4, "optimal_high": 8,
        "clinical_name": "Homocysteine",
        "interpretation": {
            "optimal": "Homocysteine is in the longevity-optimal range — excellent methylation status.",
            "normal": "Homocysteine is within normal range but above optimal — methylation support indicated.",
            "high": "Homocysteine is elevated — cardiovascular, neurological, and endothelial risk. Methylation deficiency likely.",
            "low": "Homocysteine is very low — generally favorable."
        },
        "genomic_snps": ["rs1801133", "rs1801131", "rs1801394", "rs602662"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your MTHFR C677T variant reduces your enzyme's ability to convert folate into its active methylated form by 40-70% (or more if homozygous). This means the folate and B12 in your diet and standard supplements are not being converted into the methyl donors your body needs to clear homocysteine. Your homocysteine level reflects this bottleneck directly — and at your current level, your methylation system is under active stress.",
            "implications": "Homocysteine above 8 umol/L damages arterial endothelium — the lining of your blood vessels — independently of cholesterol. It accelerates atherosclerosis, increases blood clot risk, and directly damages neurons in the hippocampus (the brain's memory center). Long-term elevated homocysteine is one of the strongest modifiable predictors of cardiovascular disease and dementia. The crucial point: standard folic acid supplementation will not fix this. Folic acid requires the MTHFR enzyme to activate it — the same enzyme your variant impairs. Standard supplements can make this worse by competing with the small amount of functional methylfolate your body does produce.",
            "actions": [
                "Switch to methylfolate (L-5-MTHF) 400-800 mcg/day immediately — this is the activated form that bypasses your MTHFR bottleneck entirely",
                "Add methylcobalamin B12 1,000 mcg/day — B12 is the cofactor for homocysteine remethylation; your MTRR variant means you need the pre-activated form",
                "Add P5P (pyridoxal-5-phosphate, active B6) 25-50 mg/day — the alternative remethylation pathway requires B6 as cofactor",
                "Stop any supplements containing folic acid (most standard B-complex and prenatal vitamins) — folic acid competes with methylfolate for the same enzyme and can paradoxically worsen methylation in MTHFR variants",
                "Retest homocysteine in 8 weeks — it should drop by 30-50% with this protocol; target below 8 umol/L",
                "Minimize alcohol — directly depletes folate stores and acutely raises homocysteine"
            ]
        }
    },
    "ferritin": {
        "aliases": ["ferritin", "serum ferritin"],
        "unit": "ng/mL", "category": "Iron",
        "standard_low": 12, "standard_high": 300,
        "optimal_low": 50, "optimal_high": 150,
        "clinical_name": "Ferritin",
        "interpretation": {
            "optimal": "Ferritin is in the longevity-optimal range.",
            "normal": "Ferritin is within normal range. Note: ferritin is also an acute phase reactant — elevated ferritin with elevated hs-CRP suggests inflammation rather than iron overload.",
            "high": "Ferritin is elevated — iron overload risk or significant inflammation (acute phase reactant). Hemochromatosis should be evaluated.",
            "low": "Ferritin is low — iron deficiency. Evaluate for blood loss, malabsorption, or insufficient dietary iron."
        },
        "genomic_snps": ["rs1799945", "rs1800562"],
        "genomic_note": {
            "direction": "worse",
            "context": "HFE gene variants reduce your liver's ability to regulate iron absorption — a process normally tightly controlled. With hemochromatosis variants, your gut absorbs iron even when your body's stores are full, leading to progressive iron accumulation in organs including the liver, heart, pancreas, and joints.",
            "implications": "Untreated hereditary hemochromatosis is one of the most common preventable genetic diseases — iron overload causes liver cirrhosis, liver cancer (hepatocellular carcinoma), diabetes, cardiomyopathy, and arthritis over decades. The insidious part: symptoms are non-specific (fatigue, joint pain, abdominal discomfort) and easily attributed to other causes until organ damage is significant. The good news: it is completely manageable when caught early through regular phlebotomy.",
            "actions": [
                "Request HFE gene testing (C282Y and H63D mutations) through your physician — this confirms whether you carry hemochromatosis variants",
                "Monitor ferritin every 6 months and keep it in the 50-100 ng/mL range — ferritin is your primary iron overload tracking biomarker",
                "Avoid iron-fortified foods and iron supplementation unless specifically directed by a physician",
                "Tea and coffee with meals reduce iron absorption by 50-60% — a practical tool for managing iron load without medication",
                "Donate blood regularly if confirmed hemochromatosis carrier — therapeutic phlebotomy is the primary treatment and blood donation achieves the same effect while helping others"
            ]
        }
    },
    "il6": {
        "aliases": ["interleukin 6", "il-6", "il6"],
        "unit": "pg/mL", "category": "Inflammation",
        "standard_low": 0, "standard_high": 7,
        "optimal_low": 0, "optimal_high": 2,
        "clinical_name": "Interleukin-6 (IL-6)",
        "interpretation": {
            "optimal": "IL-6 is in the optimal range — low pro-inflammatory signaling.",
            "normal": "IL-6 is within normal range but above optimal.",
            "high": "IL-6 is elevated — significant pro-inflammatory state.",
            "low": "IL-6 is at low optimal levels."
        },
        "genomic_snps": ["rs1800795", "rs1800629"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your IL-6 promoter variant (rs1800795) increases transcription of the IL-6 gene, meaning your immune cells produce more IL-6 in response to the same inflammatory triggers compared to individuals without this variant. Combined with TNF-alpha variants that operate upstream, your body runs a persistently elevated inflammatory background signal that is partially genetic in origin and partially lifestyle-modifiable.",
            "implications": "Chronically elevated IL-6 drives multiple pathologies simultaneously: it stimulates CRP production in the liver (explaining elevated hs-CRP), induces insulin resistance in muscle and fat cells, promotes cortisol production via HPA axis stimulation, accelerates muscle wasting, and crosses the blood-brain barrier to drive neuroinflammation and depression risk. IL-6 is also the primary driver of the acute phase response — elevated baseline IL-6 means you are running a low-grade acute phase response chronically, accelerating biological aging.",
            "actions": [
                "Visceral fat reduction is the single most impactful intervention — adipose tissue is a major IL-6 source; each kilogram of visceral fat loss reduces circulating IL-6 meaningfully",
                "Omega-3 EPA specifically (not just DHA) directly suppresses IL-6 production at the cellular level — 3-4g EPA+DHA daily targets your variant mechanism",
                "Optimize sleep to 7.5-9 hours — sleep deprivation acutely elevates IL-6 by 30-40% in the following day; your variant amplifies this response",
                "Curcumin (phospholipid complex or theracurmin form for absorption) 500-1000mg/day inhibits NF-kB and directly reduces IL-6 gene transcription",
                "Retest IL-6 and hs-CRP simultaneously — they should track together; if hs-CRP is high but IL-6 is low, explore non-IL-6 inflammatory pathways"
            ]
        }
    },

    # =========================================================================
    # VITAMINS & MICRONUTRIENTS
    # =========================================================================
    "vitamin_d": {
        "aliases": ["vitamin d", "25-oh vitamin d", "25 oh vitamin d", "25-hydroxyvitamin d", "vitamin d, 25-hydroxy", "25(oh)d", "25-ohd"],
        "unit": "ng/mL", "category": "Vitamins",
        "standard_low": 30, "standard_high": 100,
        "optimal_low": 50, "optimal_high": 80,
        "clinical_name": "Vitamin D (25-OH)",
        "interpretation": {
            "optimal": "Vitamin D is in the longevity-optimal range.",
            "normal": "Vitamin D is within standard range but below longevity-optimal target.",
            "high": "Vitamin D is above the upper optimal range — monitor for toxicity above 100 ng/mL.",
            "low": "Vitamin D is deficient — supplementation and sun exposure warranted."
        },
        "genomic_snps": ["rs10741657", "rs2282679", "rs12785878", "rs2228570", "rs731236", "rs1544410"],
        "genomic_note": {
            "direction": "worse",
            "context": "You carry variants affecting multiple steps of vitamin D metabolism simultaneously — synthesis in the skin (DHCR7), transport in blood (GC binding protein), conversion to active form (CYP2R1), and cellular receptor sensitivity (VDR). Your serum level appears 'sufficient' by standard lab ranges, but these variants mean your cells are receiving significantly less active vitamin D signal than someone with the same serum level and no variants.",
            "implications": "Functional vitamin D insufficiency in someone with your variant stack is associated with significantly elevated risk of immune dysfunction, bone density loss starting in your 30s, reduced muscle strength and recovery, higher cardiovascular inflammatory markers, increased depression risk, and impaired insulin sensitivity. The reason this matters specifically for you: your standard blood level is a poor proxy for what is actually happening at the cellular level. Your VDR variant means even the vitamin D that does reach your cells is less effective at binding and signaling.",
            "actions": [
                "Target serum 25-OH vitamin D of 60-80 ng/mL — not the standard 'sufficient' threshold of 30 ng/mL",
                "Start vitamin D3 at 5,000-8,000 IU/day with vitamin K2 (MK-7, 200 mcg) — K2 is essential to direct calcium to bones rather than arteries when taking higher D3 doses",
                "Take vitamin D with your fattiest meal of the day — absorption increases 40-50% with dietary fat",
                "Retest 25-OH vitamin D after 90 days to calibrate your dose — your genetic variants mean your dose-response curve is different from average",
                "Consider testing 1,25-OH vitamin D (the active form) — low active vitamin D with normal 25-OH confirms conversion impairment from your CYP2R1 variant"
            ]
        }
    },
    "vitamin_b12": {
        "aliases": ["vitamin b12", "b12", "cobalamin", "cyanocobalamin", "vitamin b-12"],
        "unit": "pg/mL", "category": "Vitamins",
        "standard_low": 200, "standard_high": 900,
        "optimal_low": 500, "optimal_high": 900,
        "clinical_name": "Vitamin B12",
        "interpretation": {
            "optimal": "Vitamin B12 is in the longevity-optimal range.",
            "normal": "Vitamin B12 is within standard range but below optimal — functional deficiency may exist despite 'normal' levels.",
            "high": "Vitamin B12 is above standard range — if supplementing, reduce dose.",
            "low": "Vitamin B12 is deficient — neurological, hematological, and cardiovascular risk."
        },
        "genomic_snps": ["rs602662", "rs492602", "rs601338", "rs1801394"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your FUT2 non-secretor variant dramatically reduces how much B12 you absorb from food — up to 50% less than secretors. Your MTRR variant then impairs how efficiently your cells use the B12 that does get absorbed. At your current B12 level, your lab report says 'normal,' but your cells may be functionally deficient because the form most labs measure (total serum B12) does not reflect intracellular utilization.",
            "implications": "Functional B12 deficiency with your variant profile causes neurological damage that accumulates silently for years — subacute combined degeneration of the spinal cord, peripheral neuropathy, and cognitive decline are all driven by B12 deficiency. It also elevates homocysteine, which independently damages blood vessel walls and accelerates cardiovascular aging. Early symptoms (fatigue, brain fog, tingling extremities) are easily attributed to other causes, making it frequently missed until significant damage has occurred.",
            "actions": [
                "Switch to methylcobalamin or adenosylcobalamin B12 — cyanocobalamin (the most common supplement form) requires conversion steps that your MTRR variant impairs; methylcobalamin bypasses this",
                "Take 1,000-2,000 mcg sublingual daily — sublingual absorption bypasses the gut absorption deficit from your FUT2 variant entirely",
                "Test methylmalonic acid (MMA) — this is the functional marker for B12 sufficiency at the cellular level; elevated MMA with normal serum B12 confirms your variant pattern is causing deficiency",
                "Target serum B12 above 600 pg/mL as your personal floor — your variants mean you need higher serum levels to achieve adequate intracellular levels",
                "Retest B12 and MMA 3 months after switching to methylcobalamin — expect MMA normalization within 60-90 days with adequate dosing"
            ]
        }
    },
    "folate": {
        "aliases": ["folate", "folic acid", "serum folate", "rbc folate", "red blood cell folate"],
        "unit": "ng/mL", "category": "Vitamins",
        "standard_low": 3, "standard_high": 20,
        "optimal_low": 10, "optimal_high": 25,
        "clinical_name": "Folate",
        "interpretation": {
            "optimal": "Folate is in the optimal range.",
            "normal": "Folate is within standard range.",
            "high": "Folate is above standard range.",
            "low": "Folate is deficient — methylation, DNA synthesis, and neural tube risk."
        },
        "genomic_snps": ["rs1801133", "rs1801131", "rs1801394"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your MTHFR C677T variant reduces the enzyme that converts dietary folate and folic acid into the active form your cells actually use — 5-methyltetrahydrofolate (5-MTHF). Enzyme activity is reduced 40-50% in heterozygous carriers and 70-80% in homozygous carriers. Standard serum folate tests measure total folate including unconverted, biologically inactive forms — making your lab result misleadingly reassuring even when intracellular methylfolate is insufficient.",
            "implications": "Functional methylfolate deficiency with your MTHFR variant elevates homocysteine (a direct vascular toxin), impairs DNA methylation and repair, reduces SAMe production (your body's universal methyl donor), and compromises neurotransmitter synthesis (serotonin, dopamine, and norepinephrine all require methylfolate cofactors). The critical insight: standard folic acid supplementation will not fix this — unconverted folic acid accumulates and can actually inhibit transport of what little methylfolate your compromised enzyme does produce.",
            "actions": [
                "Switch immediately to methylfolate (L-5-MTHF) — the pre-converted active form that completely bypasses your MTHFR bottleneck; 400-800 mcg/day is a reasonable starting dose",
                "Discontinue any supplements containing folic acid — this includes most B-complex vitamins, prenatal vitamins, and fortified foods; read labels carefully",
                "Pair methylfolate with methylcobalamin B12 (1,000 mcg/day) — the methylation cycle requires both cofactors together; B12 deficiency stalls the cycle even with adequate methylfolate",
                "Test homocysteine 8 weeks after switching to methylfolate — expect a 30-50% reduction; persistent elevation suggests you also need P5P (active B6) and riboflavin (B2) supplementation",
                "Eat leafy greens daily (spinach, romaine, arugula) — dietary folate from whole foods is in the natural 5-MTHF form that does not require MTHFR conversion, unlike folic acid in supplements and fortified foods"
            ]
        }
    },
    "magnesium": {
        "aliases": ["magnesium", "serum magnesium", "magnesium, serum"],
        "unit": "mg/dL", "category": "Minerals",
        "standard_low": 1.7, "standard_high": 2.5,
        "optimal_low": 2.0, "optimal_high": 2.5,
        "clinical_name": "Magnesium",
        "interpretation": {
            "optimal": "Serum magnesium is in the optimal range. Note: serum magnesium is a poor proxy for intracellular status — RBC magnesium is more informative.",
            "normal": "Serum magnesium is within standard range.",
            "high": "Serum magnesium is above standard range.",
            "low": "Serum magnesium is low — deficiency affects >300 enzymatic reactions including energy production, glucose metabolism, and sleep quality."
        },
        "genomic_snps": ["rs1801133"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your MTHFR variant increases your functional demand for magnesium by accelerating its consumption in methylation reactions. Magnesium is a required cofactor for the MTHFR enzyme itself — meaning your compromised enzyme needs more magnesium to function. Additionally, serum magnesium is a poor proxy for total body magnesium status: 99% of magnesium is intracellular, and serum levels stay 'normal' until intracellular depletion is significant.",
            "implications": "Magnesium deficiency is implicated in over 300 enzymatic reactions. For your specific variant profile, the most consequential deficits are: impaired ATP synthesis (energy production requires magnesium-ATP complexes), reduced insulin sensitivity (magnesium is required for insulin receptor signaling), poor sleep quality (magnesium gates NMDA receptors and regulates melatonin), cardiac arrhythmia risk (magnesium controls cardiac ion channels), and worsened anxiety and HPA axis hyperreactivity. These effects compound with your existing MTHFR-related methylation stress.",
            "actions": [
                "Target serum magnesium at the upper optimal range (2.2-2.5 mg/dL) — given your elevated methylation demand, low-normal is likely insufficient",
                "Request RBC magnesium testing — this measures intracellular magnesium and is far more informative than serum; target RBC magnesium above 5.5 mg/dL",
                "Supplement with magnesium glycinate or threonate 300-400 mg elemental magnesium daily — these forms have superior absorption and minimal GI side effects compared to magnesium oxide",
                "Avoid magnesium oxide supplementation — it has less than 5% bioavailability and is primarily a laxative, not a therapeutic form",
                "Eat magnesium-rich foods daily: dark leafy greens, pumpkin seeds, dark chocolate, legumes — aim to get 400+ mg from food before adding supplements"
            ]
        }
    },
    "zinc": {
        "aliases": ["zinc", "serum zinc", "zinc, serum", "zinc, plasma"],
        "unit": "ug/dL", "category": "Minerals",
        "standard_low": 60, "standard_high": 120,
        "optimal_low": 80, "optimal_high": 120,
        "clinical_name": "Zinc",
        "interpretation": {
            "optimal": "Zinc is in the optimal range.",
            "normal": "Zinc is within standard range but below optimal.",
            "high": "Zinc is above standard range — excess zinc can interfere with copper absorption.",
            "low": "Zinc is deficient — immune function, hormone synthesis, and wound healing affected."
        },
        "genomic_snps": ["rs2228570"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your VDR variant affects the vitamin D receptor, which regulates zinc-dependent transcription factors and immune pathways that significantly overlap with vitamin D signaling. VDR and zinc share downstream targets in immune regulation, thymic function, and antioxidant enzyme activity. Individuals with VDR variants who also have suboptimal zinc create a compound deficit in immune modulation that neither deficiency alone would cause to the same degree.",
            "implications": "Zinc deficiency with VDR variants disproportionately impairs thymic hormone production (thymulin requires both zinc and vitamin D signaling), T-cell development and differentiation, testosterone synthesis (zinc is rate-limiting for Leydig cell function), and the antioxidant enzyme superoxide dismutase (Cu/Zn-SOD). Your immune system's ability to mount appropriately regulated responses — neither under- nor over-reacting — is compromised by both inputs simultaneously, with particular relevance for your autoimmune risk profile.",
            "actions": [
                "Target serum zinc at 90-110 ug/dL — the upper portion of the optimal range, given your elevated immune regulatory demand",
                "Supplement zinc picolinate or bisglycinate 15-25 mg/day with food — these forms have significantly better absorption than zinc oxide or sulfate",
                "Always pair zinc supplementation with copper at an 8:1-15:1 ratio (e.g., 15mg zinc + 1-2mg copper) — long-term zinc supplementation without copper causes copper deficiency",
                "Eat zinc-rich foods: oysters (the richest source by far), red meat, pumpkin seeds, hemp seeds — vegetarians should supplement given phytate-reduced absorption from plant sources",
                "Optimize vitamin D simultaneously — VDR function improves with adequate D3 (target 60-80 ng/mL), which synergizes with zinc in the immune pathways your variant affects"
            ]
        }
    },

    # =========================================================================
    # THYROID
    # =========================================================================
    "tsh": {
        "aliases": ["tsh", "thyroid stimulating hormone", "thyroid-stimulating hormone", "thyrotropin"],
        "unit": "mIU/L", "category": "Thyroid",
        "standard_low": 0.4, "standard_high": 4.5,
        "optimal_low": 0.5, "optimal_high": 2.0,
        "clinical_name": "TSH (Thyroid Stimulating Hormone)",
        "interpretation": {
            "optimal": "TSH is in the longevity-optimal range — thyroid function appears well-regulated.",
            "normal": "TSH is within standard range but above optimal — subclinical hypothyroid pattern warrants monitoring.",
            "high": "TSH is elevated — hypothyroidism likely. Evaluate with Free T4 and Free T3.",
            "low": "TSH is suppressed — hyperthyroidism or excessive thyroid hormone replacement."
        },
        "genomic_snps": ["rs2476601", "rs3087243"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your PTPN22 and/or CTLA4 variants impair immune checkpoint regulation, making your immune system more likely to attack self-tissue — including the thyroid. Hashimoto's thyroiditis, the autoimmune destruction of thyroid tissue, is the most common thyroid disease and progresses silently for years before TSH becomes overtly abnormal. Your TSH trending above 2.0 mIU/L is an early signal that warrants attention.",
            "implications": "Undetected Hashimoto's leads to progressive thyroid tissue destruction. By the time TSH crosses the diagnostic hypothyroidism threshold (>4.5), typically 30-50% of thyroid tissue has been destroyed. Subclinical hypothyroidism (TSH 2.0-4.5) is associated with elevated cardiovascular risk, depression, cognitive slowing, weight resistance, and infertility — all before a doctor would typically treat. With your autoimmune genetics, early detection is the difference between arresting the disease and spending decades managing the consequences.",
            "actions": [
                "Request thyroid antibody testing: TPO antibodies and Tg antibodies — positive antibodies confirm Hashimoto's even with normal TSH and allow treatment to begin before further destruction occurs",
                "Selenium supplementation 200 mcg/day — selenium is required for thyroid hormone synthesis and has been shown in multiple trials to reduce TPO antibody titers by 20-40%, directly slowing the autoimmune attack",
                "Optimize vitamin D to 60-80 ng/mL — vitamin D deficiency is strongly associated with autoimmune thyroid disease and its correction is associated with reduced antibody levels",
                "Consider a 90-day gluten elimination trial and retest antibodies — molecular mimicry between gliadin and thyroid tissue is documented; a subset of Hashimoto's patients show significant antibody reduction with gluten removal",
                "Track TSH, Free T3, and Free T4 annually — the longitudinal trend is more informative than any single measurement for your genotype"
            ]
        }
    },
    "free_t4": {
        "aliases": ["free t4", "ft4", "thyroxine free", "free thyroxine", "t4 free"],
        "unit": "ng/dL", "category": "Thyroid",
        "standard_low": 0.8, "standard_high": 1.8,
        "optimal_low": 1.0, "optimal_high": 1.5,
        "clinical_name": "Free T4",
        "interpretation": {
            "optimal": "Free T4 is in the optimal range.",
            "normal": "Free T4 is within standard range.",
            "high": "Free T4 is elevated — hyperthyroid pattern.",
            "low": "Free T4 is low — hypothyroid pattern."
        },
        "genomic_snps": ["rs2476601"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your PTPN22 autoimmune variant predisposes to Hashimoto's thyroiditis — autoimmune destruction of thyroid tissue that reduces thyroid hormone output over time. Free T4 directly reflects thyroid gland secretion, and values trending toward the lower half of the standard range in the context of autoimmune genetics warrant careful longitudinal tracking. Hashimoto's is characterized by fluctuating thyroid function before progressive decline — Free T4 may be normal while TPO antibodies confirm active tissue destruction.",
            "implications": "Free T4 in the lower third of the standard range (0.8-1.0 ng/dL) combined with autoimmune genetic risk represents a pattern that will likely progress to overt hypothyroidism without intervention. Standard clinical practice does not treat until Free T4 drops below 0.8 — but progressive thyroid tissue destruction continues silently in the subclinical range. Untreated Hashimoto's leads to cumulative fatigue, weight resistance, depression, cognitive slowing, cardiovascular risk elevation, and infertility.",
            "actions": [
                "Test thyroid peroxidase (TPO) antibodies and thyroglobulin (Tg) antibodies — positive antibodies confirm active Hashimoto's autoimmunity even when Free T4 appears normal",
                "Track Free T4, Free T3, and TSH together longitudinally at least annually — the trend over 2-3 years is more informative than any single measurement",
                "Selenium 200 mcg/day — reduces TPO antibody titers by 20-40% in clinical trials and is the primary evidence-based supplement for slowing Hashimoto's progression",
                "Optimize vitamin D to 60-80 ng/mL — vitamin D deficiency is strongly associated with autoimmune thyroid disease incidence and progression",
                "If Free T4 drops below 1.0 ng/mL with TSH above 2.0 and positive antibodies, discuss early levothyroxine therapy with your physician — emerging evidence supports earlier treatment in antibody-positive patients"
            ]
        }
    },
    "free_t3": {
        "aliases": ["free t3", "ft3", "triiodothyronine free", "free triiodothyronine", "t3 free"],
        "unit": "pg/mL", "category": "Thyroid",
        "standard_low": 2.0, "standard_high": 4.4,
        "optimal_low": 3.0, "optimal_high": 4.2,
        "clinical_name": "Free T3",
        "interpretation": {
            "optimal": "Free T3 is in the optimal range — active thyroid hormone levels are adequate.",
            "normal": "Free T3 is within standard range but below optimal — poor T4-to-T3 conversion may be occurring.",
            "high": "Free T3 is elevated — hyperthyroid pattern.",
            "low": "Free T3 is low — hypothyroid pattern or poor T4-to-T3 conversion."
        },
        "genomic_snps": ["rs2476601"],
        "genomic_note": {
            "direction": "worse",
            "context": "Free T3 is the active thyroid hormone — the form that actually enters cells and regulates metabolism, body temperature, cardiac output, and cognitive function. T3 is primarily produced by peripheral conversion of T4 by deiodinase enzymes (DIO1, DIO2). Certain variants in these genes impair this conversion, causing Free T3 to remain low even when TSH and Free T4 are technically normal — a pattern called poor T4-to-T3 conversion that is systematically missed by TSH-only thyroid panels.",
            "implications": "Low Free T3 with normal TSH and T4 means your thyroid gland is working but your cells are not receiving adequate active hormone signal. Symptoms are identical to hypothyroidism (fatigue, brain fog, cold intolerance, weight resistance, depression) but will not be detected by TSH-only testing, which is the standard of care in most primary care settings. Selenium deficiency, chronic stress, caloric restriction, and systemic inflammation all worsen T4-to-T3 conversion — all of which are amplified by your inflammatory genetic variants.",
            "actions": [
                "Always test Free T3 alongside TSH and Free T4 — TSH alone will not detect conversion impairment",
                "Your Free T3 target is the upper half of the optimal range (3.5-4.2 pg/mL), not merely above the lower standard limit",
                "Selenium 200 mcg/day is the most evidence-supported intervention for improving T4-to-T3 conversion — selenium is the cofactor for all three deiodinase enzymes",
                "Reduce systemic inflammation (hs-CRP, omega-3 supplementation) — TNF-alpha and IL-6 directly inhibit DIO2 and shift T4 toward reverse T3 rather than active T3",
                "If Free T3 consistently runs in the lower third of range despite optimization, discuss T3 supplementation (liothyronine) or combination T4/T3 therapy with an endocrinologist"
            ]
        }
    },
    "reverse_t3": {
        "aliases": ["reverse t3", "rt3", "reverse triiodothyronine"],
        "unit": "ng/dL", "category": "Thyroid",
        "standard_low": 9.2, "standard_high": 24.1,
        "optimal_low": 9, "optimal_high": 15,
        "clinical_name": "Reverse T3",
        "interpretation": {
            "optimal": "Reverse T3 is in the optimal range.",
            "normal": "Reverse T3 is elevated within standard range — stress, caloric restriction, or inflammation may be driving T4 toward inactive rT3 rather than active T3.",
            "high": "Reverse T3 is elevated — significant rT3 dominance pattern; evaluate for chronic stress, inflammation, cortisol dysregulation.",
            "low": "Reverse T3 is very low — generally favorable."
        },
        "genomic_snps": ["rs1800629"],
        "genomic_note": {
            "direction": "worse",
            "context": "Reverse T3 (rT3) is an inactive isomer of T3 that competes with active T3 for cellular receptor binding — it is effectively a braking signal that slows metabolism when the body perceives stress or resource scarcity. Your TNF-alpha variant chronically activates the same inflammatory signaling pathways that shift T4 conversion toward rT3 rather than active T3. This means your genomic inflammation is directly impairing thyroid hormone utilization, even when your TSH appears normal.",
            "implications": "Elevated reverse T3 with your pro-inflammatory genotype creates a functional hypothyroid state at the cellular level despite normal standard thyroid labs. The ratio of Free T3 to reverse T3 (Free T3/rT3 ratio, optimal above 20) is more diagnostically useful than rT3 alone. Chronically elevated rT3 is associated with fatigue, weight resistance, cold intolerance, depression, and cognitive slowing — all of which are misattributed to other causes when the rT3 mechanism is not investigated.",
            "actions": [
                "Calculate your Free T3/rT3 ratio — optimal is above 20; below 15 indicates significant rT3 dominance and warrants clinical evaluation",
                "Treating the underlying inflammation is the primary intervention — reduce TNF-alpha and IL-6 (omega-3, curcumin, sleep optimization) rather than just treating the thyroid readout",
                "Avoid very low calorie diets (below 1,400 kcal/day) — severe caloric restriction is the most potent trigger for rT3 upregulation; it signals starvation and the body responds by downregulating metabolism",
                "Address cortisol dysregulation — elevated cortisol directly shifts T4 toward rT3; assess with 4-point salivary cortisol testing if adrenal stress is suspected",
                "Selenium 200 mcg/day supports selenoprotein deiodinase function, influencing the balance between T3 and rT3 production"
            ]
        }
    },

    # =========================================================================
    # HORMONAL
    # =========================================================================
    "testosterone_total": {
        "aliases": ["testosterone", "testosterone total", "total testosterone", "testosterone, total", "testosterone, serum"],
        "unit": "ng/dL", "category": "Hormonal",
        "standard_low": 264, "standard_high": 916,
        "optimal_low": 600, "optimal_high": 900,
        "clinical_name": "Total Testosterone",
        "interpretation": {
            "optimal": "Total testosterone is in the longevity-optimal range.",
            "normal": "Total testosterone is within standard range but below longevity-optimal — evaluate free testosterone and SHBG.",
            "high": "Total testosterone is above standard range.",
            "low": "Total testosterone is below standard range — hypogonadism evaluation warranted."
        },
        "genomic_snps": ["rs6152", "rs523349", "rs1799941", "rs6257"],
        "genomic_note": {
            "direction": "neutral",
            "context": "Your androgen receptor (AR) sensitivity variants and SRD5A2 (5-alpha reductase) variants mean testosterone's actual effect at target tissues does not scale linearly with your serum total testosterone level. AR variants alter receptor binding affinity — some require higher testosterone levels to achieve the same cellular signal. SRD5A2 variants change how efficiently testosterone is converted to dihydrotestosterone (DHT), the more potent androgen active in muscle, brain, and prostate. SHBG variants then determine how much total testosterone is bound (inactive) vs. free (bioactive).",
            "implications": "Total testosterone in the lower-normal range (350-500 ng/dL) with high SHBG and AR sensitivity variants can result in functionally deficient androgenic signaling despite 'normal' labs — explaining fatigue, reduced muscle accretion, libido decline, and mood changes in men with technically normal testosterone. Conversely, high SRD5A2 activity with normal testosterone can increase DHT-related effects (hair loss, prostate stimulation) disproportionately. Total testosterone is a starting point, not a conclusion.",
            "actions": [
                "Always interpret total testosterone alongside free testosterone and SHBG — these three together tell the real story that total testosterone alone cannot",
                "Your personal total testosterone target should be in the upper optimal range (700-900 ng/dL) given that AR variants and SHBG variants reduce effective androgenic signaling per unit testosterone",
                "Resistance training 3-4x/week is the most evidence-based lifestyle intervention for optimizing testosterone — particularly compound lifts (deadlift, squat, bench press)",
                "Optimize sleep (7.5-9 hours) — 60-80% of daily testosterone is produced during deep sleep; chronic sleep restriction of even 5 days reduces testosterone by 15-20%",
                "Zinc (25 mg/day) and vitamin D (target 60-80 ng/mL) are the two most evidence-supported micronutrients for testosterone optimization in deficient individuals"
            ]
        }
    },
    "testosterone_free": {
        "aliases": ["free testosterone", "testosterone free", "testosterone, free", "bioavailable testosterone"],
        "unit": "pg/mL", "category": "Hormonal",
        "standard_low": 5, "standard_high": 21,
        "optimal_low": 12, "optimal_high": 20,
        "clinical_name": "Free Testosterone",
        "interpretation": {
            "optimal": "Free testosterone is in the longevity-optimal range.",
            "normal": "Free testosterone is within standard range but below optimal.",
            "high": "Free testosterone is above standard range.",
            "low": "Free testosterone is low — may be low despite normal total testosterone if SHBG is elevated."
        },
        "genomic_snps": ["rs6152", "rs1799941", "rs6257"],
        "genomic_note": {
            "direction": "worse",
            "context": "Free testosterone is the fraction not bound to SHBG or albumin — the only form immediately available for cellular uptake. Your SHBG variants set your binding protein levels genetically, meaning that even with adequate total testosterone production, SHBG can sequester a disproportionate fraction, leaving free testosterone low. AR sensitivity variants then determine how efficiently target tissues respond to the free testosterone that does reach them.",
            "implications": "Free testosterone below 12 pg/mL, even with total testosterone in the 'normal' range, explains many androgen-deficiency symptoms that physicians miss when only ordering total testosterone. Muscle mass maintenance, libido, mood stability, cognitive sharpness, and bone density are all regulated by free androgen signal — not total. High-SHBG genotypes are particularly prone to this discordance pattern. It is possible to have a total testosterone of 600 ng/dL and be functionally androgen-deficient if SHBG is above 70 nmol/L.",
            "actions": [
                "Track free testosterone and SHBG at every hormonal checkup — these are more informative than total testosterone for your genotype",
                "Your free testosterone target is 15-20 pg/mL — the upper portion of the optimal range given your SHBG binding characteristics",
                "Interventions that lower SHBG increase free testosterone: improving insulin sensitivity, modest zinc supplementation, and boron (6-12 mg/day) has emerging evidence as an SHBG reducer",
                "Avoid unnecessary thyroid hormone excess — elevated T4 increases SHBG production in the liver",
                "If free testosterone remains below 12 pg/mL despite lifestyle optimization, discuss TRT evaluation with an endocrinologist who measures free testosterone rather than total only"
            ]
        }
    },
    "shbg": {
        "aliases": ["shbg", "sex hormone binding globulin", "sex hormone-binding globulin"],
        "unit": "nmol/L", "category": "Hormonal",
        "standard_low": 10, "standard_high": 57,
        "optimal_low": 20, "optimal_high": 40,
        "clinical_name": "SHBG (Sex Hormone Binding Globulin)",
        "interpretation": {
            "optimal": "SHBG is in the optimal range.",
            "normal": "SHBG is within standard range.",
            "high": "SHBG is elevated — reduces bioavailable testosterone and estrogen. Evaluate insulin sensitivity (insulin lowers SHBG).",
            "low": "SHBG is low — increases free sex hormones; associated with insulin resistance and metabolic syndrome."
        },
        "genomic_snps": ["rs1799941", "rs6257", "rs13164856"],
        "genomic_note": {
            "direction": "neutral",
            "context": "Your SHBG gene variants directly set your sex hormone binding globulin concentration at a genetically determined baseline. SHBG is produced by the liver and binds tightly to both testosterone and estradiol, rendering bound hormone biologically inactive. The direction of your SHBG variant's effect (high vs. low SHBG) determines whether you trend toward androgen deficiency (high SHBG) or excess free sex hormones (low SHBG) — both of which carry distinct clinical implications that are missed when SHBG is not measured.",
            "implications": "High SHBG reduces free testosterone and estradiol, creating functional deficiency in androgenic and estrogenic signaling despite normal total hormone levels. Low SHBG has the opposite effect — increased free testosterone and estradiol, which in men is associated with increased aromatization to estrogen and in women with androgen excess (PCOS pattern). SHBG is also inversely correlated with insulin resistance — low SHBG is an independent marker of metabolic syndrome risk regardless of hormone levels.",
            "actions": [
                "Interpret SHBG in the context of your free testosterone and estradiol levels — SHBG alone without hormone context is insufficient for clinical decisions",
                "If SHBG is high (above 45 nmol/L): consider boron 6-12 mg/day which has modest evidence for reducing SHBG; optimize insulin sensitivity as chronically elevated insulin suppresses SHBG production in some contexts",
                "If SHBG is low (below 20 nmol/L): treat it as an independent marker of insulin resistance — weight loss and metabolic health improvement reduce free sex hormone excess more reliably than any supplement",
                "Monitor estradiol alongside SHBG and testosterone — low SHBG in men can lead to estradiol elevation as free testosterone aromatizes at higher rates",
                "Retest SHBG with fasting insulin and free testosterone at the same draw — these form a hormonal triad that tells a complete metabolic and endocrine story"
            ]
        }
    },
    "estradiol": {
        "aliases": ["estradiol", "e2", "estradiol, serum", "17-beta estradiol"],
        "unit": "pg/mL", "category": "Hormonal",
        "standard_low": 7.6, "standard_high": 42.6,
        "optimal_low": 20, "optimal_high": 40,
        "clinical_name": "Estradiol (E2)",
        "interpretation": {
            "optimal": "Estradiol is in the optimal range (male reference used — adjust for female cycle phase).",
            "normal": "Estradiol is within standard range.",
            "high": "Estradiol is elevated — evaluate aromatase activity, adiposity, and alcohol intake.",
            "low": "Estradiol is low — affects bone density, cardiovascular health, and cognitive function."
        },
        "genomic_snps": ["rs10046", "rs700518", "rs4775936", "rs4680", "rs1056836"],
        "genomic_note": {
            "direction": "worse",
            "context": "You carry variants affecting multiple estrogen metabolism pathways simultaneously. CYP19A1 (aromatase) variants alter the rate at which testosterone converts to estradiol. COMT Val158Met variants slow the breakdown of catechol estrogens in the liver — meaning estrogen metabolites accumulate, particularly the genotoxic 4-OH estrogen metabolite. CYP1B1 variants shunt estrogen metabolism preferentially toward this same genotoxic 4-OH pathway. This combination creates both elevated estrogen load and more dangerous estrogen metabolite accumulation.",
            "implications": "The clinical consequence is not simply 'high' or 'low' estrogen — it is an imbalance in how estrogen is processed and eliminated. 4-OH estrogen metabolites form DNA adducts that are directly genotoxic, linking this pattern to elevated breast cancer risk in women and prostate cancer risk in men. Slow COMT allows catechol estrogens to accumulate beyond the liver's detoxification capacity. Even within the 'normal' estradiol range, this combination represents a more dangerous hormonal metabolism pattern than a simple number would suggest.",
            "actions": [
                "Test estrogen metabolites through a DUTCH urine hormone panel — this distinguishes which metabolic pathways are dominant in your case and guides targeted intervention",
                "Eat cruciferous vegetables daily (broccoli, Brussels sprouts, cauliflower) — contain DIM and I3C precursors that shift estrogen metabolism toward the safer 2-OH pathway rather than the genotoxic 4-OH pathway your CYP1B1 variant favors",
                "DIM supplementation 100-200 mg/day — concentrated indole-3-carbinol derivative that upregulates 2-OH estrogen metabolism and reduces 4-OH metabolite burden",
                "Calcium-D-glucarate 1,500 mg/day — inhibits beta-glucuronidase, the enzyme that re-activates conjugated estrogens in the gut (enterohepatic recirculation), reducing total estrogen load",
                "Minimize alcohol intake — alcohol directly inhibits COMT enzyme activity, compounding your genetic slow-COMT pattern and further slowing estrogen clearance"
            ]
        }
    },
    "dhea_s": {
        "aliases": ["dhea-s", "dheas", "dhea sulfate", "dehydroepiandrosterone sulfate", "dhea-sulfate"],
        "unit": "ug/dL", "category": "Hormonal",
        "standard_low": 80, "standard_high": 560,
        "optimal_low": 250, "optimal_high": 450,
        "clinical_name": "DHEA-S",
        "interpretation": {
            "optimal": "DHEA-S is in the longevity-optimal range.",
            "normal": "DHEA-S is within standard range but declining trend warrants monitoring as an aging biomarker.",
            "high": "DHEA-S is elevated — evaluate adrenal function.",
            "low": "DHEA-S is low — adrenal insufficiency or accelerated aging pattern. DHEA supplementation is evidence-based for levels below optimal."
        },
        "genomic_snps": ["rs743572", "rs1764391"],
        "genomic_note": {
            "direction": "neutral",
            "context": "Your CYP17A1 variants affect the enzyme that catalyzes key steps in both DHEA and cortisol synthesis in the adrenal glands. CYP17A1 sits at a metabolic branch point — variants can shift substrate flow preferentially toward either glucocorticoid (cortisol) or androgen (DHEA, androstenedione) pathways. Depending on your variant's functional direction, this can result in chronically lower DHEA-S relative to cortisol output, or vice versa — creating a stress hormone imbalance that is genetically set rather than purely lifestyle-driven.",
            "implications": "DHEA-S declines with age at approximately 2% per year after peak production in your mid-20s — but your genetic variant may accelerate this decline or set your peak lower than average. DHEA-S below 250 ug/dL is associated with accelerated aging markers including reduced immune function, lower bone mineral density, impaired insulin sensitivity, and higher all-cause mortality in aging populations. The DHEA-cortisol ratio is clinically as important as DHEA-S alone — a chronically low ratio indicates your HPA axis is prioritizing cortisol at the expense of anabolic hormone output.",
            "actions": [
                "Track DHEA-S alongside morning cortisol at the same blood draw — the ratio is more informative than DHEA-S in isolation",
                "Your DHEA-S target is 300-450 ug/dL — the upper portion of the longevity-optimal range associated with best outcomes in aging research",
                "DHEA supplementation 25-50 mg/day (micronized, pharmaceutical grade) is clinically validated for individuals with levels below optimal — start at 25 mg and retest after 90 days",
                "Chronic psychological stress is the primary lifestyle driver of low DHEA-S — cortisol production during stress consumes the same substrate (pregnenolone) that DHEA production requires, creating a direct competition",
                "Sleep optimization is non-negotiable — the majority of DHEA production occurs during slow-wave sleep; 7-9 hours of quality sleep directly supports adrenal DHEA output"
            ]
        }
    },
    "cortisol": {
        "aliases": ["cortisol", "serum cortisol", "cortisol, serum", "cortisol, am"],
        "unit": "ug/dL", "category": "Hormonal",
        "standard_low": 6, "standard_high": 23,
        "optimal_low": 10, "optimal_high": 18,
        "clinical_name": "Cortisol (AM)",
        "interpretation": {
            "optimal": "Morning cortisol is in the optimal range.",
            "normal": "Morning cortisol is within standard range.",
            "high": "Morning cortisol is elevated — chronic stress, Cushing's syndrome, or adrenal hyperactivity.",
            "low": "Morning cortisol is low — adrenal insufficiency or HPA axis dysregulation."
        },
        "genomic_snps": ["rs743572", "rs1764391", "rs4680"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your CYP17A1 variants affect the enzymatic synthesis of cortisol in the adrenal cortex, potentially altering both baseline production and stress response magnitude. Your COMT Val158Met variant then slows the clearance of catecholamines (epinephrine, norepinephrine) from your system — meaning the stress response triggered by cortisol release takes longer to wind down. This creates a pattern of prolonged stress recovery: your cortisol spike from the same stressor is both higher and longer-lasting than it would be without these variants.",
            "implications": "Chronically elevated cortisol — even within the 'normal' laboratory range — is one of the most destructive hormonal patterns in long-term health. Cortisol above optimal suppresses immune function, breaks down muscle and bone tissue, impairs hippocampal function (memory and learning), elevates blood glucose, promotes visceral fat deposition, and suppresses DHEA production. Your COMT slow-clearance variant amplifies all of these effects by extending the time cortisol's downstream effects persist after each stressor.",
            "actions": [
                "Test cortisol as a 4-point salivary diurnal profile (morning, noon, evening, midnight) rather than a single AM blood draw — the pattern across the day is more informative than a single peak level",
                "Your optimal morning cortisol is 12-18 ug/dL, consistent with the natural cortisol awakening response; values above 20 suggest chronic HPA hyperactivation",
                "Phosphatidylserine 300-600 mg/day is the most evidence-supported supplement for attenuating cortisol response — shown to reduce exercise and psychological stress-induced cortisol by 20-30%",
                "Ashwagandha (KSM-66 extract, 300-600 mg/day) has the strongest evidence base among adaptogens for reducing cortisol — multiple RCTs show 20-30% reduction in chronic stress contexts",
                "Cold-to-warm contrast exposure (cold shower followed by sauna) trains HPA axis regulation and improves cortisol recovery speed, directly targeting your COMT-related slow stress recovery"
            ]
        }
    },
    "igf1": {
        "aliases": ["igf-1", "igf1", "insulin-like growth factor 1", "insulin like growth factor"],
        "unit": "ng/mL", "category": "Hormonal",
        "standard_low": 88, "standard_high": 246,
        "optimal_low": 150, "optimal_high": 220,
        "clinical_name": "IGF-1",
        "interpretation": {
            "optimal": "IGF-1 is in the longevity-optimal range.",
            "normal": "IGF-1 is within standard range.",
            "high": "IGF-1 is elevated — may accelerate cellular proliferation; evaluate GH axis.",
            "low": "IGF-1 is low — growth hormone deficiency pattern; affects muscle mass, bone density, and recovery."
        },
        "genomic_snps": ["rs934198", "rs2229765"],
        "genomic_note": {
            "direction": "neutral",
            "context": "Your IGF1 and IGF1R variants affect either insulin-like growth factor production or how efficiently your cells respond to IGF-1 signaling. This creates a nuanced interpretation challenge: lower IGF-1 from IGF1R loss-of-function variants is associated with longevity in multiple animal models and human centenarian studies, while higher IGF-1 supports muscle mass, bone density, and recovery. The optimal range for longevity may differ from the optimal range for physical performance.",
            "implications": "IGF-1 is a powerful mitogenic signal — it promotes cellular growth and proliferation, which is beneficial for muscle and bone but potentially disadvantageous for cancer risk if chronically elevated. Individuals with IGF1R variants associated with reduced receptor signaling may require higher circulating IGF-1 to achieve the same anabolic signal. The longevity-optimal IGF-1 range (150-220 ng/mL) attempts to balance anabolic benefit against proliferative risk.",
            "actions": [
                "Aim for IGF-1 in the 150-220 ng/mL range — high enough to support muscle, bone, and recovery; not so high as to drive proliferative risk",
                "IGF-1 is highly responsive to protein intake — adequate protein (1.6-2.2g/kg body weight/day) supports IGF-1 in the optimal range without driving chronic excess",
                "Resistance training and high-intensity intervals are the primary natural drivers of IGF-1 — they raise it to optimal levels without the chronic elevation that caloric excess creates",
                "Sleep is essential for IGF-1 production — growth hormone (which stimulates IGF-1 production in the liver) is secreted primarily during deep sleep; prioritize 7.5-9 hours",
                "If IGF-1 is below 150 ng/mL: optimize sleep, protein intake, and resistance training first; evaluate for growth hormone deficiency with an endocrinologist if it remains low despite lifestyle optimization"
            ]
        }
    },

    # =========================================================================
    # RENAL FUNCTION
    # =========================================================================
    "creatinine": {
        "aliases": ["creatinine", "serum creatinine", "creatinine, serum"],
        "unit": "mg/dL", "category": "Renal",
        "standard_low": 0.57, "standard_high": 1.0,
        "optimal_low": 0.6, "optimal_high": 0.9,
        "clinical_name": "Creatinine",
        "interpretation": {
            "optimal": "Creatinine is in the optimal range.",
            "normal": "Creatinine is within standard range.",
            "high": "Creatinine is elevated — impaired renal function or dehydration.",
            "low": "Creatinine is low — may reflect low muscle mass."
        },
        "genomic_snps": [],
        "genomic_note": ""
    },
    "egfr": {
        "aliases": ["egfr", "gfr", "estimated gfr", "glomerular filtration rate", "creatinine egfr"],
        "unit": "mL/min/1.73m2", "category": "Renal",
        "standard_low": 60, "standard_high": 999,
        "optimal_low": 90, "optimal_high": 120,
        "clinical_name": "eGFR (Estimated Glomerular Filtration Rate)",
        "interpretation": {
            "optimal": "eGFR reflects normal kidney function.",
            "normal": "eGFR is in the mildly reduced range — monitor for progression.",
            "high": "eGFR is very high — generally favorable.",
            "low": "eGFR is below 60 — chronic kidney disease evaluation warranted."
        },
        "genomic_snps": ["rs1800629"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TNF-alpha variant sustains chronically elevated inflammatory signaling that, over years to decades, contributes to progressive renal function decline. The kidney is particularly vulnerable to chronic inflammatory load — TNF-alpha and IL-6 directly promote mesangial cell proliferation, podocyte injury, and interstitial fibrosis, which are the cellular mechanisms underlying CKD progression. Your current eGFR may be entirely normal, but the trajectory over decades with uncontrolled inflammation is meaningfully worse than for someone with anti-inflammatory genetics.",
            "implications": "eGFR declines at approximately 0.5-1 mL/min/1.73m2 per year in healthy adults. With a chronic pro-inflammatory genotype, this rate can be 2-3x higher in the presence of other risk factors (hypertension, diabetes, NSAID use, dehydration). CKD stage 3 (eGFR 30-59) is associated with dramatically elevated cardiovascular risk — the same inflammatory milieu that damages the kidney is also damaging the heart and vessels. Preserving eGFR is both a renal and cardiovascular priority.",
            "actions": [
                "Track eGFR and creatinine at every annual bloodwork — the rate of change over time is more important than any single value",
                "Keep hs-CRP below 1.0 mg/L as a proxy for renal-protective inflammation control — this is your single most important renal preservation target",
                "Avoid regular NSAID use (ibuprofen, naproxen) — NSAIDs reduce renal blood flow and are the most common medication-related cause of eGFR decline; use acetaminophen for pain when possible",
                "Stay well hydrated (pale yellow urine throughout the day) — chronic mild dehydration concentrates inflammatory mediators in the renal tubules and accelerates filtration membrane damage",
                "Blood pressure control is essential — even borderline hypertension (130-139/85-89 mmHg) accelerates eGFR decline by 30-50% in the context of underlying inflammation"
            ]
        }
    },
    "bun": {
        "aliases": ["bun", "blood urea nitrogen", "urea nitrogen"],
        "unit": "mg/dL", "category": "Renal",
        "standard_low": 7, "standard_high": 25,
        "optimal_low": 10, "optimal_high": 18,
        "clinical_name": "BUN (Blood Urea Nitrogen)",
        "interpretation": {
            "optimal": "BUN is in the optimal range.",
            "normal": "BUN is within standard range.",
            "high": "BUN is elevated — dehydration, high protein intake, or renal dysfunction.",
            "low": "BUN is low — may reflect low protein intake or liver dysfunction."
        },
        "genomic_snps": [],
        "genomic_note": ""
    },
    "uric_acid": {
        "aliases": ["uric acid", "serum uric acid", "uric acid, serum", "urate"],
        "unit": "mg/dL", "category": "Renal",
        "standard_low": 2.4, "standard_high": 7.0,
        "optimal_low": 3.0, "optimal_high": 5.5,
        "clinical_name": "Uric Acid",
        "interpretation": {
            "optimal": "Uric acid is in the longevity-optimal range.",
            "normal": "Uric acid is within standard range but above optimal — gout risk and cardiovascular/metabolic association.",
            "high": "Uric acid is elevated — gout risk, renal stone risk, and metabolic syndrome association.",
            "low": "Uric acid is very low — loss of antioxidant capacity."
        },
        "genomic_snps": ["rs1260326"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your GCKR variant increases hepatic fructose-1-phosphate production during carbohydrate metabolism, which accelerates purine catabolism and generates excess uric acid as a byproduct. GCKR variants that drive elevated triglycerides operate through the same de novo lipogenesis pathway that simultaneously generates excess uric acid — these two biomarkers consistently co-elevate in GCKR carriers. You may have a genetically elevated uric acid baseline that is 0.5-1.5 mg/dL higher than someone with the same diet and no GCKR variant.",
            "implications": "Uric acid above 5.5 mg/dL is independently associated with gout risk, hypertension (uric acid inhibits nitric oxide production in the endothelium, raising blood pressure), insulin resistance (uric acid inhibits insulin signaling in adipose tissue and muscle), and renal function decline. The mechanisms extend beyond gout crystal formation — uric acid is a pro-inflammatory mediator that activates the NLRP3 inflammasome. Your GCKR-driven uric acid elevation means these risks are partially genetic and require more aggressive dietary management to compensate.",
            "actions": [
                "Your personal uric acid target is below 5.0 mg/dL — not the standard cutoff of 7.0 mg/dL; the lower your uric acid within the optimal range, the better your cardiometabolic outcomes",
                "Eliminate fructose and high-fructose corn syrup — fructose is the most potent dietary uric acid driver, directly accelerating the hepatic purine breakdown your GCKR variant already amplifies",
                "Reduce alcohol (especially beer and liquor) — alcohol inhibits renal uric acid excretion while simultaneously generating uric acid through adenosine catabolism",
                "Tart cherry extract 480mg/day has clinical evidence for reducing uric acid and gout flares — it inhibits xanthine oxidase (the enzyme that produces uric acid) and enhances renal excretion",
                "Stay well-hydrated — each additional liter of daily water intake increases renal uric acid excretion meaningfully; targeting 2.5-3L/day is a simple, free intervention"
            ]
        }
    },

    # =========================================================================
    # HEPATIC FUNCTION
    # =========================================================================
    "alt": {
        "aliases": ["alt", "alanine aminotransferase", "sgpt", "alanine transaminase"],
        "unit": "U/L", "category": "Hepatic",
        "standard_low": 0, "standard_high": 40,
        "optimal_low": 0, "optimal_high": 25,
        "clinical_name": "ALT (Alanine Aminotransferase)",
        "interpretation": {
            "optimal": "ALT is in the optimal range — liver not under stress.",
            "normal": "ALT is within standard range but above optimal — early hepatic stress or NAFLD.",
            "high": "ALT is elevated — hepatic inflammation, NAFLD, or hepatotoxin exposure.",
            "low": "ALT is very low — generally favorable."
        },
        "genomic_snps": ["rs780094", "rs7903146"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your GCKR variant drives excess hepatic de novo lipogenesis — converting carbohydrates to fat at an accelerated rate. When fat accumulates in the liver faster than it can be exported as VLDL or oxidized, hepatocytes become lipid-laden (steatosis). ALT is released into the bloodstream when hepatocytes are under stress or dying, making it an early marker of fatty liver disease. ALT elevation in the context of GCKR variants strongly suggests non-alcoholic fatty liver disease (NAFLD) in its early phase, even before imaging would confirm fat accumulation.",
            "implications": "NAFLD progresses silently for years to decades. GCKR carriers have approximately 2x the population baseline risk for NAFLD. Even mild ALT elevation (25-40 U/L) in this context is not 'borderline normal' — it is early hepatocellular stress that, if unaddressed, progresses through NASH (non-alcoholic steatohepatitis), fibrosis, cirrhosis, and hepatocellular carcinoma. The window for reversal with lifestyle intervention is wide in the early phase and narrow once fibrosis begins.",
            "actions": [
                "Request liver ultrasound or FibroScan if ALT is consistently above 25 U/L — these directly assess hepatic steatosis and fibrosis and should be standard for GCKR carriers with elevated ALT",
                "Fructose and refined carbohydrate elimination is the primary intervention — these are the substrate for your GCKR-driven hepatic fat accumulation; the liver can regenerate substantially within weeks on a low-carbohydrate diet",
                "Vitamin E 400 IU/day (natural alpha-tocopherol, not synthetic dl-alpha-tocopherol) — shown in NASH trials to reduce ALT and hepatic inflammation significantly",
                "Coffee consumption (2-4 cups/day, black) has the strongest observational evidence of any dietary factor for reducing NAFLD severity and fibrosis progression",
                "Track ALT with GGT and AST every 3 months after dietary changes — expect ALT to drop 30-50% within 6-8 weeks of effective carbohydrate reduction"
            ]
        }
    },
    "ast": {
        "aliases": ["ast", "aspartate aminotransferase", "sgot", "aspartate transaminase"],
        "unit": "U/L", "category": "Hepatic",
        "standard_low": 0, "standard_high": 40,
        "optimal_low": 0, "optimal_high": 25,
        "clinical_name": "AST (Aspartate Aminotransferase)",
        "interpretation": {
            "optimal": "AST is in the optimal range.",
            "normal": "AST is within standard range but above optimal.",
            "high": "AST is elevated — hepatic or muscle damage. Elevated AST with normal ALT suggests muscle source (check CK).",
            "low": "AST is very low — generally favorable."
        },
        "genomic_snps": [],
        "genomic_note": ""
    },
    "ggtp": {
        "aliases": ["ggt", "ggtp", "gamma-glutamyltransferase", "gamma gt", "gamma-gt"],
        "unit": "U/L", "category": "Hepatic",
        "standard_low": 0, "standard_high": 65,
        "optimal_low": 0, "optimal_high": 16,
        "clinical_name": "GGT (Gamma-Glutamyltransferase)",
        "interpretation": {
            "optimal": "GGT is in the optimal range — excellent hepatic and oxidative stress marker.",
            "normal": "GGT is within standard range but above optimal — sensitive marker of oxidative stress and early hepatic dysfunction.",
            "high": "GGT is elevated — alcohol intake, hepatic dysfunction, or significant oxidative stress.",
            "low": "GGT is very low — optimal."
        },
        "genomic_snps": ["rs1695", "rs1800566"],
        "genomic_note": {
            "direction": "worse",
            "context": "GGT is the most sensitive marker of both hepatic oxidative stress and glutathione system demand. Your GSTP1 variant reduces the activity of glutathione S-transferase pi, the enzyme that conjugates glutathione to reactive compounds for detoxification. Your NQO1 variant impairs another detoxification enzyme that prevents quinone compounds from redox cycling. Together, these variants simultaneously generate more oxidative stress through impaired detoxification and maintain lower glutathione reserves to neutralize it — GGT rises as the body attempts to compensate by increasing glutathione synthesis and turnover.",
            "implications": "GGT above 16 U/L is increasingly recognized as one of the strongest independent predictors of cardiovascular mortality, all-cause mortality, and insulin resistance — even within the 'normal' laboratory range. It is more sensitive than ALT or AST for detecting early hepatic oxidative stress. For your genotype, elevated GGT reflects a genuine systems-level deficit in antioxidant capacity with direct downstream effects on every organ system vulnerable to oxidative damage: heart, brain, liver, and kidney.",
            "actions": [
                "Your GGT target is below 16 U/L — the longevity-optimal range, not the standard 65 U/L clinical cutoff",
                "N-acetylcysteine (NAC) 600-1,200 mg/day is the most direct intervention — NAC is the rate-limiting substrate for glutathione synthesis and directly compensates for your GSTP1-driven glutathione consumption",
                "Lipoic acid 600 mg/day — recycles oxidized glutathione back to reduced (active) form and is itself a potent direct antioxidant; particularly effective alongside NQO1 variants",
                "Eliminate alcohol completely — alcohol consumes glutathione at an extraordinary rate and competes with NAC for cysteine substrate; even moderate alcohol causes GGT elevation disproportionate to liver damage in GSTP1/NQO1 variants",
                "Eat cruciferous vegetables daily (broccoli, Brussels sprouts) — sulforaphane activates Nrf2, the master regulator of antioxidant gene expression that compensates for your GSTP1 and NQO1 variant-driven deficits"
            ]
        }
    },
    "bilirubin_total": {
        "aliases": ["bilirubin total", "total bilirubin", "bilirubin, total"],
        "unit": "mg/dL", "category": "Hepatic",
        "standard_low": 0.1, "standard_high": 1.2,
        "optimal_low": 0.2, "optimal_high": 1.0,
        "clinical_name": "Total Bilirubin",
        "interpretation": {
            "optimal": "Total bilirubin is in the optimal range.",
            "normal": "Total bilirubin is within standard range.",
            "high": "Total bilirubin is elevated — evaluate for hemolysis, hepatic dysfunction, or biliary obstruction. Mildly elevated with normal liver enzymes suggests Gilbert's syndrome (UGT1A1 variant).",
            "low": "Total bilirubin is very low."
        },
        "genomic_snps": ["rs4148323", "rs8175347"],
        "genomic_note": {
            "direction": "neutral",
            "context": "UGT1A1 variants cause Gilbert's syndrome, a benign condition where the liver's bilirubin conjugation enzyme has reduced activity — typically 30-50% of normal. Indirect (unconjugated) bilirubin accumulates, resulting in mildly elevated total bilirubin (1.1-3.0 mg/dL) with otherwise completely normal liver function. This is not liver disease — it is a metabolic variant affecting only this one step of bilirubin processing. Gilbert's syndrome is present in approximately 8-10% of the population but is frequently misdiagnosed as liver disease.",
            "implications": "The counterintuitive finding: mildly elevated bilirubin from Gilbert's syndrome is associated with protective effects. Bilirubin is a potent endogenous antioxidant — indirect bilirubin at 1-3 mg/dL inhibits LDL oxidation, reduces cardiovascular disease risk, and has anti-inflammatory properties. Multiple large cohort studies have shown that Gilbert's syndrome carriers have significantly lower rates of cardiovascular disease, type 2 diabetes, and cancer than the general population. Your elevated bilirubin is likely an asset, not a liability.",
            "actions": [
                "Confirm the diagnosis: check direct (conjugated) bilirubin separately — Gilbert's syndrome causes only indirect bilirubin elevation; if direct bilirubin is also elevated, that requires further investigation",
                "No treatment is needed or recommended for Gilbert's syndrome — this is a benign variant that requires only awareness",
                "Inform your physicians and prescribers of your Gilbert's syndrome status — certain medications (irinotecan chemotherapy, indinavir HIV medication, atazanavir) are metabolized by UGT1A1 and can cause toxicity at standard doses in Gilbert's carriers",
                "Be aware that fasting, illness, alcohol, and physical stress can temporarily elevate bilirubin further in Gilbert's carriers — this is normal and self-resolving",
                "Consider this a genomic asset: your elevated bilirubin likely confers meaningful antioxidant protection that partially offsets other cardiovascular risk factors you carry"
            ]
        }
    },

    # =========================================================================
    # HEMATOLOGIC
    # =========================================================================
    "wbc": {
        "aliases": ["wbc", "white blood cells", "white blood cell count", "leukocytes"],
        "unit": "K/uL", "category": "Hematologic",
        "standard_low": 4.5, "standard_high": 11.0,
        "optimal_low": 4.5, "optimal_high": 7.0,
        "clinical_name": "WBC (White Blood Cell Count)",
        "interpretation": {
            "optimal": "WBC is in the optimal range.",
            "normal": "WBC is within standard range but above optimal — chronic low-grade inflammation or immune activation.",
            "high": "WBC is elevated — infection, inflammation, or hematologic evaluation warranted.",
            "low": "WBC is low — bone marrow suppression, autoimmune, or medication effect."
        },
        "genomic_snps": ["rs1800629", "rs1800795"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your TNF-alpha (rs1800629) and IL-6 (rs1800795) pro-inflammatory variants create a chronically elevated baseline immune activation state. WBC — particularly monocytes and neutrophils — are persistently recruited and maintained at higher levels in the context of chronic low-grade inflammation. A WBC of 6.0-8.0 K/uL, while technically within standard range, may reflect ongoing TNF-alpha/IL-6 driven immune mobilization rather than a healthy resting immune state.",
            "implications": "Chronically elevated WBC above 7.0 K/uL — even within 'normal' — is independently associated with cardiovascular mortality, insulin resistance, and accelerated biological aging in longitudinal studies. The immune cells themselves, when chronically activated, release additional inflammatory mediators (myeloperoxidase, reactive oxygen species, proteases) that damage arterial walls, impair insulin signaling, and drive mitochondrial dysfunction. For your genotype, WBC above 7.0 K/uL should be interpreted as a signal of uncontrolled underlying inflammation, not a benign finding.",
            "actions": [
                "Your WBC target is below 6.5 K/uL — track this longitudinally alongside hs-CRP as dual inflammation markers",
                "Gut health optimization is your highest-leverage intervention for WBC reduction — intestinal permeability is the primary chronic driver of systemic immune activation in pro-inflammatory genotypes; L-glutamine 5g/day and butyrate support gut barrier integrity",
                "Omega-3 supplementation (EPA+DHA 3-4g/day) directly reduces neutrophil and monocyte inflammatory activity, targeting the cells that make up the elevated WBC fraction",
                "Identify and eliminate chronic immune triggers: food intolerances, dental infections, sleep apnea, and gut dysbiosis are all common and underdiagnosed drivers of chronically elevated WBC",
                "Retest WBC with hs-CRP and IL-6 simultaneously — if all three are elevated, the pattern confirms genetically amplified chronic inflammation requiring multi-modal intervention"
            ]
        }
    },
    "rbc": {
        "aliases": ["rbc", "red blood cells", "red blood cell count", "erythrocytes"],
        "unit": "M/uL", "category": "Hematologic",
        "standard_low": 3.9, "standard_high": 5.7,
        "optimal_low": 4.0, "optimal_high": 5.2,
        "clinical_name": "RBC (Red Blood Cell Count)",
        "interpretation": {
            "optimal": "RBC is in the optimal range.",
            "normal": "RBC is within standard range.",
            "high": "RBC is elevated — polycythemia, dehydration, or altitude adaptation.",
            "low": "RBC is low — anemia evaluation warranted."
        },
        "genomic_snps": [],
        "genomic_note": ""
    },
    "hemoglobin": {
        "aliases": ["hemoglobin", "hgb", "hb", "haemoglobin"],
        "unit": "g/dL", "category": "Hematologic",
        "standard_low": 11.5, "standard_high": 17.5,
        "optimal_low": 13.5, "optimal_high": 17.0,
        "clinical_name": "Hemoglobin",
        "interpretation": {
            "optimal": "Hemoglobin is in the optimal range.",
            "normal": "Hemoglobin is within standard range.",
            "high": "Hemoglobin is elevated — polycythemia or dehydration.",
            "low": "Hemoglobin is low — anemia. Evaluate iron, B12, folate, and hemolysis."
        },
        "genomic_snps": ["rs602662", "rs1801133"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your FUT2 non-secretor variant reduces intestinal B12 absorption by up to 50%, while your MTHFR C677T variant impairs folate metabolism — both of which are critical cofactors for DNA synthesis in developing red blood cells. Both nutrients are required for proper erythropoiesis, and deficiencies in either cause macrocytic anemia with low hemoglobin. Standard lab values may show hemoglobin in the 'low normal' range while the underlying nutritional deficiency responsible is not yet severe enough to flag on the same blood draw.",
            "implications": "Progressive B12 or folate deficiency first depletes bone marrow precursor cells before dropping peripheral hemoglobin — meaning you can have normal hemoglobin while your bone marrow is already under nutritional stress. Additionally, the neurological effects of B12 deficiency (peripheral neuropathy, cognitive changes) precede hematological changes, so waiting for anemia to confirm deficiency means neurological damage has already begun. Low-normal hemoglobin with your variant profile warrants B12 and folate evaluation regardless of whether hemoglobin has crossed the anemia threshold.",
            "actions": [
                "Test methylmalonic acid (MMA) and homocysteine alongside hemoglobin — elevated MMA confirms functional B12 deficiency; elevated homocysteine with low-normal hemoglobin confirms methylation deficiency-driven erythropoiesis stress",
                "Switch to sublingual methylcobalamin B12 (1,000-2,000 mcg/day) — bypasses your FUT2 gut absorption deficit entirely through salivary-gland mediated absorption, independent of intrinsic factor",
                "Switch to methylfolate (L-5-MTHF 400-800 mcg/day) — the pre-converted active form that bypasses your MTHFR bottleneck for DNA synthesis in developing red blood cells",
                "Target hemoglobin in the upper optimal range (14-16 g/dL for men, 13-15 g/dL for women) — not merely above the anemia threshold",
                "If hemoglobin is below 13.5 g/dL, also test serum ferritin and reticulocyte count — iron deficiency and nutritional deficiency can coexist, and distinguishing macrocytic (B12/folate) from microcytic (iron) anemia guides the correct intervention"
            ]
        }
    },
    "platelets": {
        "aliases": ["platelets", "platelet count", "plt"],
        "unit": "K/uL", "category": "Hematologic",
        "standard_low": 150, "standard_high": 400,
        "optimal_low": 175, "optimal_high": 300,
        "clinical_name": "Platelets",
        "interpretation": {
            "optimal": "Platelet count is in the optimal range.",
            "normal": "Platelet count is within standard range.",
            "high": "Platelet count is elevated — reactive thrombocytosis or myeloproliferative evaluation.",
            "low": "Platelet count is low — thrombocytopenia evaluation warranted."
        },
        "genomic_snps": ["rs5918", "rs1126643"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your ITGB3 (PlA2 allele, rs5918) and ITGA2 variants affect platelet surface receptors — specifically integrin αIIbβ3 (the primary platelet fibrinogen receptor) and integrin α2β1 (the platelet collagen receptor). These receptor variants alter platelet activation thresholds and aggregation speed. The result is not a change in platelet count but a change in platelet reactivity — your platelets activate more aggressively at the same level of vascular injury or shear stress that would cause minimal activation in individuals without these variants.",
            "implications": "Hyperreactive platelets with these variants increase risk of arterial thrombosis — the type of sudden blood clotting that causes myocardial infarction and ischemic stroke — independently of platelet count. The ITGB3 PlA2 variant has been associated with aspirin resistance in some studies, meaning standard aspirin therapy may be less effective for thrombotic risk reduction in your case. This platelet reactivity also interacts with elevated Lp(a) (if present) and systemic inflammation to compound thrombotic risk beyond what any single factor suggests.",
            "actions": [
                "Discuss aspirin therapy with your physician in the context of your platelet reactivity variants — if aspirin therapy is indicated, platelet aggregation testing can assess whether you are a responder",
                "Omega-3 supplementation (EPA+DHA 3-4g/day) reduces platelet aggregation through thromboxane A2 inhibition, targeting your specific platelet receptor hypersensitivity mechanism",
                "Avoid combining NSAIDs (ibuprofen, naproxen) with aspirin — NSAIDs competitively inhibit aspirin's COX-1 platelet effect, which matters more for your genotype given potential aspirin resistance",
                "Ginger (2g/day) and nattokinase have evidence for reducing platelet aggregation through mechanisms complementary to omega-3; ginger inhibits thromboxane synthase, nattokinase has fibrinolytic activity",
                "Monitor cardiovascular risk factors (blood pressure, lipids, blood glucose) with extra vigilance — with hyperreactive platelets, even modest elevations in traditional risk factors carry amplified thrombotic consequences"
            ]
        }
    },

    # =========================================================================
    # CARDIOVASCULAR ADVANCED
    # =========================================================================
    "omega3_index": {
        "aliases": ["omega-3 index", "omega 3 index", "epa+dha", "omega3 index"],
        "unit": "%", "category": "Cardiovascular",
        "standard_low": 4, "standard_high": 20,
        "optimal_low": 8, "optimal_high": 12,
        "clinical_name": "Omega-3 Index",
        "interpretation": {
            "optimal": "Omega-3 Index is in the optimal range — excellent cardiovascular and anti-inflammatory status.",
            "normal": "Omega-3 Index is within standard range but below optimal — supplementation warranted.",
            "high": "Omega-3 Index is above optimal — very high dose supplementation; monitor for anticoagulation effects.",
            "low": "Omega-3 Index is low — significantly elevated cardiovascular and inflammatory risk."
        },
        "genomic_snps": ["rs174537", "rs1535", "rs174575"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your FADS1 and FADS2 variants severely impair the conversion of short-chain ALA (alpha-linolenic acid, the omega-3 found in flaxseed, chia, and walnuts) into the long-chain EPA and DHA that your body uses for anti-inflammatory signaling, brain structure, and cardiovascular protection. In individuals without FADS variants, approximately 5-10% of ALA converts to EPA and under 1% to DHA. With FADS1/2 variants, these conversion rates drop to near zero — making you almost completely dependent on direct dietary sources of EPA and DHA.",
            "implications": "If your diet relies on plant-based omega-3 sources (ALA) as your primary omega-3 intake, you are functionally omega-3 deficient regardless of your total ALA consumption. Omega-3 Index below 8% is associated with significantly elevated cardiovascular mortality risk, inflammatory disease risk, and cognitive aging. For your genotype, even a moderate Omega-3 Index achieved through ALA consumption provides near-zero actual EPA/DHA benefit — you need direct preformed EPA+DHA from marine or algae sources.",
            "actions": [
                "Your Omega-3 Index target is 10-12% — higher than the general population target of 8%, because your impaired conversion means you need more EPA/DHA input to achieve the same tissue incorporation",
                "Switch from plant-based omega-3 sources as your primary intake to marine EPA+DHA — triglyceride-form fish oil or algae-derived EPA+DHA are your only effective options",
                "Dose 3-4g EPA+DHA per day — higher than the standard 1g recommendation, calibrated to your impaired baseline incorporation rate",
                "Take omega-3 with your largest fat-containing meal — absorption increases 40-60% when taken with dietary fat",
                "Retest Omega-3 Index after 4-6 months of consistent supplementation — if it remains below 8% despite 3g+ EPA+DHA daily, switch to phospholipid-form omega-3 (krill oil) which has superior tissue incorporation efficiency"
            ]
        }
    },
    "myeloperoxidase": {
        "aliases": ["myeloperoxidase", "mpo"],
        "unit": "pmol/L", "category": "Cardiovascular",
        "standard_low": 0, "standard_high": 470,
        "optimal_low": 0, "optimal_high": 200,
        "clinical_name": "Myeloperoxidase (MPO)",
        "interpretation": {
            "optimal": "MPO is in the optimal range — low neutrophil activation and vascular oxidative stress.",
            "normal": "MPO is within standard range but elevated relative to optimal.",
            "high": "MPO is elevated — significant vascular inflammation and oxidative LDL modification.",
            "low": "MPO is very low — favorable."
        },
        "genomic_snps": ["rs662", "rs854560"],
        "genomic_note": {
            "direction": "worse",
            "context": "Your PON1 variants reduce the activity of paraoxonase-1, the primary enzyme HDL particles carry to neutralize oxidized LDL and lipid peroxides in arterial walls. Myeloperoxidase is released by neutrophils and monocytes at sites of vascular inflammation — it oxidizes LDL cholesterol, generating oxidized LDL (oxLDL) that foam cells ingest to form plaques. The combination of elevated MPO output (from your pro-inflammatory variants) and impaired PON1 clearance of oxidized lipids creates a self-amplifying cycle of vascular oxidative damage.",
            "implications": "Elevated MPO with reduced PON1 activity means your atherosclerotic plaque formation rate is driven not just by LDL quantity but by LDL oxidation rate and clearance failure — a mechanism completely invisible to standard lipid testing. You could have a 'normal' LDL-C while your arterial walls experience significantly accelerated plaque deposition through the MPO-oxLDL pathway. MPO above 200 pmol/L is associated with 2-3x increased near-term cardiovascular event risk in individuals with existing coronary artery disease.",
            "actions": [
                "Omega-3 supplementation (EPA+DHA 3-4g/day) is the most effective intervention for reducing MPO-driven vascular oxidation — EPA directly inhibits myeloperoxidase activity in neutrophils",
                "Pomegranate extract 1,000mg/day — contains punicalagins and ellagic acid that have demonstrated ability to increase PON1 enzyme activity by 20-30% in clinical studies, directly compensating for your PON1 variant",
                "Extra virgin olive oil daily (2-4 tablespoons) — polyphenols in EVOO are cofactors for PON1 activity and independently reduce LDL oxidation",
                "Minimize cigarette smoke and air pollutant exposure — these are the strongest environmental activators of neutrophil MPO release and directly amplify your genetic PON1 vulnerability",
                "Request an oxLDL test if available — oxidized LDL directly measures the downstream product of your MPO/PON1 imbalance and is a more specific cardiovascular risk marker for your genotype than total LDL-C"
            ]
        }
    },
}


# =============================================================================
# BIOMARKER EXTRACTION ENGINE
# =============================================================================

def extract_biomarkers_with_ai(pdf_path: str, bedrock_client) -> dict:
    """
    Use Nova Pro to extract biomarkers from a lab PDF.
    Returns dict of {biomarker_name: {"value": float, "unit": str, "raw_text": str}}
    """
    import base64

    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode("utf-8")

    extraction_prompt = """Extract ALL biomarker values from this lab report. 
    Return ONLY a JSON object with this exact format:
    {
        "biomarker_name": {"value": numeric_value, "unit": "unit_string", "reference_range": "lab_reference_as_string"},
        ...
    }
    
    Rules:
    - Use lowercase biomarker names with underscores (e.g. "total_cholesterol", "hdl", "tsh")
    - Include every biomarker with a numeric result
    - Convert all values to numeric (no ranges, no inequalities)
    - If a value has < or > prefix, use the numeric part
    - Include the lab's reference range as a string exactly as printed
    - Return ONLY the JSON object, no other text
    
    Common name mappings to use:
    - "Glucose" -> "glucose"
    - "HbA1c" or "Hemoglobin A1c" -> "hba1c"  
    - "Total Cholesterol" -> "total_cholesterol"
    - "LDL" or "LDL Cholesterol" -> "ldl"
    - "HDL" or "HDL Cholesterol" -> "hdl"
    - "Triglycerides" -> "triglycerides"
    - "TSH" -> "tsh"
    - "Free T4" -> "free_t4"
    - "Free T3" -> "free_t3"
    - "Vitamin D" or "25-OH Vitamin D" -> "vitamin_d"
    - "Vitamin B12" -> "vitamin_b12"
    - "hs-CRP" or "CRP" -> "hscrp"
    - "Homocysteine" -> "homocysteine"
    - "Ferritin" -> "ferritin"
    - "ALT" or "SGPT" -> "alt"
    - "AST" or "SGOT" -> "ast"
    - "GGT" -> "ggtp"
    - "Creatinine" -> "creatinine"
    - "eGFR" -> "egfr"
    - "BUN" -> "bun"
    - "Testosterone" -> "testosterone_total"
    - "Free Testosterone" -> "testosterone_free"
    - "SHBG" -> "shbg"
    - "Estradiol" -> "estradiol"
    - "DHEA-S" -> "dhea_s"
    - "Cortisol" -> "cortisol"
    - "IGF-1" -> "igf1"
    - "ApoB" or "Apolipoprotein B" -> "apob"
    - "Lp(a)" -> "lpa"
    - "WBC" -> "wbc"
    - "RBC" -> "rbc"
    - "Hemoglobin" -> "hemoglobin"
    - "Platelets" -> "platelets"
    - "Uric Acid" -> "uric_acid"
    - "Magnesium" -> "magnesium"
    - "Zinc" -> "zinc"
    - "Folate" -> "folate"
    - "Total Bilirubin" -> "bilirubin_total"
    - "Total Protein" -> "total_protein"
    - "Albumin" -> "albumin"
    - "Calcium" -> "calcium"
    - "Sodium" -> "sodium"
    - "Potassium" -> "potassium"
    - "CO2" or "Bicarbonate" -> "co2"
    - "Chloride" -> "chloride"
    """

    body = json.dumps({
        "system": [{"text": "You are a medical data extraction system. Extract biomarker values from lab reports and return ONLY valid JSON. No markdown, no explanation."}],
        "messages": [{
            "role": "user",
            "content": [
                {
                    "document": {
                        "format": "pdf",
                        "name": "lab_report",
                        "source": {"bytes": pdf_data}
                    }
                },
                {"text": extraction_prompt}
            ]
        }],
        "inferenceConfig": {"maxTokens": 3000, "temperature": 0.1}
    })

    response = bedrock_client.invoke_model(
        modelId="us.amazon.nova-pro-v1:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    text = result["output"]["message"]["content"][0]["text"].strip()

    # Clean up any markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        extracted = json.loads(text)
        return extracted
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _build_alias_patterns():
    alias_patterns = []
    seen = set()

    for db_key, entry in BIOMARKER_DB.items():
        candidates = [db_key.replace("_", " "), entry.get("clinical_name", "")]
        candidates.extend(entry.get("aliases", []))

        for alias in candidates:
            normalized = _normalize_alias(alias)
            if not normalized or (db_key, normalized) in seen:
                continue
            seen.add((db_key, normalized))

            parts = [re.escape(part) for part in normalized.split() if part]
            if not parts:
                continue

            alias_patterns.append({
                "db_key": db_key,
                "unit": entry.get("unit", ""),
                "length": len(normalized),
                "pattern": re.compile(
                    r"^" + r"[\s,()/:%-]*".join(parts) + r"[\s,()/:%-]*",
                    re.IGNORECASE,
                ),
            })

    alias_patterns.sort(key=lambda item: item["length"], reverse=True)
    return alias_patterns


ALIAS_PATTERNS = _build_alias_patterns()


def extract_biomarkers_locally(pdf_path: str) -> dict:
    """
    Deterministic fallback parser for text-based PDFs.
    Works well on digitally generated lab reports and avoids total failure when
    AI extraction is unavailable.
    """
    if PdfReader is None:
        return {}

    try:
        reader = PdfReader(pdf_path)
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return {}

    if not raw_text.strip():
        return {}

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    extracted = {}

    for line in lines:
        for item in ALIAS_PATTERNS:
            match = item["pattern"].match(line)
            if not match:
                continue

            remainder = line[match.end():].strip()
            value_match = re.search(r"-?\d+(?:\.\d+)?", remainder)
            if not value_match:
                break

            value = value_match.group()
            tail = remainder[value_match.end():].strip()
            unit = item["unit"]
            reference_range = "N/A"

            if unit and unit in tail:
                reference_range = tail.split(unit, 1)[0].strip() or "N/A"
            elif tail:
                range_match = re.search(
                    r"((?:<|>|<=|>=)\s*-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?)",
                    tail,
                )
                if range_match:
                    reference_range = range_match.group(1).strip()

            extracted[item["db_key"]] = {
                "value": value,
                "unit": unit,
                "reference_range": reference_range,
            }
            break

    return extracted


def load_extracted_biomarkers_sidecar(pdf_path: str) -> dict:
    """
    Load a local JSON sidecar for offline/demo parsing.

    Expected shape:
    {
        "biomarker_key": {"value": 123, "unit": "mg/dL", "reference_range": "70-99"}
    }
    """
    candidate_paths = [
        Path(pdf_path).with_suffix(".json"),
        Path(__file__).resolve().parent / Path(pdf_path).with_suffix(".json").name,
    ]

    for sidecar_path in candidate_paths:
        if not sidecar_path.exists():
            continue

        with sidecar_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    return {}


def evaluate_biomarker(name: str, value: float, snp_results: dict = None) -> dict:
    """
    Evaluate a biomarker against standard and optimal ranges.
    Cross-reference with SNP data if provided.
    Returns evaluation dict.
    """
    # Find matching biomarker in DB
    db_entry = None
    db_key = None

    # Direct match
    if name.lower() in BIOMARKER_DB:
        db_key = name.lower()
        db_entry = BIOMARKER_DB[db_key]
    else:
        # Search aliases
        for key, entry in BIOMARKER_DB.items():
            if name.lower() in [a.lower() for a in entry.get("aliases", [])]:
                db_key = key
                db_entry = entry
                break

    if not db_entry:
        return {
            "name": name,
            "value": value,
            "status": "unknown",
            "in_db": False
        }

    std_low = db_entry["standard_low"]
    std_high = db_entry["standard_high"]
    opt_low = db_entry["optimal_low"]
    opt_high = db_entry["optimal_high"]

    # Determine standard status
    if value < std_low:
        std_status = "below_standard"
    elif value > std_high:
        std_status = "above_standard"
    else:
        std_status = "within_standard"

    # Determine optimal status
    if value < opt_low:
        opt_status = "below_optimal"
    elif value > opt_high:
        opt_status = "above_optimal"
    else:
        opt_status = "optimal"

    # Overall flag
    if std_status in ["below_standard", "above_standard"]:
        flag = "abnormal"
    elif opt_status in ["below_optimal", "above_optimal"]:
        flag = "suboptimal"
    else:
        flag = "optimal"

    # Check genomic context
    genomic_context = None
    genomic_direction = None
    genomic_implications = None
    genomic_actions = None
    genomic_recontextualized = False

    if snp_results and db_entry.get("genomic_snps") and db_entry.get("genomic_note"):
        relevant_snps = db_entry["genomic_snps"]
        found_risk_snps = []

        for category_snps in snp_results.values():
            for snp in category_snps:
                if snp["rsid"].lower() in [r.lower() for r in relevant_snps]:
                    if snp["zygosity"] in ["heterozygous", "homozygous_risk"]:
                        found_risk_snps.append(snp)

        if found_risk_snps:
            note = db_entry["genomic_note"]
            if isinstance(note, dict):
                genomic_context = note.get("context", "")
                genomic_direction = note.get("direction", "worse")
                genomic_implications = note.get("implications", "")
                genomic_actions = note.get("actions", [])
            else:
                # Legacy flat string format
                genomic_context = note
                genomic_direction = "worse"
                genomic_implications = ""
                genomic_actions = []

            genomic_recontextualized = True
            if flag in ["optimal", "suboptimal"]:
                flag = "genomic_recontextualized"

    interpretation_key = "optimal" if flag == "optimal" else (
        "high" if std_status == "above_standard" else (
        "low" if std_status == "below_standard" else "normal"
    ))

    return {
        "name": db_entry["clinical_name"],
        "db_key": db_key,
        "value": value,
        "unit": db_entry["unit"],
        "category": db_entry["category"],
        "standard_range": f"{std_low} - {std_high}",
        "optimal_range": f"{opt_low} - {opt_high}",
        "standard_status": std_status,
        "optimal_status": opt_status,
        "flag": flag,
        "interpretation": db_entry["interpretation"].get(interpretation_key, ""),
        "genomic_context": genomic_context,
        "genomic_direction": genomic_direction,
        "genomic_implications": genomic_implications,
        "genomic_actions": genomic_actions,
        "genomic_recontextualized": genomic_recontextualized,
        "in_db": True
    }


def process_blood_work(pdf_path: str, snp_results: dict = None) -> dict:
    """
    Full blood work processing pipeline.
    Returns structured results ready for report generation.
    """
    print(f"Extracting biomarkers from: {pdf_path}")
    raw_extracted = extract_biomarkers_locally(pdf_path)
    if raw_extracted:
        print(f"  Local text extraction found {len(raw_extracted)} biomarkers")

    if len(raw_extracted) < 8:
        bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
        try:
            ai_extracted = extract_biomarkers_with_ai(pdf_path, bedrock)
            if len(ai_extracted) > len(raw_extracted):
                raw_extracted = ai_extracted
        except Exception as exc:
            print(f"  AI extraction failed: {exc}")

    if not raw_extracted:
        raw_extracted = load_extracted_biomarkers_sidecar(pdf_path)

    if not raw_extracted:
        raise RuntimeError(
            "Could not extract biomarkers from the PDF. "
            "Try a text-based lab PDF, or use a cleaner export with selectable text."
        )

    print(f"  Extracted {len(raw_extracted)} biomarkers")

    evaluated = {}
    by_category = {}
    flagged = {"abnormal": [], "suboptimal": [], "genomic_recontextualized": [], "optimal": []}

    for name, data in raw_extracted.items():
        value = data.get("value")
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        result = evaluate_biomarker(name, value, snp_results)
        result["reference_range_lab"] = data.get("reference_range", "N/A")
        evaluated[name] = result

        if result["in_db"]:
            cat = result["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
            flagged[result["flag"]].append(result)
        else:
            # Still track unknown biomarkers
            if "Other" not in by_category:
                by_category["Other"] = []
            by_category["Other"].append(result)

    total = len(evaluated)
    in_db = sum(1 for r in evaluated.values() if r["in_db"])
    print(f"  Evaluated {in_db} known biomarkers, {total - in_db} additional values")
    print(f"  Flags: {len(flagged['abnormal'])} abnormal, {len(flagged['suboptimal'])} suboptimal, "
          f"{len(flagged['genomic_recontextualized'])} genomically recontextualized, "
          f"{len(flagged['optimal'])} optimal")

    return {
        "raw_extracted": raw_extracted,
        "evaluated": evaluated,
        "by_category": by_category,
        "flagged": flagged,
        "total_biomarkers": total,
        "in_db_count": in_db,
        "has_genomic_context": snp_results is not None
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 blood_parser.py <lab_report.pdf>")
        sys.exit(1)

    results = process_blood_work(sys.argv[1])
    print(json.dumps(results, indent=2))
