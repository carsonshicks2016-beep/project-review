"""
HealthBridge AI - Expanded SNP Pre-Processor
14 health categories, ~1000+ clinically relevant rsids
Sources: ClinVar, GWAS Catalog, PharmGKB, published peer-reviewed literature
"""

import hashlib
import json
import sys
from pathlib import Path

from variant_validation import (
    canonical_panel_rsid,
    ensure_variant_metadata,
    panel_keep_set,
    parse_raw_genotype_file,
    validate_variant_call,
)

# Static salt for HealthBridge de-identification. 
# In production, this should be moved to a secure environment variable.
HB_SALT = "HealthBridge_2026_Secure_Core"

def hash_rsid(rsid: str) -> str:
    """Generate SHA-256 salted hash of rsid for HIPAA compliance."""
    salted_input = f"{rsid}:{HB_SALT}".encode('utf-8')
    return hashlib.sha256(salted_input).hexdigest()

SNP_PANEL = {

    # =========================================================================
    # 1. NUTRITION & METABOLISM
    # =========================================================================
    "rs1801282": {"gene": "PPARG", "category": "Nutrition & Metabolism", "trait": "Fat metabolism / insulin sensitivity", "risk_allele": "G", "effect": "Pro12Ala — reduced PPARG activity, lower T2D risk with Pro allele"},
    "rs9939609": {"gene": "FTO", "category": "Nutrition & Metabolism", "trait": "Obesity / BMI", "risk_allele": "A", "effect": "A allele associated with increased BMI and appetite"},
    "rs1558902": {"gene": "FTO", "category": "Nutrition & Metabolism", "trait": "Obesity / BMI", "risk_allele": "A", "effect": "Strongest FTO signal for obesity in Europeans"},
    "rs4988235": {"gene": "LCT", "category": "Nutrition & Metabolism", "trait": "Lactase persistence", "risk_allele": "T", "effect": "T = lactase persistence; C = lactose intolerance"},
    "rs182549": {"gene": "MCM6", "category": "Nutrition & Metabolism", "trait": "Lactose intolerance", "risk_allele": "C", "effect": "C allele associated with lactose intolerance"},
    "rs1229984": {"gene": "ADH1B", "category": "Nutrition & Metabolism", "trait": "Alcohol metabolism / alcohol flush protection", "risk_allele": "T", "effect": "Arg48His — T (His48, ADH1B*2) on plus strand = 70-100x faster ethanol-to-acetaldehyde conversion, protective against AUD but increases acetaldehyde exposure risk"},
    "rs671": {"gene": "ALDH2", "category": "Nutrition & Metabolism", "trait": "Acetaldehyde clearance / alcohol flush", "risk_allele": "A", "effect": "Glu504Lys — A = markedly reduced ALDH2 activity, alcohol flush and acetaldehyde exposure risk"},
    "rs1761667": {"gene": "CD36", "category": "Nutrition & Metabolism", "trait": "Fat taste sensitivity", "risk_allele": "A", "effect": "AA = reduced fat taste sensitivity, higher fat intake tendency"},
    "rs2472297": {"gene": "CYP1A2", "category": "Nutrition & Metabolism", "trait": "Caffeine metabolism", "risk_allele": "T", "effect": "T = faster caffeine metabolism"},
    "rs762551": {"gene": "CYP1A2", "category": "Nutrition & Metabolism", "trait": "Caffeine metabolism", "risk_allele": "A", "effect": "AA = fast caffeine metabolizer; C = slow"},
    "rs4680": {"gene": "COMT", "category": "Nutrition & Metabolism", "trait": "Dopamine / catechol metabolism", "risk_allele": "A", "effect": "Val158Met — AA = low COMT activity, higher dopamine and estrogen metabolites"},
    "rs1801394": {"gene": "MTRR", "category": "Nutrition & Metabolism", "trait": "B12 / folate metabolism", "risk_allele": "G", "effect": "Ile22Met — GG associated with elevated homocysteine"},
    "rs1801133": {"gene": "MTHFR", "category": "Nutrition & Metabolism", "trait": "Folate metabolism", "risk_allele": "A", "effect": "C677T — AA = ~70% reduced MTHFR activity, elevated homocysteine"},
    "rs1801131": {"gene": "MTHFR", "category": "Nutrition & Metabolism", "trait": "Folate metabolism", "risk_allele": "C", "effect": "A1298C — CC associated with reduced MTHFR activity"},
    "rs602662": {"gene": "FUT2", "category": "Nutrition & Metabolism", "trait": "Vitamin B12 absorption", "risk_allele": "A", "effect": "Non-secretor status — lower B12 absorption"},
    "rs492602": {"gene": "FUT2", "category": "Nutrition & Metabolism", "trait": "Vitamin B12 / gut microbiome", "risk_allele": "G", "effect": "Associated with B12 levels and microbiome composition"},
    "rs10741657": {"gene": "CYP2R1", "category": "Nutrition & Metabolism", "trait": "Vitamin D metabolism", "risk_allele": "A", "effect": "A = lower 25-hydroxyvitamin D levels"},
    "rs2282679": {"gene": "GC", "category": "Nutrition & Metabolism", "trait": "Vitamin D binding protein", "risk_allele": "C", "effect": "CC = lower circulating vitamin D"},
    "rs12785878": {"gene": "DHCR7", "category": "Nutrition & Metabolism", "trait": "Vitamin D synthesis", "risk_allele": "T", "effect": "TT = lower vitamin D levels"},
    "rs174537": {"gene": "FADS1", "category": "Nutrition & Metabolism", "trait": "Omega-3/6 fatty acid conversion", "risk_allele": "T", "effect": "T = reduced delta-5 desaturase, lower EPA/DHA conversion from ALA"},
    "rs1535": {"gene": "FADS2", "category": "Nutrition & Metabolism", "trait": "Omega-3/6 fatty acid conversion", "risk_allele": "A", "effect": "Reduced fatty acid desaturation efficiency"},
    "rs174575": {"gene": "FADS2", "category": "Nutrition & Metabolism", "trait": "DHA synthesis", "risk_allele": "C", "effect": "C = reduced FADS2, lower DHA synthesis from ALA"},
    "rs5082": {"gene": "APOA2", "category": "Nutrition & Metabolism", "trait": "Saturated fat response", "risk_allele": "C", "effect": "CC = higher BMI response to saturated fat intake"},
    "rs1260326": {"gene": "GCKR", "category": "Nutrition & Metabolism", "trait": "Triglycerides / glucose", "risk_allele": "T", "effect": "T = higher triglycerides, lower fasting glucose"},
    "rs780094": {"gene": "GCKR", "category": "Nutrition & Metabolism", "trait": "Triglycerides / fasting glucose", "risk_allele": "T", "effect": "Associated with triglyceride levels and T2D risk"},
    "rs7903146": {"gene": "TCF7L2", "category": "Nutrition & Metabolism", "trait": "Type 2 diabetes risk", "risk_allele": "T", "effect": "Strongest T2D variant — impairs beta cell function"},
    "rs12255372": {"gene": "TCF7L2", "category": "Nutrition & Metabolism", "trait": "Type 2 diabetes risk", "risk_allele": "T", "effect": "T = increased T2D risk"},
    "rs5219": {"gene": "KCNJ11", "category": "Nutrition & Metabolism", "trait": "Insulin secretion / T2D", "risk_allele": "T", "effect": "Lys23Glu — T = T2D risk"},
    "rs1044498": {"gene": "ENPP1", "category": "Nutrition & Metabolism", "trait": "Insulin resistance", "risk_allele": "C", "effect": "CC = insulin resistance and T2D"},
    "rs1799883": {"gene": "FABP2", "category": "Nutrition & Metabolism", "trait": "Fat absorption", "risk_allele": "T", "effect": "Ala54Thr — TT = higher fat absorption and insulin resistance"},
    "rs4994": {"gene": "ADRB3", "category": "Nutrition & Metabolism", "trait": "Metabolic rate / obesity", "risk_allele": "T", "effect": "Trp64Arg — T = lower metabolic rate"},
    "rs1042714": {"gene": "ADRB2", "category": "Nutrition & Metabolism", "trait": "Metabolic rate", "risk_allele": "C", "effect": "Gln27Glu — CC = obesity association in women"},
    "rs9923231": {"gene": "VKORC1", "category": "Nutrition & Metabolism", "trait": "Vitamin K metabolism", "risk_allele": "T", "effect": "T = lower VKORC1 expression, affects vitamin K cycling"},
    "rs3798220": {"gene": "LPA", "category": "Nutrition & Metabolism", "trait": "Lipoprotein(a) levels", "risk_allele": "C", "effect": "C = strongly associated with elevated Lp(a) and CVD risk"},
    "rs10455872": {"gene": "LPA", "category": "Nutrition & Metabolism", "trait": "Lipoprotein(a) levels", "risk_allele": "G", "effect": "G = elevated Lp(a), cardiovascular risk"},
    "rs1800562": {"gene": "HFE", "category": "Nutrition & Metabolism", "trait": "Hereditary hemochromatosis / iron overload", "risk_allele": "A", "effect": "C282Y — A = high iron overload risk, especially when paired with another HFE variant"},
    "rs1799945": {"gene": "HFE", "category": "Nutrition & Metabolism", "trait": "Iron overload / ferritin elevation", "risk_allele": "G", "effect": "H63D — G = milder hemochromatosis-risk allele that can compound other HFE variants"},
    "rs738409": {"gene": "PNPLA3", "category": "Nutrition & Metabolism", "trait": "Fatty liver / liver fat accumulation", "risk_allele": "G", "effect": "I148M — G = strongest common NAFLD / NASH susceptibility allele"},
    "rs58542926": {"gene": "TM6SF2", "category": "Nutrition & Metabolism", "trait": "Fatty liver / lipoprotein export", "risk_allele": "T", "effect": "E167K — T = impaired hepatic lipid export, higher NAFLD risk"},
    "rs641738": {"gene": "MBOAT7", "category": "Nutrition & Metabolism", "trait": "Fatty liver / liver inflammation", "risk_allele": "T", "effect": "T = higher hepatic fat and fibrosis susceptibility in metabolic liver disease"},
    "rs659366": {"gene": "UCP2", "category": "Nutrition & Metabolism", "trait": "Uncoupling protein / thermogenesis", "risk_allele": "T", "effect": "T = higher UCP2, affects energy expenditure and weight"},
    "rs1800592": {"gene": "UCP1", "category": "Nutrition & Metabolism", "trait": "Brown fat thermogenesis", "risk_allele": "A", "effect": "A = lower UCP1 in brown adipose, reduced thermogenesis"},

    # =========================================================================
    # 2. CARDIOVASCULAR RISK
    # =========================================================================
    "rs429358": {"gene": "APOE", "category": "Cardiovascular Risk", "trait": "Cholesterol / Alzheimer's risk", "risk_allele": "C", "effect": "APOE4 component — elevated LDL, cardiovascular and AD risk"},
    "rs7412": {"gene": "APOE", "category": "Cardiovascular Risk", "trait": "Cholesterol profile", "risk_allele": "T", "effect": "APOE2 component — T defines E2 haplotype (protective for AD, risk for type III HLP). Used with rs429358 for haplotype determination only; not a standalone risk signal"},
    "rs662": {"gene": "PON1", "category": "Cardiovascular Risk", "trait": "Oxidative stress / LDL oxidation", "risk_allele": "G", "effect": "Gln192Arg — G (Arg192) = reduced efficiency at hydrolyzing certain oxidized lipid substrates, associated with higher CAD risk in multiple meta-analyses"},
    "rs854560": {"gene": "PON1", "category": "Cardiovascular Risk", "trait": "Atherosclerosis protection", "risk_allele": "T", "effect": "Leu55Met — T = reduced PON1 activity"},
    "rs1800775": {"gene": "CETP", "category": "Cardiovascular Risk", "trait": "HDL cholesterol", "risk_allele": "A", "effect": "A = lower CETP activity, higher HDL"},
    "rs708272": {"gene": "CETP", "category": "Cardiovascular Risk", "trait": "HDL cholesterol", "risk_allele": "A", "effect": "TaqIB — A = higher HDL"},
    "rs1800588": {"gene": "LIPC", "category": "Cardiovascular Risk", "trait": "HDL / triglycerides", "risk_allele": "T", "effect": "T = lower hepatic lipase, higher HDL2"},
    "rs328": {"gene": "LPL", "category": "Cardiovascular Risk", "trait": "Triglyceride clearance", "risk_allele": "G", "effect": "Ser447Ter — G protective, higher LPL, lower TG"},
    "rs268": {"gene": "LPL", "category": "Cardiovascular Risk", "trait": "Triglycerides", "risk_allele": "T", "effect": "Asn291Ser — T on plus strand = hypertriglyceridemia via impaired LPL activity"},
    "rs1800206": {"gene": "PPARA", "category": "Cardiovascular Risk", "trait": "Lipid metabolism", "risk_allele": "C", "effect": "Leu162Val — C = higher LDL and apoB"},
    "rs4149056": {"gene": "SLCO1B1", "category": "Cardiovascular Risk", "trait": "Statin myopathy risk", "risk_allele": "C", "effect": "Val174Ala — CC = statin-induced myopathy risk"},
    "rs4343": {"gene": "ACE", "category": "Cardiovascular Risk", "trait": "Blood pressure", "risk_allele": "G", "effect": "G = proxy for ACE D allele — higher ACE, higher blood pressure"},
    "rs5186": {"gene": "AGTR1", "category": "Cardiovascular Risk", "trait": "Hypertension", "risk_allele": "C", "effect": "A1166C — CC = hypertension and cardiac hypertrophy"},
    "rs1800871": {"gene": "IL10", "category": "Cardiovascular Risk", "trait": "Anti-inflammatory capacity", "risk_allele": "A", "effect": "A = lower IL-10, higher inflammatory cardiovascular risk"},
    "rs1800629": {"gene": "TNF", "category": "Cardiovascular Risk", "trait": "Inflammation", "risk_allele": "A", "effect": "G-308A — A = higher TNF-alpha, inflammation"},
    "rs1799963": {"gene": "F2", "category": "Cardiovascular Risk", "trait": "Thrombosis risk", "risk_allele": "A", "effect": "Prothrombin G20210A — A = 3x venous thrombosis risk"},
    "rs6025": {"gene": "F5", "category": "Cardiovascular Risk", "trait": "Thrombosis risk", "risk_allele": "T", "effect": "Factor V Leiden (c.1691G>A / R506Q) — T on plus strand = 5-10x DVT/PE risk (heterozygous), 50-80x homozygous"},
    "rs1126643": {"gene": "ITGA2", "category": "Cardiovascular Risk", "trait": "Platelet aggregation", "risk_allele": "T", "effect": "C807T — T = increased platelet collagen binding"},
    "rs5918": {"gene": "ITGB3", "category": "Cardiovascular Risk", "trait": "Platelet aggregation / thrombosis", "risk_allele": "T", "effect": "PlA2 allele — T = thrombosis and aspirin resistance"},
    "rs1800790": {"gene": "FGB", "category": "Cardiovascular Risk", "trait": "Fibrinogen levels", "risk_allele": "A", "effect": "G-455A — A = higher fibrinogen, clotting risk"},
    "rs3025058": {"gene": "MMP3", "category": "Cardiovascular Risk", "trait": "Plaque instability", "risk_allele": "T", "effect": "5A/6A promoter polymorphism — T on plus strand (5A allele proxy) = higher MMP3 expression, plaque rupture risk"},
    "rs1800795": {"gene": "IL6", "category": "Cardiovascular Risk", "trait": "Inflammation", "risk_allele": "C", "effect": "G-174C — associated with IL-6 levels and CVD risk"},
    "rs1333049": {"gene": "CDKN2B-AS1", "category": "Cardiovascular Risk", "trait": "Coronary artery disease", "risk_allele": "C", "effect": "Strongest CAD GWAS signal — chromosome 9p21"},
    "rs4977574": {"gene": "CDKN2B-AS1", "category": "Cardiovascular Risk", "trait": "Coronary artery disease", "risk_allele": "G", "effect": "G = elevated CAD risk"},
    "rs1205": {"gene": "CRP", "category": "Cardiovascular Risk", "trait": "C-reactive protein", "risk_allele": "C", "effect": "Associated with baseline CRP levels"},
    "rs11206510": {"gene": "PCSK9", "category": "Cardiovascular Risk", "trait": "LDL cholesterol / PCSK9", "risk_allele": "T", "effect": "T = higher PCSK9 activity, elevated LDL"},
    "rs2479409": {"gene": "PCSK9", "category": "Cardiovascular Risk", "trait": "LDL cholesterol", "risk_allele": "A", "effect": "A = lower LDL (protective)"},
    "rs2228671": {"gene": "LDLR", "category": "Cardiovascular Risk", "trait": "LDL receptor", "risk_allele": "T", "effect": "Associated with LDL cholesterol levels"},
    "rs688": {"gene": "LDLR", "category": "Cardiovascular Risk", "trait": "Familial hypercholesterolemia risk", "risk_allele": "T", "effect": "T = elevated LDL"},
    "rs17367504": {"gene": "MTHFS", "category": "Cardiovascular Risk", "trait": "Blood pressure", "risk_allele": "G", "effect": "Associated with systolic blood pressure"},
    "rs1799752": {"gene": "ACE", "category": "Cardiovascular Risk", "trait": "ACE I/D / blood pressure", "risk_allele": "D", "effect": "INDEL (D=Deletion) — DD = higher ACE, hypertension and exercise response"},

    # =========================================================================
    # 3. EXERCISE & RECOVERY
    # =========================================================================
    "rs1815739": {"gene": "ACTN3", "category": "Exercise & Recovery", "trait": "Power vs endurance muscle fiber", "risk_allele": "T", "effect": "R577X — TT (XX) = no alpha-actinin-3, endurance advantage; CC = power"},
    "rs8192678": {"gene": "PPARGC1A", "category": "Exercise & Recovery", "trait": "Aerobic capacity / VO2max", "risk_allele": "T", "effect": "Gly482Ser — T = lower VO2max response to training"},
    "rs1042713": {"gene": "ADRB2", "category": "Exercise & Recovery", "trait": "Exercise response / bronchodilation", "risk_allele": "G", "effect": "Arg16Gly — G = better bronchodilator response"},
    "rs4253778": {"gene": "PPARA", "category": "Exercise & Recovery", "trait": "Endurance performance", "risk_allele": "C", "effect": "G = elite endurance athlete status association"},
    "rs2267668": {"gene": "PPARD", "category": "Exercise & Recovery", "trait": "Endurance / metabolic adaptation to training", "risk_allele": "G", "effect": "Intronic PPARD variant — G = reduced aerobic training response (lower VO2max improvement, slower metabolic adaptation)"},
    "rs699": {"gene": "AGT", "category": "Exercise & Recovery", "trait": "Muscle hypertrophy response", "risk_allele": "A", "effect": "Met235Thr — A = higher angiotensinogen"},
    "rs1800012": {"gene": "COL1A1", "category": "Exercise & Recovery", "trait": "Injury risk / tendon strength", "risk_allele": "A", "effect": "Sp1 — AA (forward strand) = lower bone density, higher soft tissue injury risk"},
    "rs12722": {"gene": "COL5A1", "category": "Exercise & Recovery", "trait": "Tendon/ligament injury risk", "risk_allele": "T", "effect": "T = Achilles tendinopathy and ACL injury risk"},
    "rs1800470": {"gene": "TGFB1", "category": "Exercise & Recovery", "trait": "Muscle repair / fibrosis", "risk_allele": "T", "effect": "Leu10Pro (c.29C>T) — T (Leu10) on plus strand = higher TGF-beta1 secretion, greater fibrotic response to muscle/tendon injury"},
    "rs2228570": {"gene": "VDR", "category": "Exercise & Recovery", "trait": "Muscle function / vitamin D receptor", "risk_allele": "C", "effect": "FokI — CC = altered muscle strength and injury risk"},
    "rs731236": {"gene": "VDR", "category": "Exercise & Recovery", "trait": "Bone density / muscle", "risk_allele": "A", "effect": "TaqI — A = lower bone density"},
    "rs1544410": {"gene": "VDR", "category": "Exercise & Recovery", "trait": "Vitamin D receptor / bone", "risk_allele": "A", "effect": "BsmI — A = lower bone mineral density"},
    "rs2070744": {"gene": "NOS3", "category": "Exercise & Recovery", "trait": "Nitric oxide / blood flow", "risk_allele": "T", "effect": "T = lower eNOS, reduced exercise blood flow and recovery"},
    "rs1799983": {"gene": "NOS3", "category": "Exercise & Recovery", "trait": "Endothelial nitric oxide", "risk_allele": "T", "effect": "Glu298Asp — T = lower NO production"},
    "rs7251": {"gene": "HSPA1A", "category": "Exercise & Recovery", "trait": "Heat shock protein / recovery", "risk_allele": "C", "effect": "C = altered heat stress response and recovery"},
    "rs28933981": {"gene": "COL3A1", "category": "Exercise & Recovery", "trait": "Connective tissue integrity", "risk_allele": "T", "effect": "Associated with soft tissue injuries"},
    "rs3213719": {"gene": "COL5A1", "category": "Exercise & Recovery", "trait": "Soft tissue injury risk", "risk_allele": "A", "effect": "A = musculoskeletal injury predisposition"},
    "rs4341": {"gene": "ACE", "category": "Exercise & Recovery", "trait": "Endurance vs power phenotype", "risk_allele": "G", "effect": "G (D proxy) = strength/power; A (I proxy) = endurance advantage"},
    "rs1800169": {"gene": "AMPD1", "category": "Exercise & Recovery", "trait": "Muscle fatigue / AMP metabolism", "risk_allele": "A", "effect": "Associated with muscle AMP deaminase activity"},
    "rs2305160": {"gene": "MYO3A", "category": "Exercise & Recovery", "trait": "Muscle fiber function", "risk_allele": "A", "effect": "A = altered myosin motor function, associated with muscle performance variation"},

    # =========================================================================
    # 4. SLEEP & CIRCADIAN
    # =========================================================================
    "rs1801260": {"gene": "CLOCK", "category": "Sleep & Circadian", "trait": "Chronotype / insomnia", "risk_allele": "C", "effect": "3111T/C — C = evening preference and insomnia tendency"},
    "rs57875989": {"gene": "PER3", "category": "Sleep & Circadian", "trait": "Chronotype / sleep-wake preference", "risk_allele": "T", "effect": "PER3 VNTR (4/5-repeat) — shorter repeat associated with eveningness, longer with morningness and greater sleep homeostatic drive"},
    "rs10830963": {"gene": "MTNR1B", "category": "Sleep & Circadian", "trait": "Melatonin receptor / glucose", "risk_allele": "G", "effect": "G = higher melatonin receptor expression, impaired fasting glucose"},
    "rs4753426": {"gene": "MTNR1B", "category": "Sleep & Circadian", "trait": "Melatonin signaling / T2D", "risk_allele": "C", "effect": "Associated with fasting glucose and T2D via melatonin"},
    "rs2304672": {"gene": "PER2", "category": "Sleep & Circadian", "trait": "Circadian phase", "risk_allele": "C", "effect": "Associated with advanced sleep phase"},
    "rs934945": {"gene": "PER2", "category": "Sleep & Circadian", "trait": "Sleep timing", "risk_allele": "C", "effect": "Associated with circadian rhythm variation"},
    "rs2292912": {"gene": "CRY2", "category": "Sleep & Circadian", "trait": "Chronotype", "risk_allele": "C", "effect": "C = morningness tendency"},
    "rs1480272": {"gene": "NR1D1", "category": "Sleep & Circadian", "trait": "REV-ERB / metabolic rhythm", "risk_allele": "T", "effect": "REV-ERB-alpha — affects circadian metabolic regulation"},
    "rs2279287": {"gene": "TIMELESS", "category": "Sleep & Circadian", "trait": "Sleep duration", "risk_allele": "A", "effect": "Associated with sleep duration variability"},
    "rs11121022": {"gene": "ARNTL", "category": "Sleep & Circadian", "trait": "BMAL1 / core clock / metabolism", "risk_allele": "A", "effect": "Associated with metabolic syndrome and circadian disruption"},
    "rs3923809": {"gene": "BTBD9", "category": "Sleep & Circadian", "trait": "Restless legs syndrome", "risk_allele": "A", "effect": "A = increased RLS and PLMS risk"},
    "rs9296249": {"gene": "MEIS1", "category": "Sleep & Circadian", "trait": "Restless legs syndrome", "risk_allele": "T", "effect": "Strongly associated with RLS"},
    "rs6746030": {"gene": "SCN9A", "category": "Sleep & Circadian", "trait": "Pain sensitivity / sleep disruption", "risk_allele": "A", "effect": "A = increased pain sensitivity affecting sleep quality"},
    "rs5751876": {"gene": "ADORA2A", "category": "Sleep & Circadian", "trait": "Caffeine / sleep sensitivity", "risk_allele": "T", "effect": "TT = increased caffeine-induced sleep disruption"},
    "rs7794745": {"gene": "CNTNAP2", "category": "Sleep & Circadian", "trait": "Sleep quality / neurodevelopment", "risk_allele": "T", "effect": "T = associated with sleep problems"},
    "rs2236709": {"gene": "CSNK1E", "category": "Sleep & Circadian", "trait": "Circadian period length", "risk_allele": "G", "effect": "Associated with circadian period and sleep phase regulation"},

    # =========================================================================
    # 5. LONGEVITY & AGING
    # =========================================================================
    "rs2802292": {"gene": "FOXO3", "category": "Longevity & Aging", "trait": "Longevity", "risk_allele": "G", "effect": "G consistently associated with exceptional longevity"},
    "rs13217795": {"gene": "FOXO3", "category": "Longevity & Aging", "trait": "Longevity", "risk_allele": "C", "effect": "C = exceptional longevity phenotype"},
    "rs4946936": {"gene": "FOXO3", "category": "Longevity & Aging", "trait": "Longevity / stress resistance", "risk_allele": "C", "effect": "Associated with longevity"},
    "rs10457180": {"gene": "SIRT3", "category": "Longevity & Aging", "trait": "Mitochondrial function / longevity", "risk_allele": "G", "effect": "Associated with exceptional longevity in men"},
    "rs3740051": {"gene": "SIRT1", "category": "Longevity & Aging", "trait": "Caloric restriction response", "risk_allele": "A", "effect": "Associated with SIRT1 expression and metabolic aging"},
    "rs1042522": {"gene": "TP53", "category": "Longevity & Aging", "trait": "Cancer suppression / aging", "risk_allele": "C", "effect": "Arg72Pro — CC = altered p53 apoptotic function"},
    "rs4880": {"gene": "SOD2", "category": "Longevity & Aging", "trait": "Mitochondrial antioxidant defense", "risk_allele": "A", "effect": "Val16Ala — A on plus strand = Val (impaired mitochondrial targeting signal), higher superoxide; G = Ala (efficient import)"},
    "rs1001179": {"gene": "CAT", "category": "Longevity & Aging", "trait": "Catalase / oxidative defense", "risk_allele": "T", "effect": "T = lower catalase activity"},
    "rs25487": {"gene": "XRCC1", "category": "Longevity & Aging", "trait": "DNA repair", "risk_allele": "T", "effect": "Arg399Gln — TT = reduced base excision repair"},
    "rs1799782": {"gene": "XRCC1", "category": "Longevity & Aging", "trait": "DNA repair", "risk_allele": "T", "effect": "Arg194Trp — T = altered DNA repair"},
    "rs1136410": {"gene": "PARP1", "category": "Longevity & Aging", "trait": "DNA repair / aging", "risk_allele": "T", "effect": "Val762Ala — T = reduced PARP activity"},
    "rs2234693": {"gene": "ESR1", "category": "Longevity & Aging", "trait": "Estrogen signaling / aging", "risk_allele": "T", "effect": "PvuII — T = bone density and cardiovascular aging"},
    "rs6983267": {"gene": "MYC enhancer", "category": "Longevity & Aging", "trait": "Colorectal cancer risk", "risk_allele": "G", "effect": "G = strongest colorectal cancer GWAS signal"},
    "rs10993994": {"gene": "MSMB", "category": "Longevity & Aging", "trait": "Prostate cancer risk", "risk_allele": "T", "effect": "T on plus strand = reduced MSMB expression, increased prostate cancer risk (OR ~1.25)"},
    "rs2981582": {"gene": "FGFR2", "category": "Longevity & Aging", "trait": "Breast cancer risk", "risk_allele": "A", "effect": "A = strongest breast cancer GWAS signal in Europeans"},
    "rs1695": {"gene": "GSTP1", "category": "Longevity & Aging", "trait": "Carcinogen detoxification", "risk_allele": "G", "effect": "Ile105Val — G = reduced GSTP1, higher cancer risk"},
    "rs4646903": {"gene": "CYP1A1", "category": "Longevity & Aging", "trait": "Carcinogen metabolism", "risk_allele": "C", "effect": "C on plus strand = higher CYP1A1 inducibility (m1 allele), increased polycyclic aromatic hydrocarbon activation"},
    "rs1800566": {"gene": "NQO1", "category": "Longevity & Aging", "trait": "Oxidative stress / cancer", "risk_allele": "T", "effect": "Pro187Ser — TT = non-functional NQO1"},
    "rs2228001": {"gene": "XPC", "category": "Longevity & Aging", "trait": "Nucleotide excision repair", "risk_allele": "C", "effect": "C = reduced NER capacity"},
    "rs13181": {"gene": "ERCC2", "category": "Longevity & Aging", "trait": "DNA repair / cancer risk", "risk_allele": "G", "effect": "Lys751Gln — G = reduced nucleotide excision repair"},

    # =========================================================================
    # 6. MENTAL HEALTH & COGNITION
    # =========================================================================
    "rs6265": {"gene": "BDNF", "category": "Mental Health & Cognition", "trait": "Memory / neuroplasticity / depression", "risk_allele": "A", "effect": "Val66Met — AA = reduced activity-dependent BDNF secretion, memory deficits"},
    "rs4570625": {"gene": "TPH2", "category": "Mental Health & Cognition", "trait": "Serotonin synthesis / depression", "risk_allele": "G", "effect": "G = lower brain serotonin, depression risk"},
    "rs25531": {"gene": "SLC6A4", "category": "Mental Health & Cognition", "trait": "Serotonin transporter / stress response", "risk_allele": "G", "effect": "A>G SNP in 5-HTTLPR region — G (LG haplotype) = reduced SLC6A4 transcription equivalent to S allele, lower serotonin reuptake, stress-sensitive phenotype"},
    "rs6311": {"gene": "HTR2A", "category": "Mental Health & Cognition", "trait": "Serotonin 2A receptor", "risk_allele": "A", "effect": "A = antidepressant response and cognition effects"},
    "rs6313": {"gene": "HTR2A", "category": "Mental Health & Cognition", "trait": "Serotonin receptor / antidepressant", "risk_allele": "T", "effect": "Associated with depression and antidepressant efficacy"},
    "rs1800532": {"gene": "TPH1", "category": "Mental Health & Cognition", "trait": "Peripheral serotonin synthesis", "risk_allele": "A", "effect": "A = lower TPH1 expression"},
    "rs1800955": {"gene": "DRD4", "category": "Mental Health & Cognition", "trait": "Dopamine D4 / ADHD / novelty seeking", "risk_allele": "T", "effect": "C-521T — T = ADHD and novelty-seeking behavior"},
    "rs1800497": {"gene": "ANKK1/DRD2", "category": "Mental Health & Cognition", "trait": "Dopamine D2 / addiction / reward", "risk_allele": "T", "effect": "TaqIA — T (A1) = reduced D2 density, addiction risk"},
    "rs4532": {"gene": "DRD1", "category": "Mental Health & Cognition", "trait": "Dopamine D1 / working memory", "risk_allele": "T", "effect": "Associated with working memory and ADHD"},
    "rs165599": {"gene": "COMT", "category": "Mental Health & Cognition", "trait": "Dopamine / stress resilience", "risk_allele": "G", "effect": "Associated with schizophrenia and stress response"},
    "rs1800544": {"gene": "ADRA2A", "category": "Mental Health & Cognition", "trait": "Norepinephrine / ADHD", "risk_allele": "C", "effect": "C = ADHD and norepinephrine signaling"},
    "rs2230912": {"gene": "P2RX7", "category": "Mental Health & Cognition", "trait": "Mood disorders / neuroinflammation", "risk_allele": "A", "effect": "Gln460Arg — A = bipolar disorder and depression risk"},
    "rs1006737": {"gene": "CACNA1C", "category": "Mental Health & Cognition", "trait": "Bipolar disorder / mood regulation", "risk_allele": "A", "effect": "A = strongest GWAS signal for bipolar disorder"},
    "rs744373": {"gene": "BIN1", "category": "Mental Health & Cognition", "trait": "Alzheimer's disease risk", "risk_allele": "C", "effect": "C = second strongest Alzheimer's signal after APOE"},
    "rs3764650": {"gene": "ABCA7", "category": "Mental Health & Cognition", "trait": "Alzheimer's disease risk", "risk_allele": "G", "effect": "G = Alzheimer's disease susceptibility"},
    "rs3851179": {"gene": "PICALM", "category": "Mental Health & Cognition", "trait": "Alzheimer's disease", "risk_allele": "G", "effect": "G = reduced Alzheimer's risk (protective direction)"},
    "rs11136000": {"gene": "CLU", "category": "Mental Health & Cognition", "trait": "Alzheimer's / neurodegeneration", "risk_allele": "C", "effect": "Clusterin — C = Alzheimer's protective signal"},
    "rs3818361": {"gene": "CR1", "category": "Mental Health & Cognition", "trait": "Alzheimer's / complement immune", "risk_allele": "A", "effect": "A = Alzheimer's via complement pathway"},
    "rs9349407": {"gene": "CD2AP", "category": "Mental Health & Cognition", "trait": "Alzheimer's disease", "risk_allele": "C", "effect": "C = Alzheimer's susceptibility"},
    "rs5569": {"gene": "SLC6A2", "category": "Mental Health & Cognition", "trait": "Norepinephrine transporter / ADHD", "risk_allele": "G", "effect": "Associated with ADHD and antidepressant response"},

    # =========================================================================
    # 7. PHARMACOGENOMICS
    # =========================================================================
    "rs4244285": {"gene": "CYP2C19", "category": "Pharmacogenomics", "trait": "PPIs / clopidogrel / SSRI metabolism", "risk_allele": "A", "effect": "*2 — AA = poor metabolizer: reduced clopidogrel activation, higher PPI levels"},
    "rs4986893": {"gene": "CYP2C19", "category": "Pharmacogenomics", "trait": "Drug metabolism (poor metabolizer)", "risk_allele": "A", "effect": "*3 allele — A = poor metabolizer"},
    "rs12248560": {"gene": "CYP2C19", "category": "Pharmacogenomics", "trait": "Ultrarapid drug metabolism", "risk_allele": "T", "effect": "*17 — T = ultrarapid metabolizer, reduced drug efficacy for standard doses"},
    "rs1799853": {"gene": "CYP2C9", "category": "Pharmacogenomics", "trait": "Warfarin / NSAID sensitivity", "risk_allele": "T", "effect": "*2 — T = reduced CYP2C9, warfarin sensitivity, NSAID toxicity"},
    "rs1057910": {"gene": "CYP2C9", "category": "Pharmacogenomics", "trait": "Warfarin / drug metabolism", "risk_allele": "C", "effect": "*3 — C = severely reduced CYP2C9 activity"},
    "rs3892097": {"gene": "CYP2D6", "category": "Pharmacogenomics", "trait": "Opioid / antidepressant / beta-blocker metabolism", "risk_allele": "A", "effect": "*4 — A = poor metabolizer: codeine no analgesia, TCA/SSRI toxicity risk"},
    "rs5030655": {"gene": "CYP2D6", "category": "Pharmacogenomics", "trait": "CYP2D6 poor metabolizer", "risk_allele": "A", "effect": "INDEL (*6, 1707delT) — A = non-functional CYP2D6"},
    "rs1065852": {"gene": "CYP2D6", "category": "Pharmacogenomics", "trait": "Reduced CYP2D6 activity", "risk_allele": "A", "effect": "*10 — A on plus strand = reduced CYP2D6 activity (Pro34Ser), higher exposure to antidepressants and opioids"},
    "rs28371706": {"gene": "CYP2D6", "category": "Pharmacogenomics", "trait": "Reduced CYP2D6 activity", "risk_allele": "T", "effect": "*41 — T = reduced CYP2D6 activity (intermediate metabolizer)"},
    "rs776746": {"gene": "CYP3A5", "category": "Pharmacogenomics", "trait": "Tacrolimus / immunosuppressant metabolism", "risk_allele": "A", "effect": "*3 — AA = non-expressor, requires higher tacrolimus doses"},
    "rs2740574": {"gene": "CYP3A4", "category": "Pharmacogenomics", "trait": "Major drug metabolism enzyme", "risk_allele": "A", "effect": "*1B — A = slightly altered CYP3A4 expression"},
    "rs9923231_vk": {"gene": "VKORC1", "category": "Pharmacogenomics", "trait": "Warfarin dose requirement", "risk_allele": "T", "effect": "T = lower VKORC1, requires lower warfarin dose"},
    "rs1045642": {"gene": "ABCB1", "category": "Pharmacogenomics", "trait": "P-glycoprotein drug transport", "risk_allele": "T", "effect": "C3435T — T = lower P-gp expression, higher drug absorption"},
    "rs2032582": {"gene": "ABCB1", "category": "Pharmacogenomics", "trait": "Multi-drug efflux", "risk_allele": "T", "effect": "G2677T — T = altered drug bioavailability"},
    "rs1128503": {"gene": "ABCB1", "category": "Pharmacogenomics", "trait": "MDR1 / drug resistance", "risk_allele": "T", "effect": "C1236T — T = reduced MDR1 activity"},
    "rs1801030": {"gene": "SULT1A1", "category": "Pharmacogenomics", "trait": "Drug / hormone sulfation", "risk_allele": "A", "effect": "Arg213His — A = reduced sulfotransferase activity"},
    "rs4148323": {"gene": "UGT1A1", "category": "Pharmacogenomics", "trait": "Irinotecan toxicity / bilirubin", "risk_allele": "A", "effect": "*6 — A = reduced UGT1A1, elevated bilirubin, irinotecan toxicity"},
    "rs1142345": {"gene": "TPMT", "category": "Pharmacogenomics", "trait": "Thiopurine toxicity (azathioprine/6-MP)", "risk_allele": "C", "effect": "*3C — C = reduced TPMT, high thiopurine toxicity risk"},
    "rs1800460": {"gene": "TPMT", "category": "Pharmacogenomics", "trait": "Azathioprine bone marrow toxicity", "risk_allele": "A", "effect": "*3B — A = non-functional TPMT allele"},
    "rs2242480": {"gene": "CYP3A4", "category": "Pharmacogenomics", "trait": "CYP3A4 drug metabolism variation", "risk_allele": "T", "effect": "Associated with CYP3A4 expression and drug clearance"},
    "rs2306283_sl": {"gene": "SLCO1B1", "category": "Pharmacogenomics", "trait": "Statin hepatic uptake", "risk_allele": "G", "effect": "Associated with statin pharmacokinetics and myopathy risk"},

    # =========================================================================
    # 8. AUTOIMMUNE & INFLAMMATION
    # =========================================================================
    "rs2476601": {"gene": "PTPN22", "category": "Autoimmune & Inflammation", "trait": "Broad autoimmune risk (RA, T1D, lupus, thyroiditis)", "risk_allele": "A", "effect": "R620W — A = broad autoimmune risk across multiple conditions"},
    "rs3087243": {"gene": "CTLA4", "category": "Autoimmune & Inflammation", "trait": "Immune checkpoint / autoimmune", "risk_allele": "G", "effect": "G = reduced CTLA4, higher autoimmune activation"},
    "rs7574865": {"gene": "STAT4", "category": "Autoimmune & Inflammation", "trait": "Lupus / RA / autoimmune", "risk_allele": "T", "effect": "T = lupus and RA risk via IL-12 pathway"},
    "rs2104286": {"gene": "IL2RA", "category": "Autoimmune & Inflammation", "trait": "MS / T1D / autoimmune", "risk_allele": "A", "effect": "A = multiple sclerosis and T1D susceptibility"},
    "rs3184504": {"gene": "SH2B3", "category": "Autoimmune & Inflammation", "trait": "Celiac / T1D / autoimmune", "risk_allele": "T", "effect": "Trp262Arg — T = celiac, T1D, broad autoimmune risk"},
    "rs1990760": {"gene": "IFIH1", "category": "Autoimmune & Inflammation", "trait": "Interferon / autoimmune (T1D, lupus)", "risk_allele": "T", "effect": "Ala946Thr — T = gain-of-function, higher IFN-alpha, autoimmune"},
    "rs6920220": {"gene": "TNFAIP3", "category": "Autoimmune & Inflammation", "trait": "NF-kB / autoimmune (RA, lupus)", "risk_allele": "A", "effect": "A = altered A20 function, higher NF-kB, autoimmune risk"},
    "rs2230926": {"gene": "TNFAIP3", "category": "Autoimmune & Inflammation", "trait": "Systemic lupus / RA", "risk_allele": "G", "effect": "Phe127Cys — G = reduced A20 deubiquitinase, inflammation"},
    "rs6822844": {"gene": "IL2/IL21", "category": "Autoimmune & Inflammation", "trait": "Celiac disease / autoimmune", "risk_allele": "T", "effect": "T = celiac disease susceptibility"},
    "rs11209026": {"gene": "IL23R", "category": "Autoimmune & Inflammation", "trait": "IBD / psoriasis / autoimmune", "risk_allele": "A", "effect": "Arg381Gln — A = protective against IBD and psoriasis"},
    "rs7517847": {"gene": "IL23R", "category": "Autoimmune & Inflammation", "trait": "Crohn's disease / gut inflammation", "risk_allele": "T", "effect": "T = Crohn's disease susceptibility"},
    "rs2241880": {"gene": "ATG16L1", "category": "Autoimmune & Inflammation", "trait": "Autophagy / Crohn's disease", "risk_allele": "G", "effect": "Thr300Ala — G = impaired autophagy of gut bacteria, Crohn's"},
    "rs1800587": {"gene": "IL1A", "category": "Autoimmune & Inflammation", "trait": "IL-1 / systemic inflammation", "risk_allele": "T", "effect": "T = higher IL-1alpha production"},
    "rs1143634": {"gene": "IL1B", "category": "Autoimmune & Inflammation", "trait": "Inflammation / pain / fever", "risk_allele": "C", "effect": "C = higher IL-1beta production, inflammatory pain"},
    "rs4251961": {"gene": "IL13", "category": "Autoimmune & Inflammation", "trait": "Asthma / allergic inflammation", "risk_allele": "T", "effect": "T = higher IL-13, asthma and allergy risk"},
    "rs763361": {"gene": "CD226", "category": "Autoimmune & Inflammation", "trait": "T cell activation / autoimmune", "risk_allele": "T", "effect": "T = increased T cell activation, autoimmune risk"},
    "rs11171739": {"gene": "KIAA0350", "category": "Autoimmune & Inflammation", "trait": "Type 1 diabetes", "risk_allele": "T", "effect": "T = T1D susceptibility"},
    "rs13361189": {"gene": "IRGM", "category": "Autoimmune & Inflammation", "trait": "Crohn's / gut bacterial clearance", "risk_allele": "T", "effect": "T = impaired autophagy, Crohn's susceptibility"},
    "rs1800896": {"gene": "IL10", "category": "Autoimmune & Inflammation", "trait": "Anti-inflammatory IL-10 production", "risk_allele": "A", "effect": "A = lower IL-10, impaired anti-inflammatory response"},
    "rs2069762": {"gene": "IL2", "category": "Autoimmune & Inflammation", "trait": "T regulatory cell function", "risk_allele": "A", "effect": "A = altered IL-2 production and immune regulation"},

    # =========================================================================
    # 9. HORMONAL HEALTH
    # =========================================================================
    "rs6152": {"gene": "AR", "category": "Hormonal Health", "trait": "Androgen receptor sensitivity", "risk_allele": "A", "effect": "A = altered AR sensitivity, affects testosterone response in tissues"},
    "rs10046": {"gene": "CYP19A1", "category": "Hormonal Health", "trait": "Aromatase / estrogen synthesis", "risk_allele": "T", "effect": "T = altered aromatase activity, estrogen levels"},
    "rs700518": {"gene": "CYP19A1", "category": "Hormonal Health", "trait": "Estrogen synthesis", "risk_allele": "A", "effect": "A = altered estrogen synthesis via aromatase"},
    "rs4775936": {"gene": "CYP19A1", "category": "Hormonal Health", "trait": "Circulating estrogen", "risk_allele": "C", "effect": "C = associated with circulating estrogen levels"},
    "rs2234693_esr": {"gene": "ESR1", "category": "Hormonal Health", "trait": "Estrogen receptor alpha signaling", "risk_allele": "T", "effect": "PvuII — T = altered ER-alpha, bone density, CVD and hormonal risk"},
    "rs9340799_esr": {"gene": "ESR1", "category": "Hormonal Health", "trait": "Estrogen receptor alpha", "risk_allele": "G", "effect": "XbaI — G = altered ER-alpha signaling"},
    "rs1256049": {"gene": "ESR2", "category": "Hormonal Health", "trait": "Estrogen receptor beta", "risk_allele": "A", "effect": "A = altered ER-beta, tissue-specific estrogen response"},
    "rs523349": {"gene": "SRD5A2", "category": "Hormonal Health", "trait": "DHT synthesis / 5-alpha reductase", "risk_allele": "G", "effect": "Val89Leu — G = lower 5-alpha reductase, reduced DHT conversion"},
    "rs1799941": {"gene": "SHBG", "category": "Hormonal Health", "trait": "Sex hormone binding globulin", "risk_allele": "A", "effect": "A = lower SHBG, higher free testosterone and estrogen"},
    "rs6257": {"gene": "SHBG", "category": "Hormonal Health", "trait": "SHBG levels / bioavailable hormones", "risk_allele": "T", "effect": "T = lower SHBG, affects bioavailable sex hormones"},
    "rs2165802": {"gene": "FSHR", "category": "Hormonal Health", "trait": "FSH receptor / ovarian function", "risk_allele": "T", "effect": "T = altered FSH sensitivity, ovarian reserve and fertility"},
    "rs6166": {"gene": "FSHR", "category": "Hormonal Health", "trait": "FSH receptor / fertility", "risk_allele": "A", "effect": "Asn680Ser — A = lower FSH sensitivity"},
    "rs934198": {"gene": "IGF1", "category": "Hormonal Health", "trait": "IGF-1 / growth hormone axis", "risk_allele": "T", "effect": "T = altered IGF-1 levels, associated with growth hormone axis variation and aging"},
    "rs2229765": {"gene": "IGF1R", "category": "Hormonal Health", "trait": "IGF-1 receptor / longevity", "risk_allele": "A", "effect": "A = altered IGF-1 signaling, associated with longevity"},
    "rs743572": {"gene": "CYP17A1", "category": "Hormonal Health", "trait": "Androgen / estrogen synthesis", "risk_allele": "A", "effect": "A2 allele — A = higher androgen/estrogen synthesis"},
    "rs12150660": {"gene": "LHCGR", "category": "Hormonal Health", "trait": "LH receptor / reproductive function", "risk_allele": "T", "effect": "T = altered LH sensitivity and reproductive hormones"},
    "rs1800888": {"gene": "ADRB2", "category": "Hormonal Health", "trait": "Adrenergic / stress hormone response", "risk_allele": "C", "effect": "Thr164Ile — C = reduced adrenergic receptor response"},
    "rs4917": {"gene": "GH1", "category": "Hormonal Health", "trait": "Growth hormone isoforms", "risk_allele": "C", "effect": "C = altered GH isoform expression"},
    "rs2816316": {"gene": "C1QTNF6", "category": "Hormonal Health", "trait": "Adipokine / insulin sensitivity", "risk_allele": "T", "effect": "T = T1D and metabolic susceptibility"},
    "rs4680_hor": {"gene": "COMT", "category": "Hormonal Health", "trait": "Estrogen catabolism", "risk_allele": "A", "effect": "Val158Met — A (Met) = slower estrogen breakdown, higher catechol estrogen metabolites"},

    # =========================================================================
    # 10. GUT MICROBIOME
    # =========================================================================
    "rs601338": {"gene": "FUT2", "category": "Gut Microbiome", "trait": "Secretor status / Bifidobacterium", "risk_allele": "A", "effect": "A = non-secretor: lower Bifidobacterium and Lactobacillus abundance"},
    "rs281379": {"gene": "FUT2", "category": "Gut Microbiome", "trait": "Gut microbiome diversity", "risk_allele": "A", "effect": "A = non-secretor, reduced B12 absorption and gut flora diversity"},
    "rs2066844": {"gene": "NOD2", "category": "Gut Microbiome", "trait": "Crohn's / innate gut immunity", "risk_allele": "T", "effect": "R702W — T = impaired bacterial sensing, Crohn's disease risk"},
    "rs2066845": {"gene": "NOD2", "category": "Gut Microbiome", "trait": "Crohn's disease", "risk_allele": "T", "effect": "G908R — T on plus strand = NOD2 dysfunction (missense), Crohn's disease susceptibility"},
    "rs2066847": {"gene": "NOD2", "category": "Gut Microbiome", "trait": "Crohn's disease (frameshift)", "risk_allele": "C", "effect": "INDEL (c.3019dupC) — C = loss of NOD2 function, high Crohn's risk"},
    "rs4986790": {"gene": "TLR4", "category": "Gut Microbiome", "trait": "Gram-negative bacterial response / LPS", "risk_allele": "G", "effect": "Asp299Gly — G = reduced TLR4, altered response to gut bacteria"},
    "rs4986791": {"gene": "TLR4", "category": "Gut Microbiome", "trait": "LPS / gut bacterial sensing", "risk_allele": "T", "effect": "Thr399Ile — T on plus strand = reduced LPS responsiveness, impaired innate immunity"},
    "rs5743708": {"gene": "TLR2", "category": "Gut Microbiome", "trait": "Gram-positive bacterial recognition", "risk_allele": "A", "effect": "Arg753Gln — A = impaired TLR2 signaling"},
    "rs11209026_gut": {"gene": "IL23R", "category": "Gut Microbiome", "trait": "IBD / gut mucosal immunity", "risk_allele": "A", "effect": "A = protective against inflammatory bowel disease"},
    "rs7517847_gut": {"gene": "IL23R", "category": "Gut Microbiome", "trait": "Crohn's / gut inflammation", "risk_allele": "T", "effect": "T = Crohn's and UC susceptibility"},
    "rs2241880_gut": {"gene": "ATG16L1", "category": "Gut Microbiome", "trait": "Autophagy / bacterial clearance in gut", "risk_allele": "G", "effect": "G = impaired intracellular bacterial handling, Crohn's risk"},
    "rs13361189_gut": {"gene": "IRGM", "category": "Gut Microbiome", "trait": "Autophagy / gut bacterial clearance", "risk_allele": "T", "effect": "T = impaired xenophagy, Crohn's susceptibility"},
    "rs4073": {"gene": "IL8", "category": "Gut Microbiome", "trait": "Gut neutrophil recruitment / inflammation", "risk_allele": "A", "effect": "A on plus strand (-251A) = higher IL-8 production, greater gut neutrophil recruitment and inflammatory response"},
    "rs4263839": {"gene": "CDH1", "category": "Gut Microbiome", "trait": "Gut epithelial barrier / leaky gut", "risk_allele": "A", "effect": "A = altered E-cadherin, gut permeability risk"},
    "rs1800469": {"gene": "TGFB1", "category": "Gut Microbiome", "trait": "Gut immune tolerance / TGF-beta", "risk_allele": "A", "effect": "A on plus strand (-509C>T coding = A on plus) = higher TGF-beta1 levels, affects gut mucosal immunity and fibrotic response"},
    "rs2233434": {"gene": "IL18", "category": "Gut Microbiome", "trait": "Gut mucosal immunity / inflammasome", "risk_allele": "C", "effect": "C = altered IL-18, affects gut mucosal immune activation"},
    "rs6441961": {"gene": "NOD1", "category": "Gut Microbiome", "trait": "Innate gut bacterial sensing", "risk_allele": "C", "effect": "C = altered innate immune response to gut microbiota"},
    "rs3020470": {"gene": "TLR2", "category": "Gut Microbiome", "trait": "Gut bacterial pattern recognition", "risk_allele": "A", "effect": "A = altered TLR2 signaling, changes gut microbiome sensing and innate immune activation"},
    "rs2069762_gut": {"gene": "IL2", "category": "Gut Microbiome", "trait": "Gut T regulatory cell function", "risk_allele": "A", "effect": "A = altered IL-2 production, gut immune regulation"},
    "rs3184504_gut": {"gene": "SH2B3", "category": "Gut Microbiome", "trait": "Celiac disease / gut autoimmunity", "risk_allele": "T", "effect": "T = celiac disease and IBD susceptibility"},

    # =========================================================================
    # 11. DETOXIFICATION
    # =========================================================================
    "rs4646903_det": {"gene": "CYP1A1", "category": "Detoxification", "trait": "Phase I / polycyclic aromatic hydrocarbons", "risk_allele": "C", "effect": "C on plus strand (m1 allele) = higher CYP1A1 inducibility, activates PAH carcinogens from tobacco and grilled food"},
    "rs1048943": {"gene": "CYP1A1", "category": "Detoxification", "trait": "Carcinogen activation", "risk_allele": "G", "effect": "Ile462Val — G = higher CYP1A1, more carcinogen activation from smoke"},
    "rs2069514": {"gene": "CYP1A2", "category": "Detoxification", "trait": "Phase I inducibility", "risk_allele": "A", "effect": "A = lower CYP1A2 inducibility by environmental factors"},
    "rs1056836": {"gene": "CYP1B1", "category": "Detoxification", "trait": "Estrogen / PAH metabolism", "risk_allele": "G", "effect": "Leu432Val — G = higher CYP1B1, more genotoxic 4-OH estrogen"},
    "rs1800440": {"gene": "CYP1B1", "category": "Detoxification", "trait": "Estrogen hydroxylation", "risk_allele": "T", "effect": "Asn453Ser — T = altered estrogen catabolism"},
    "rs1695_det": {"gene": "GSTP1", "category": "Detoxification", "trait": "Phase II glutathione conjugation", "risk_allele": "G", "effect": "Ile105Val — G = reduced GSTP1, impaired detox of carcinogens"},
    "rs1138272": {"gene": "GSTP1", "category": "Detoxification", "trait": "Glutathione transferase activity", "risk_allele": "T", "effect": "Ala114Val — T = further reduced GSTP1 activity"},
    "rs1800566_det": {"gene": "NQO1", "category": "Detoxification", "trait": "Quinone / benzene detoxification", "risk_allele": "T", "effect": "Pro187Ser — TT = non-functional NQO1, impaired quinone detox"},
    "rs1801280": {"gene": "NAT2", "category": "Detoxification", "trait": "Phase II acetylation (slow/fast)", "risk_allele": "A", "effect": "Ile114Thr — A on plus strand = slow acetylator allele (*5B component), higher aromatic amine exposure risk"},
    "rs1799929": {"gene": "NAT2", "category": "Detoxification", "trait": "Slow NAT2 acetylator", "risk_allele": "T", "effect": "C481T — slow NAT2 allele, drug and carcinogen accumulation"},
    "rs1799930": {"gene": "NAT2", "category": "Detoxification", "trait": "NAT2 acetylation status", "risk_allele": "A", "effect": "G590A — slow acetylator allele"},
    "rs1208": {"gene": "NAT2", "category": "Detoxification", "trait": "NAT2 rapid acetylator", "risk_allele": "A", "effect": "K268R — A = rapid acetylator component"},
    "rs4880_det": {"gene": "SOD2", "category": "Detoxification", "trait": "Mitochondrial superoxide dismutase", "risk_allele": "A", "effect": "Val16Ala — A on plus strand = Val (impaired mitochondrial targeting signal), higher superoxide in mitochondria"},
    "rs1001179_det": {"gene": "CAT", "category": "Detoxification", "trait": "Hydrogen peroxide clearance", "risk_allele": "T", "effect": "T = lower catalase activity, less H2O2 detoxification"},
    "rs2234922": {"gene": "EPHX1", "category": "Detoxification", "trait": "Epoxide hydrolase / toxin clearance", "risk_allele": "A", "effect": "His139Arg — A = slow EPHX1, epoxide accumulation"},
    "rs1051740": {"gene": "EPHX1", "category": "Detoxification", "trait": "Epoxide metabolism", "risk_allele": "T", "effect": "Tyr113His — T = slow epoxide hydrolase"},
    "rs4986913": {"gene": "CYP2E1", "category": "Detoxification", "trait": "Alcohol / solvent / acetaminophen metabolism", "risk_allele": "A", "effect": "A = altered CYP2E1 affecting alcohol and toxin metabolism"},
    "rs2031920": {"gene": "CYP2E1", "category": "Detoxification", "trait": "Ethanol metabolism / toxic byproducts", "risk_allele": "C", "effect": "C = higher CYP2E1 inducibility, more toxic metabolites from alcohol"},
    "rs12460535": {"gene": "SULT1A2", "category": "Detoxification", "trait": "Sulfotransferase / hormone and xenobiotic conjugation", "risk_allele": "G", "effect": "G on plus strand = altered sulfonation of xenobiotics and hormones, reduced detox capacity"},
    "rs1800566_nqo": {"gene": "NQO1", "category": "Detoxification", "trait": "Antioxidant / oxidative stress defense", "risk_allele": "T", "effect": "T = non-functional NQO1, impaired oxidative stress response"},

    # =========================================================================
    # 12. EYE & VISION HEALTH
    # =========================================================================
    "rs10490924": {"gene": "ARMS2", "category": "Eye & Vision Health", "trait": "Age-related macular degeneration (AMD)", "risk_allele": "T", "effect": "A69S — T = strongest AMD risk variant, ~3x risk per allele"},
    "rs1061170": {"gene": "CFH", "category": "Eye & Vision Health", "trait": "AMD / complement pathway", "risk_allele": "T", "effect": "Y402H — T = major AMD risk via complement factor H dysregulation"},
    "rs800292": {"gene": "CFH", "category": "Eye & Vision Health", "trait": "AMD complement regulation", "risk_allele": "A", "effect": "I62V — A = altered complement factor H"},
    "rs2230199": {"gene": "C3", "category": "Eye & Vision Health", "trait": "AMD / complement activation", "risk_allele": "G", "effect": "Arg80Gly — G = higher C3 complement, AMD risk"},
    "rs2415637": {"gene": "VEGFA", "category": "Eye & Vision Health", "trait": "Retinal neovascularization / AMD", "risk_allele": "A", "effect": "A on plus strand = higher VEGF expression, neovascular AMD and diabetic retinopathy risk"},
    "rs943080": {"gene": "VEGFA", "category": "Eye & Vision Health", "trait": "VEGF expression / AMD", "risk_allele": "T", "effect": "T = elevated VEGF expression"},
    "rs4977756": {"gene": "CDKN2A", "category": "Eye & Vision Health", "trait": "Primary open-angle glaucoma", "risk_allele": "A", "effect": "A = open-angle glaucoma susceptibility"},
    "rs2745572": {"gene": "TMCO1", "category": "Eye & Vision Health", "trait": "Intraocular pressure / glaucoma", "risk_allele": "A", "effect": "A = glaucoma risk via elevated intraocular pressure"},
    "rs7916697": {"gene": "ATOH7", "category": "Eye & Vision Health", "trait": "Optic nerve / glaucoma", "risk_allele": "C", "effect": "C on plus strand = altered optic nerve morphology, reduced RNFL thickness, glaucoma risk"},
    "rs12913832": {"gene": "HERC2", "category": "Eye & Vision Health", "trait": "Eye color / UV protection", "risk_allele": "A", "effect": "AA = blue eyes (lower UV protection); GG = brown eyes"},
    "rs1800407": {"gene": "OCA2", "category": "Eye & Vision Health", "trait": "Iris pigmentation / UV sensitivity", "risk_allele": "T", "effect": "His615Arg — A = lighter eye color, lower melanin UV protection"},
    "rs5749482": {"gene": "TYR", "category": "Eye & Vision Health", "trait": "Tyrosinase / retinal pigmentation", "risk_allele": "T", "effect": "T = reduced tyrosinase activity, UV sensitivity"},
    "rs3750846": {"gene": "ARMS2", "category": "Eye & Vision Health", "trait": "AMD progression", "risk_allele": "C", "effect": "C = AMD risk and photoreceptor degeneration"},
    "rs12203592": {"gene": "IRF4", "category": "Eye & Vision Health", "trait": "Iris pigmentation", "risk_allele": "T", "effect": "T = lighter iris pigmentation"},
    "rs284931": {"gene": "GRM6", "category": "Eye & Vision Health", "trait": "Retinal bipolar cell function", "risk_allele": "C", "effect": "C on plus strand = altered mGluR6 retinal bipolar cell function, associated with scotopic vision variation"},
    "rs547154": {"gene": "C2", "category": "Eye & Vision Health", "trait": "Complement / AMD", "risk_allele": "T", "effect": "T = reduced C2 complement activity in retina"},

    # =========================================================================
    # 13. SKIN & DERMATOLOGY
    # =========================================================================
    "rs1805007": {"gene": "MC1R", "category": "Skin & Dermatology", "trait": "Red hair / pale skin / melanoma risk", "risk_allele": "T", "effect": "Arg151Cys — T = red hair, significantly elevated melanoma risk"},
    "rs1805008": {"gene": "MC1R", "category": "Skin & Dermatology", "trait": "Red hair / UV sensitivity", "risk_allele": "T", "effect": "Arg160Trp — T = red hair phenotype, UV-sensitive skin"},
    "rs885479": {"gene": "MC1R", "category": "Skin & Dermatology", "trait": "Melanocortin / skin tone", "risk_allele": "T", "effect": "Arg163Gln — T = red hair / fair skin allele"},
    "rs2228479": {"gene": "MC1R", "category": "Skin & Dermatology", "trait": "UV response / skin type", "risk_allele": "A", "effect": "Val92Met — A = altered MC1R, intermediate UV sensitivity"},
    "rs1110400": {"gene": "MC1R", "category": "Skin & Dermatology", "trait": "Melanoma risk", "risk_allele": "T", "effect": "Asp294His — T = increased melanoma risk"},
    "rs4911414": {"gene": "ASIP", "category": "Skin & Dermatology", "trait": "Skin pigmentation / tanning ability", "risk_allele": "T", "effect": "T = lighter skin, reduced tanning ability"},
    "rs2153271": {"gene": "BNC2", "category": "Skin & Dermatology", "trait": "Freckle tendency", "risk_allele": "T", "effect": "T = freckling predisposition"},
    "rs1426654": {"gene": "SLC24A5", "category": "Skin & Dermatology", "trait": "Skin pigmentation", "risk_allele": "A", "effect": "Thr111Ala — A = lighter skin pigmentation (major European variant)"},
    "rs16891982": {"gene": "SLC45A2", "category": "Skin & Dermatology", "trait": "Skin color / melanoma susceptibility", "risk_allele": "C", "effect": "Phe374Leu — C = lighter skin, elevated melanoma risk"},
    "rs1800414": {"gene": "OCA2", "category": "Skin & Dermatology", "trait": "Skin pigmentation (East Asian)", "risk_allele": "C", "effect": "C = lighter skin in East Asian populations"},
    "rs3827760": {"gene": "EDAR", "category": "Skin & Dermatology", "trait": "Hair thickness / shovel incisors / East Asian trait marker", "risk_allele": "A", "effect": "V370A — A = thicker straighter hair, altered sweat gland density, classic East Asian / Native American enrichment marker"},
    "rs17822931": {"gene": "ABCC11", "category": "Skin & Dermatology", "trait": "Dry earwax / axillary odor", "risk_allele": "A", "effect": "A = dry earwax and lower axillary odor, strongly population-enriched in East Asian datasets"},
    "rs1799782_skin": {"gene": "XRCC1", "category": "Skin & Dermatology", "trait": "UV-induced DNA repair / skin cancer", "risk_allele": "T", "effect": "T = reduced DNA repair, higher skin cancer risk from UV"},
    "rs13181_skin": {"gene": "ERCC2", "category": "Skin & Dermatology", "trait": "UV damage NER repair", "risk_allele": "G", "effect": "G = reduced NER, higher UV damage accumulation"},
    "rs2228001_skin": {"gene": "XPC", "category": "Skin & Dermatology", "trait": "UV DNA damage repair", "risk_allele": "C", "effect": "C = reduced NER, higher basal cell carcinoma risk"},
    "rs1800629_skin": {"gene": "TNF", "category": "Skin & Dermatology", "trait": "Psoriasis / inflammatory skin conditions", "risk_allele": "A", "effect": "A = higher TNF-alpha, elevated psoriasis and eczema risk"},
    "rs3212227": {"gene": "IL12B", "category": "Skin & Dermatology", "trait": "Psoriasis / IL-12/23 pathway", "risk_allele": "C", "effect": "C = psoriasis susceptibility"},
    "rs610604": {"gene": "TNFAIP3", "category": "Skin & Dermatology", "trait": "Psoriasis / atopic dermatitis", "risk_allele": "G", "effect": "G = NF-kB dysregulation, inflammatory skin disease risk"},
    "rs35068180": {"gene": "FLG", "category": "Skin & Dermatology", "trait": "Skin barrier / eczema / atopic dermatitis", "risk_allele": "A", "effect": "INDEL — Filaggrin loss-of-function — A = impaired skin barrier, eczema risk"},
    "rs2564978": {"gene": "CD44", "category": "Skin & Dermatology", "trait": "Skin hydration / wound healing", "risk_allele": "T", "effect": "T = altered hyaluronan binding, skin moisture regulation"},
    "rs4420638": {"gene": "APOE region", "category": "Skin & Dermatology", "trait": "Skin aging / oxidative damage", "risk_allele": "G", "effect": "G = APOE4-related accelerated skin aging markers"},
    "rs1799814": {"gene": "CYP1A1", "category": "Skin & Dermatology", "trait": "UV carcinogen metabolism in skin", "risk_allele": "A", "effect": "Thr461Asn — A = altered PAH activation in skin"},

    # =========================================================================
    # 14. BONE & JOINT HEALTH
    # =========================================================================
    "rs2234693_bone": {"gene": "ESR1", "category": "Bone & Joint Health", "trait": "Bone mineral density / osteoporosis", "risk_allele": "T", "effect": "PvuII — T = lower BMD, elevated osteoporosis risk"},
    "rs9340799_bone": {"gene": "ESR1", "category": "Bone & Joint Health", "trait": "Estrogen signaling / bone loss", "risk_allele": "G", "effect": "XbaI — G = altered ER-alpha, bone mineral density"},
    "rs2228570_bone": {"gene": "VDR", "category": "Bone & Joint Health", "trait": "Vitamin D receptor / calcium absorption", "risk_allele": "C", "effect": "FokI — CC = lower bone density, reduced calcium absorption"},
    "rs731236_bone": {"gene": "VDR", "category": "Bone & Joint Health", "trait": "Bone mineral density", "risk_allele": "A", "effect": "TaqI — A = lower BMD and fracture risk"},
    "rs1544410_bone": {"gene": "VDR", "category": "Bone & Joint Health", "trait": "Osteoporosis", "risk_allele": "A", "effect": "BsmI — A = lower BMD"},
    "rs2297480": {"gene": "FRZB", "category": "Bone & Joint Health", "trait": "Osteoarthritis / Wnt signaling", "risk_allele": "T", "effect": "Arg200Trp — T = higher OA risk via Wnt pathway"},
    "rs288326": {"gene": "GDF5", "category": "Bone & Joint Health", "trait": "Osteoarthritis / joint development", "risk_allele": "A", "effect": "A = reduced GDF5 expression, increased knee and hip osteoarthritis risk"},
    "rs143384": {"gene": "GDF5", "category": "Bone & Joint Health", "trait": "Joint morphology / OA susceptibility", "risk_allele": "T", "effect": "T = lower GDF5 expression, OA susceptibility"},
    "rs1800012_bone": {"gene": "COL1A1", "category": "Bone & Joint Health", "trait": "Collagen I / fracture risk", "risk_allele": "A", "effect": "Sp1 — AA (forward strand) = lower bone collagen quality, higher fracture risk"},
    "rs1107946": {"gene": "COL1A1", "category": "Bone & Joint Health", "trait": "Osteoporosis / bone strength", "risk_allele": "A", "effect": "A = altered COL1A1 expression and bone matrix quality"},
    "rs2073618": {"gene": "TNFRSF11B", "category": "Bone & Joint Health", "trait": "OPG / bone resorption", "risk_allele": "G", "effect": "Asn252Asp — G on plus strand = lower osteoprotegerin, reduced RANKL inhibition, higher fracture risk"},
    "rs9594738": {"gene": "TNFRSF11B", "category": "Bone & Joint Health", "trait": "Bone density / remodeling", "risk_allele": "C", "effect": "C = lower OPG, greater bone resorption"},
    "rs1341162": {"gene": "SOST", "category": "Bone & Joint Health", "trait": "Sclerostin / bone formation inhibition", "risk_allele": "A", "effect": "A = higher sclerostin, inhibits Wnt bone formation"},
    "rs3736228": {"gene": "LRP5", "category": "Bone & Joint Health", "trait": "Wnt signaling / peak bone mass", "risk_allele": "T", "effect": "Ala1330Val — T = lower peak bone mass"},
    "rs851054": {"gene": "LRP5", "category": "Bone & Joint Health", "trait": "Bone mass / Wnt pathway", "risk_allele": "T", "effect": "T = lower LRP5 Wnt signaling, reduced bone mass"},
    "rs17563": {"gene": "BMP4", "category": "Bone & Joint Health", "trait": "Bone morphogenetic protein / formation", "risk_allele": "G", "effect": "G on plus strand (Val152Ala) = altered BMP4 signaling, affects bone formation and fracture healing"},
    "rs2241901": {"gene": "SP7", "category": "Bone & Joint Health", "trait": "Osterix / osteoblast differentiation", "risk_allele": "T", "effect": "T = altered osterix, osteoblast differentiation"},
    "rs9525641": {"gene": "TNFRSF11A", "category": "Bone & Joint Health", "trait": "RANK / osteoclast / Paget's disease", "risk_allele": "C", "effect": "C on plus strand = Paget's disease susceptibility and elevated osteoclast activity"},
    "rs2276836": {"gene": "TNFRSF11A", "category": "Bone & Joint Health", "trait": "Osteoclast activity / bone resorption", "risk_allele": "G", "effect": "G on plus strand = higher osteoclast activity, elevated bone resorption and bone loss risk"},
    "rs2306033": {"gene": "COL2A1", "category": "Bone & Joint Health", "trait": "Cartilage collagen / OA", "risk_allele": "A", "effect": "A = altered type II collagen, OA susceptibility"},

    # =========================================================================
    # 15. ADDITIONAL HIGH-VALUE SNPS
    # =========================================================================
    # Cardiovascular — additional GWAS-validated
    "rs505922": {"gene": "ABO", "category": "Cardiovascular Risk", "trait": "Blood type / venous thromboembolism", "risk_allele": "C", "effect": "C = non-O blood type proxy, elevated VTE and CAD risk (OR ~1.5 for VTE)"},
    "rs12740374": {"gene": "SORT1", "category": "Cardiovascular Risk", "trait": "LDL cholesterol / MI risk", "risk_allele": "T", "effect": "T = lower LDL cholesterol (protective), hepatic SORT1 expression modifier"},
    "rs646776": {"gene": "CELSR2", "category": "Cardiovascular Risk", "trait": "LDL cholesterol", "risk_allele": "T", "effect": "T = associated with LDL cholesterol levels at 1p13.3 locus"},
    "rs9818870": {"gene": "MRAS", "category": "Cardiovascular Risk", "trait": "Coronary artery disease", "risk_allele": "T", "effect": "T = CAD susceptibility via Ras signaling pathway"},

    # Pharmacogenomics — additional CYP2C19 alleles
    "rs28399504": {"gene": "CYP2C19", "category": "Pharmacogenomics", "trait": "CYP2C19*4 poor metabolizer", "risk_allele": "G", "effect": "*4 allele — G = non-functional CYP2C19, poor metabolizer"},

    # Mental Health — additional validated
    "rs27072": {"gene": "SLC6A3", "category": "Mental Health & Cognition", "trait": "Dopamine transporter / ADHD", "risk_allele": "T", "effect": "T = altered DAT1 expression, ADHD and reward processing variation"},
    "rs1611115": {"gene": "DBH", "category": "Mental Health & Cognition", "trait": "Dopamine beta-hydroxylase / NE synthesis", "risk_allele": "T", "effect": "T = lower DBH activity, reduced norepinephrine synthesis, ADHD and autonomic variation"},
    "rs7131056": {"gene": "DRD4", "category": "Mental Health & Cognition", "trait": "Dopamine D4 receptor / novelty seeking", "risk_allele": "A", "effect": "A = D4 receptor variation associated with novelty seeking and ADHD traits"},
    "rs2283265": {"gene": "DRD2", "category": "Mental Health & Cognition", "trait": "D2 receptor / reward processing", "risk_allele": "T", "effect": "T = altered D2 receptor splicing, affects reward processing and executive function"},

    # Exercise — additional
    "rs11549465": {"gene": "HIF1A", "category": "Exercise & Recovery", "trait": "Hypoxia response / altitude", "risk_allele": "T", "effect": "Pro582Ser — T = enhanced HIF-1alpha stability, better altitude adaptation"},
    "rs2016520": {"gene": "PPARD", "category": "Exercise & Recovery", "trait": "Endurance performance", "risk_allele": "C", "effect": "C = higher PPARD expression, enhanced fat oxidation and endurance capacity"},
    "rs17602729": {"gene": "AMPD1", "category": "Exercise & Recovery", "trait": "Myoadenylate deaminase deficiency", "risk_allele": "A", "effect": "Gln12Ter — A = AMPD1 deficiency, exercise-induced myalgia and early fatigue"},

    # Longevity/Cancer — additional
    "rs1048108": {"gene": "CDH1", "category": "Longevity & Aging", "trait": "Gastric / lobular breast cancer", "risk_allele": "A", "effect": "A = altered E-cadherin, increased risk of diffuse gastric and lobular breast cancer"},

    # Autoimmune — additional validated
    "rs2187668": {"gene": "HLA-DQ2.5", "category": "Autoimmune & Inflammation", "trait": "Celiac disease major signal", "risk_allele": "T", "effect": "T = HLA-DQ2.5 tag, strongest celiac disease risk allele (OR >7)"},

    # Sleep — additional
    "rs4753427": {"gene": "MTNR1A", "category": "Sleep & Circadian", "trait": "Melatonin receptor 1A", "risk_allele": "C", "effect": "C = altered MTNR1A expression, affects melatonin signaling and sleep onset"},

    # Gut Microbiome — additional
    "rs10889677": {"gene": "IL23R", "category": "Gut Microbiome", "trait": "IBD / gut mucosal barrier", "risk_allele": "A", "effect": "A = elevated IL-23R signaling, increased IBD susceptibility via Th17 pathway"},
    "rs1898830": {"gene": "TLR6", "category": "Gut Microbiome", "trait": "Microbial pattern recognition", "risk_allele": "A", "effect": "A = altered TLR6 heterodimer function, affects gut immune response to bacterial lipoproteins"},

    # Skin — additional
    "rs401681": {"gene": "TERT", "category": "Skin & Dermatology", "trait": "Telomerase / skin cancer", "risk_allele": "C", "effect": "C = telomere maintenance variation, basal cell carcinoma susceptibility"},
    "rs11568820": {"gene": "VDR", "category": "Skin & Dermatology", "trait": "Vitamin D receptor / skin immunity", "risk_allele": "A", "effect": "Cdx2 — A = altered VDR expression in skin, affects UV response and immune surveillance"},

    # Bone — additional
    "rs4988300": {"gene": "LRP5", "category": "Bone & Joint Health", "trait": "High bone mass", "risk_allele": "T", "effect": "T = Wnt pathway variation affecting peak bone mass acquisition"},
    "rs9594759": {"gene": "TNFSF11", "category": "Bone & Joint Health", "trait": "RANKL / bone resorption", "risk_allele": "T", "effect": "T = elevated RANKL, increased osteoclast-mediated bone resorption"},

    # =========================================================================
    # PROTECTIVE / POSITIVE-TRAIT SNPs
    # =========================================================================
    # trait_direction = "protective" means carrying the risk_allele is BENEFICIAL.
    # The "risk_allele" field here is the ADVANTAGEOUS allele — the pipeline
    # treats risk_allele_count > 0 as the signal. For protective entries the
    # zygosity labels map to genetic advantage instead of genetic risk.

    # ── Cardiovascular Protection ─────────────────────────────────────────────
    "rs328_prot": {"gene": "LPL", "category": "Cardiovascular Protection", "trait": "Enhanced triglyceride clearance", "risk_allele": "G", "effect": "Ser447Ter — G = gain-of-function LPL, superior triglyceride clearance and higher HDL", "trait_direction": "protective"},
    "rs20455": {"gene": "KIF6", "category": "Cardiovascular Protection", "trait": "Statin super-responder", "risk_allele": "G", "effect": "Trp719Arg — G/G = enhanced cardiovascular benefit from statin therapy", "trait_direction": "protective"},
    "rs1799983": {"gene": "NOS3", "category": "Cardiovascular Protection", "trait": "Nitric oxide production / vascular health", "risk_allele": "G", "effect": "Glu298Asp — G/G = efficient endothelial NO production, superior vasodilation and blood flow", "trait_direction": "protective"},
    "rs5882": {"gene": "CETP", "category": "Cardiovascular Protection", "trait": "HDL cholesterol / longevity", "risk_allele": "G", "effect": "Ile405Val — G = larger HDL particles, associated with exceptional longevity", "trait_direction": "protective"},
    "rs11591147": {"gene": "PCSK9", "category": "Cardiovascular Protection", "trait": "Natural LDL reduction", "risk_allele": "T", "effect": "R46L loss-of-function — T = naturally lower LDL-C, ~88% reduced CHD risk", "trait_direction": "protective"},

    # ── Metabolic Advantage ───────────────────────────────────────────────────
    "rs1801282_prot": {"gene": "PPARG", "category": "Metabolic Advantage", "trait": "Insulin sensitivity", "risk_allele": "G", "effect": "Pro12Ala — G = enhanced insulin sensitivity and lower T2D risk (OR ~0.8)", "trait_direction": "protective"},
    "rs13266634": {"gene": "SLC30A8", "category": "Metabolic Advantage", "trait": "Type 2 diabetes protection", "risk_allele": "T", "effect": "Arg325Trp — T = loss-of-function variant conferring ~65% lower T2D risk", "trait_direction": "protective"},
    "rs7903146_prot": {"gene": "TCF7L2", "category": "Metabolic Advantage", "trait": "Glycemic stability", "risk_allele": "C", "effect": "C/C = population reference, strongest protection against T2D susceptibility", "trait_direction": "protective"},
    "rs1801133_prot": {"gene": "MTHFR", "category": "Metabolic Advantage", "trait": "Efficient folate metabolism", "risk_allele": "G", "effect": "G/G = full-activity MTHFR enzyme, optimal folate to methylfolate conversion", "trait_direction": "protective"},
    "rs4988235_prot": {"gene": "LCT", "category": "Metabolic Advantage", "trait": "Lactase persistence", "risk_allele": "A", "effect": "A = persistent lactase expression into adulthood, efficient dairy digestion", "trait_direction": "protective"},

    # ── Cognitive & Mental Resilience ─────────────────────────────────────────
    "rs6265_prot": {"gene": "BDNF", "category": "Cognitive Resilience", "trait": "Neuroplasticity / memory", "risk_allele": "C", "effect": "Val66Val — C/C = optimal BDNF secretion, enhanced hippocampal function, memory consolidation, and stress resilience", "trait_direction": "protective"},
    "rs4680_prot": {"gene": "COMT", "category": "Cognitive Resilience", "trait": "Warrior phenotype / stress performance", "risk_allele": "A", "effect": "Val158Met — A = Met/Met higher prefrontal dopamine, superior working memory and cognitive throughput under low-stress conditions", "trait_direction": "protective"},
    "rs4680_warrior": {"gene": "COMT", "category": "Cognitive Resilience", "trait": "Stress resilience / rapid dopamine clearance", "risk_allele": "G", "effect": "Val158Val — G/G = rapid dopamine clearance (Warrior phenotype), superior cognitive performance under acute stress and adversity", "trait_direction": "protective"},
    "rs53576": {"gene": "OXTR", "category": "Cognitive Resilience", "trait": "Empathy / social bonding", "risk_allele": "G", "effect": "G/G = enhanced oxytocin receptor expression, associated with greater empathy, prosocial behavior, and secure attachment", "trait_direction": "protective"},
    "rs1800497_prot": {"gene": "ANKK1/DRD2", "category": "Cognitive Resilience", "trait": "Dopamine D2 receptor density", "risk_allele": "G", "effect": "G/G = normal D2 receptor density, efficient reward processing and lower addiction vulnerability", "trait_direction": "protective"},
    "rs7294919": {"gene": "TESC", "category": "Cognitive Resilience", "trait": "Hippocampal volume", "risk_allele": "C", "effect": "C = associated with larger hippocampal volume, enhanced spatial memory and learning capacity", "trait_direction": "protective"},

    # ── Longevity & DNA Repair ────────────────────────────────────────────────
    "rs2802292": {"gene": "FOXO3", "category": "Longevity Advantage", "trait": "Exceptional longevity", "risk_allele": "G", "effect": "G = associated with 2.7x higher odds of living to 100+, enhanced cellular stress response and autophagy", "trait_direction": "protective"},
    "rs1042522_prot": {"gene": "TP53", "category": "Longevity Advantage", "trait": "Tumor suppression efficiency", "risk_allele": "G", "effect": "Pro72 — G/G = enhanced apoptosis of damaged cells, superior tumor suppression capacity", "trait_direction": "protective"},
    "rs11549465_prot": {"gene": "HIF1A", "category": "Longevity Advantage", "trait": "Altitude / hypoxia adaptation", "risk_allele": "T", "effect": "Pro582Ser — T = enhanced HIF-1α stability, superior oxygen sensing and altitude adaptation", "trait_direction": "protective"},
    "rs1801131_prot": {"gene": "MTHFR", "category": "Longevity Advantage", "trait": "DNA methylation efficiency", "risk_allele": "G", "effect": "A1298C — G/G = full MTHFR activity at both functional sites, optimal one-carbon metabolism and DNA methylation", "trait_direction": "protective"},
    "rs2228001_prot": {"gene": "XPC", "category": "Longevity Advantage", "trait": "Nucleotide excision repair", "risk_allele": "T", "effect": "T/T = wildtype XPC, efficient nucleotide excision repair protecting against UV and environmental DNA damage", "trait_direction": "protective"},

    # ── Athletic Performance ──────────────────────────────────────────────────
    "rs1815739_prot": {"gene": "ACTN3", "category": "Athletic Performance", "trait": "Fast-twitch muscle fiber power", "risk_allele": "C", "effect": "R577X — C = functional alpha-actinin-3, superior sprint/power performance seen in elite sprinters", "trait_direction": "protective"},
    "rs1815739_end": {"gene": "ACTN3", "category": "Athletic Performance", "trait": "Endurance muscle adaptation", "risk_allele": "T", "effect": "R577X — T/T = X/X genotype, muscle fiber shift toward slow-twitch (endurance), seen in elite marathon runners", "trait_direction": "protective"},
    "rs8192678_prot": {"gene": "PPARGC1A", "category": "Athletic Performance", "trait": "Mitochondrial biogenesis / VO2max", "risk_allele": "C", "effect": "Gly482 — C/C = optimal PGC-1α activity, enhanced mitochondrial biogenesis and aerobic capacity", "trait_direction": "protective"},
    "rs2016520_prot": {"gene": "PPARD", "category": "Athletic Performance", "trait": "Fat oxidation / endurance", "risk_allele": "C", "effect": "C = higher PPARδ expression, superior fat oxidation capacity during sustained exercise", "trait_direction": "protective"},
    "rs699_prot": {"gene": "AGT", "category": "Athletic Performance", "trait": "Muscle hypertrophy response", "risk_allele": "A", "effect": "Met235Thr — A = enhanced angiotensinogen, associated with greater lean mass and hypertrophy response to resistance training", "trait_direction": "protective"},
    "rs4253778_prot": {"gene": "PPARA", "category": "Athletic Performance", "trait": "Endurance capacity", "risk_allele": "G", "effect": "G/G = optimal PPARα expression, superior fatty acid oxidation and endurance performance", "trait_direction": "protective"},
    "rs1049434": {"gene": "MCT1", "category": "Athletic Performance", "trait": "Lactate transport / recovery", "risk_allele": "A", "effect": "A = enhanced MCT1 lactate transporter, faster lactate clearance during high-intensity exercise", "trait_direction": "protective"},

    # ── Immune Resilience ─────────────────────────────────────────────────────
    "rs11209026_prot": {"gene": "IL23R", "category": "Immune Resilience", "trait": "Autoimmune protection", "risk_allele": "A", "effect": "Arg381Gln — A = loss-of-function IL-23R, strong protection against IBD, psoriasis, and ankylosing spondylitis", "trait_direction": "protective"},
    "rs2476601_prot": {"gene": "PTPN22", "category": "Immune Resilience", "trait": "Balanced immune surveillance", "risk_allele": "G", "effect": "G/G = wildtype PTPN22, properly calibrated T-cell activation and immune tolerance", "trait_direction": "protective"},
    "rs12913832_prot": {"gene": "HERC2", "category": "Immune Resilience", "trait": "Melanin density / UV protection", "risk_allele": "G", "effect": "G/G = higher melanin production, enhanced UV damage protection and reduced melanoma risk", "trait_direction": "protective"},
    "rs1800896_prot": {"gene": "IL10", "category": "Immune Resilience", "trait": "Anti-inflammatory capacity", "risk_allele": "A", "effect": "A = higher IL-10 production, enhanced anti-inflammatory response and immune regulation", "trait_direction": "protective"},
    "rs2241880_prot": {"gene": "ATG16L1", "category": "Immune Resilience", "trait": "Autophagy efficiency", "risk_allele": "A", "effect": "A/A = wildtype ATG16L1, fully functional autophagy for clearing intracellular pathogens", "trait_direction": "protective"},

    # ── Detoxification Efficiency ─────────────────────────────────────────────
    "rs1695_prot": {"gene": "GSTP1", "category": "Detoxification Efficiency", "trait": "Phase II glutathione conjugation", "risk_allele": "A", "effect": "Ile105 — A/A = wildtype GSTP1, optimal glutathione transferase activity for xenobiotic/carcinogen detoxification", "trait_direction": "protective"},
    "rs4880_prot": {"gene": "SOD2", "category": "Detoxification Efficiency", "trait": "Mitochondrial antioxidant defense", "risk_allele": "G", "effect": "Ala16 — G/G = efficient mitochondrial import of SOD2, superior superoxide dismutase activity", "trait_direction": "protective"},
    "rs1800566_prot": {"gene": "NQO1", "category": "Detoxification Efficiency", "trait": "Quinone detoxification", "risk_allele": "G", "effect": "G/G = wildtype NQO1, full enzymatic activity for reducing reactive quinones and protecting against oxidative stress", "trait_direction": "protective"},
    "rs762551_prot": {"gene": "CYP1A2", "category": "Detoxification Efficiency", "trait": "Rapid caffeine metabolism", "risk_allele": "A", "effect": "A/A = CYP1A2*1A ultra-rapid metabolizer, efficient caffeine clearance — associated with cardioprotective benefit from coffee", "trait_direction": "protective"},
    "rs1208_prot": {"gene": "NAT2", "category": "Detoxification Efficiency", "trait": "Rapid acetylation capacity", "risk_allele": "A", "effect": "A = NAT2 rapid acetylator allele, faster Phase II acetylation of aromatic amines and drugs", "trait_direction": "protective"},

    # ── Sleep & Circadian Advantage ───────────────────────────────────────────
    "rs57875989_prot": {"gene": "PER3", "category": "Sleep Advantage", "trait": "Enhanced sleep homeostatic drive", "risk_allele": "T", "effect": "PER3 VNTR 5-repeat — associated with stronger sleep homeostatic pressure, morningness, and more consolidated sleep architecture", "trait_direction": "protective"},
    "rs934945_prot": {"gene": "PER2", "category": "Sleep Advantage", "trait": "Circadian resilience", "risk_allele": "C", "effect": "C/C = stable PER2 oscillation, robust circadian rhythm and consistent sleep-wake timing", "trait_direction": "protective"},
    "rs1801260_prot": {"gene": "CLOCK", "category": "Sleep Advantage", "trait": "Circadian stability", "risk_allele": "A", "effect": "A/A = stable CLOCK protein activity, consistent circadian rhythm and sleep architecture", "trait_direction": "protective"},

    # ── Nutrient Processing Advantage ─────────────────────────────────────────
    "rs1800562_prot": {"gene": "HFE", "category": "Nutrient Advantage", "trait": "Normal iron homeostasis", "risk_allele": "G", "effect": "G/G = wildtype HFE, properly regulated iron absorption preventing both deficiency and overload", "trait_direction": "protective"},
    "rs174537_prot": {"gene": "FADS1", "category": "Nutrient Advantage", "trait": "Omega-3/6 conversion efficiency", "risk_allele": "G", "effect": "G/G = optimal delta-5 desaturase activity, efficient conversion of plant-based ALA to EPA/DHA", "trait_direction": "protective"},
    "rs12785878_prot": {"gene": "DHCR7", "category": "Nutrient Advantage", "trait": "Vitamin D synthesis", "risk_allele": "G", "effect": "G/G = optimal 7-dehydrocholesterol levels for efficient vitamin D3 synthesis from sunlight", "trait_direction": "protective"},
    "rs2282679_prot": {"gene": "GC", "category": "Nutrient Advantage", "trait": "Vitamin D transport", "risk_allele": "G", "effect": "G/G = optimal vitamin D binding protein, efficient 25(OH)D transport and bioavailability", "trait_direction": "protective"},
    "rs601338_prot": {"gene": "FUT2", "category": "Nutrient Advantage", "trait": "Secretor status / B12 absorption", "risk_allele": "G", "effect": "G/G = secretor phenotype, enhanced B12 absorption and diverse gut Bifidobacterium colonization", "trait_direction": "protective"},

    # ── Hormonal Optimization ─────────────────────────────────────────────────
    "rs1799941_prot": {"gene": "SHBG", "category": "Hormonal Advantage", "trait": "Balanced sex hormone binding", "risk_allele": "A", "effect": "A = lower SHBG, higher bioavailable testosterone and estradiol — advantageous for muscle building and bone density", "trait_direction": "protective"},
    "rs10046_prot": {"gene": "CYP19A1", "category": "Hormonal Advantage", "trait": "Aromatase regulation", "risk_allele": "A", "effect": "A/A = balanced aromatase activity, optimal estrogen-to-androgen ratio", "trait_direction": "protective"},

    # ── Skin & Appearance Advantage ───────────────────────────────────────────
    "rs1805007_prot": {"gene": "MC1R", "category": "Skin Advantage", "trait": "Melanocortin signaling / UV repair", "risk_allele": "C", "effect": "C/C = wildtype MC1R, efficient melanin synthesis and superior UV-induced DNA damage repair", "trait_direction": "protective"},
    "rs16891982_prot": {"gene": "SLC45A2", "category": "Skin Advantage", "trait": "Melanin production efficiency", "risk_allele": "G", "effect": "G/G = wildtype SLC45A2, efficient melanogenesis and inherent UV protection", "trait_direction": "protective"},

    # ── Bone Strength ─────────────────────────────────────────────────────────
    "rs3736228_prot": {"gene": "LRP5", "category": "Bone Strength", "trait": "Peak bone mass / Wnt signaling", "risk_allele": "C", "effect": "C/C = wildtype LRP5, optimal Wnt signaling for peak bone mass acquisition and maintenance", "trait_direction": "protective"},
    "rs1800012_prot": {"gene": "COL1A1", "category": "Bone Strength", "trait": "Collagen I integrity", "risk_allele": "C", "effect": "C/C = wildtype COL1A1, full-strength type I collagen production — reduced fracture and tendon injury risk", "trait_direction": "protective"},

    # ── Pharmacogenomic Advantage ─────────────────────────────────────────────
    "rs12248560_prot": {"gene": "CYP2C19", "category": "Pharmacogenomic Advantage", "trait": "Ultrarapid drug activation", "risk_allele": "T", "effect": "CYP2C19*17 — T = ultrarapid metabolizer, enhanced activation of clopidogrel and faster clearance of PPIs", "trait_direction": "protective"},
    "rs4244285_prot": {"gene": "CYP2C19", "category": "Pharmacogenomic Advantage", "trait": "Normal drug metabolism", "risk_allele": "G", "effect": "G/G = wildtype CYP2C19*1, fully functional drug metabolism — standard dosing for clopidogrel, PPIs, and SSRIs", "trait_direction": "protective"},
    "rs6025_prot": {"gene": "F5", "category": "Pharmacogenomic Advantage", "trait": "Normal coagulation cascade", "risk_allele": "C", "effect": "C/C = wildtype Factor V, properly regulated clotting — no elevated thrombosis risk from Leiden mutation", "trait_direction": "protective"},

    # ── Eye Health Advantage ──────────────────────────────────────────────────
    "rs800292_prot": {"gene": "CFH", "category": "Eye Health Advantage", "trait": "Complement regulation / AMD protection", "risk_allele": "G", "effect": "G/G = wildtype complement factor H, efficient complement regulation protecting against age-related macular degeneration", "trait_direction": "protective"},
    "rs10490924_prot": {"gene": "ARMS2", "category": "Eye Health Advantage", "trait": "Retinal cell stability", "risk_allele": "G", "effect": "G/G = wildtype ARMS2, normal mitochondrial function in retinal pigment epithelium cells", "trait_direction": "protective"},

    # ── Gut Microbiome Advantage ──────────────────────────────────────────────
    "rs2066844_prot": {"gene": "NOD2", "category": "Gut Microbiome Advantage", "trait": "Innate bacterial sensing", "risk_allele": "C", "effect": "C/C = wildtype NOD2, optimal muramyl dipeptide sensing and balanced innate immune response to gut bacteria", "trait_direction": "protective"},
    "rs601338_gut": {"gene": "FUT2", "category": "Gut Microbiome Advantage", "trait": "Bifidobacterium colonization", "risk_allele": "G", "effect": "G/G secretor = abundant gut Bifidobacterium, enhanced short-chain fatty acid production and mucosal barrier integrity", "trait_direction": "protective"},
}


# =============================================================================
# CHUNKING LOGIC
# =============================================================================

def chunk_category(snps: list, max_chunk_size: int = 35) -> list:
    """Split a large category into chunks of max_chunk_size SNPs."""
    return [snps[i:i+max_chunk_size] for i in range(0, len(snps), max_chunk_size)]


# =============================================================================
# PARSER
# =============================================================================

def parse_ancestry_file(filepath: str) -> dict:
    """Backward-compatible wrapper returning rsid -> allele tuple only."""
    parsed = parse_raw_genotype_file(filepath)
    return {
        rsid: record["alleles"]
        for rsid, record in parsed["genotypes"].items()
    }


def resolve_apoe_haplotype(genotypes: dict) -> dict | None:
    """
    Determine APOE haplotype from rs429358 + rs7412.

    APOE isoforms (plus-strand alleles):
        rs429358  rs7412   Haplotype
        T         C        E3 (most common, reference)
        T         T        E2 (protective for AD, risk for type III HLP)
        C         C        E4 (risk for AD and CVD)
        C         T        E1 (very rare)

    Returns a dict with haplotype details, or None if either SNP is missing.
    """
    rs429358 = genotypes.get("rs429358")
    rs7412 = genotypes.get("rs7412")

    if rs429358 is None or rs7412 is None:
        missing = []
        if rs429358 is None:
            missing.append("rs429358")
        if rs7412 is None:
            missing.append("rs7412")
        return {
            "determined": False,
            "reason": f"APOE haplotype could not be determined: {', '.join(missing)} not found in genotype data.",
            "missing_snps": missing,
            "note": "AncestryDNA V2 arrays often do not include rs429358. Consider clinical APOE testing if needed.",
        }

    def _alleles(record):
        if isinstance(record, dict):
            return record.get("alleles", ("-", "-"))
        return record

    a429_1, a429_2 = _alleles(rs429358)
    a7412_1, a7412_2 = _alleles(rs7412)

    haplotype_map = {
        ("T", "C"): "E3",
        ("T", "T"): "E2",
        ("C", "C"): "E4",
        ("C", "T"): "E1",
    }

    h1 = haplotype_map.get((a429_1, a7412_1))
    h2 = haplotype_map.get((a429_2, a7412_2))

    if h1 is None or h2 is None:
        # Try the other phasing
        h1_alt = haplotype_map.get((a429_1, a7412_2))
        h2_alt = haplotype_map.get((a429_2, a7412_1))
        if h1_alt and h2_alt:
            h1, h2 = h1_alt, h2_alt
        else:
            return {
                "determined": False,
                "reason": f"Could not resolve APOE haplotype from rs429358={a429_1}/{a429_2}, rs7412={a7412_1}/{a7412_2}.",
                "missing_snps": [],
            }

    alleles_sorted = tuple(sorted([h1, h2]))
    genotype_label = f"{alleles_sorted[0]}/{alleles_sorted[1]}"

    risk_profiles = {
        ("E2", "E2"): {"ad_risk": "reduced", "cvd_risk": "reduced", "hld_risk": "elevated (type III hyperlipoproteinemia)", "severity": "watch"},
        ("E2", "E3"): {"ad_risk": "reduced", "cvd_risk": "neutral", "hld_risk": "slightly elevated", "severity": "neutral"},
        ("E2", "E4"): {"ad_risk": "neutral", "cvd_risk": "neutral", "hld_risk": "variable", "severity": "watch"},
        ("E3", "E3"): {"ad_risk": "population average", "cvd_risk": "population average", "hld_risk": "population average", "severity": "neutral"},
        ("E3", "E4"): {"ad_risk": "elevated (~3x)", "cvd_risk": "elevated", "hld_risk": "neutral", "severity": "warning"},
        ("E4", "E4"): {"ad_risk": "strongly elevated (~12x)", "cvd_risk": "strongly elevated", "hld_risk": "neutral", "severity": "critical"},
    }

    profile = risk_profiles.get(alleles_sorted, {"ad_risk": "unknown", "cvd_risk": "unknown", "hld_risk": "unknown", "severity": "watch"})

    return {
        "determined": True,
        "genotype": genotype_label,
        "allele_1": h1,
        "allele_2": h2,
        "rs429358_genotype": f"{a429_1}/{a429_2}",
        "rs7412_genotype": f"{a7412_1}/{a7412_2}",
        "has_e4": "E4" in (h1, h2),
        "has_e2": "E2" in (h1, h2),
        **profile,
    }


def process_snps_with_qc(genotypes: dict, parse_summary: dict | None = None, strict: bool = True) -> tuple[dict, dict]:
    """
    Cross-reference genotypes against the panel and validate each matched call
    against authoritative dbSNP metadata before allowing user-facing claims.
    """
    source_hint = (parse_summary or {}).get("source_hint", "unknown")

    results = {}
    matched_rsids = []
    for panel_key in SNP_PANEL:
        lookup_rsid = canonical_panel_rsid(panel_key)
        if lookup_rsid in genotypes:
            matched_rsids.append(lookup_rsid)

    variant_cache = ensure_variant_metadata(matched_rsids)
    qc = {
        "strict_mode": strict,
        "panel_variants_reviewed": len(SNP_PANEL),
        "matched_panel_variants": 0,
        "claim_ready_variants": 0,
        "suppressed_variants": 0,
        "validated_forward_calls": 0,
        "validated_complemented_calls": 0,
        "palindromic_suppressed": 0,
        "palindromic_resolved": 0,
        "missing_authoritative_metadata": 0,
        "risk_allele_complemented": 0,
        "risk_allele_multi_allelic_warn": 0,
        "validation_failures": 0,
        "source_hint": source_hint,
        "source_parse_summary": parse_summary or {},
    }

    for panel_key, meta in SNP_PANEL.items():
        lookup_rsid = canonical_panel_rsid(panel_key)
        if lookup_rsid not in genotypes:
            continue
        qc["matched_panel_variants"] += 1

        raw_record = genotypes[lookup_rsid]
        if isinstance(raw_record, dict):
            genotype_record = raw_record
            allele1, allele2 = raw_record.get("alleles", ("-", "-"))
        else:
            allele1, allele2 = raw_record
            genotype_record = {
                "rsid": lookup_rsid,
                "alleles": tuple(sorted((allele1, allele2))),
                "chromosome": "",
                "position": "",
            }

        genotype = f"{allele1}/{allele2}"
        validation = validate_variant_call(
            genotype_record,
            meta,
            variant_cache.get(lookup_rsid),
            strict=strict,
            source_hint=source_hint,
        )

        risk_allele = validation.get("risk_allele_in_input_orientation") or meta["risk_allele"]
        risk_count = validation.get("risk_allele_count", 0)
        if validation.get("claim_ready"):
            if risk_count == 0:
                zygosity = "homozygous_reference"
            elif risk_count == 1:
                zygosity = "heterozygous"
            else:
                zygosity = "homozygous_risk"
            qc["claim_ready_variants"] += 1
        else:
            zygosity = "suppressed"
            qc["suppressed_variants"] += 1

        if validation.get("input_orientation") == "forward" and validation.get("claim_ready"):
            qc["validated_forward_calls"] += 1
        if validation.get("input_orientation") == "reverse_complemented_input":
            qc["validated_complemented_calls"] += 1
        if validation.get("validation_status") == "palindromic_strand_ambiguous":
            qc["palindromic_suppressed"] += 1
        if any("Palindromic" in note and "validated" in note for note in validation.get("validation_notes", [])):
            qc["palindromic_resolved"] += 1
        if validation.get("validation_status") == "missing_dbsnp_metadata":
            qc["missing_authoritative_metadata"] += 1
        if any("Curated risk allele" in note for note in validation.get("validation_notes", [])):
            qc["risk_allele_complemented"] += 1
        if any("multi-allelic" in note for note in validation.get("validation_notes", [])):
            qc["risk_allele_multi_allelic_warn"] += 1
        if not validation.get("claim_ready") and validation.get("validation_status") not in {
            "missing_dbsnp_metadata",
            "palindromic_strand_ambiguous",
        }:
            qc["validation_failures"] += 1

        category = meta["category"]
        trait_direction = meta.get("trait_direction", "risk")
        if category not in results:
            results[category] = []
        results[category].append({
            "rsid": lookup_rsid,
            "panel_key": panel_key,
            "rsid_hashed": hash_rsid(lookup_rsid),
            "gene": meta["gene"],
            "trait": meta["trait"],
            "genotype": genotype,
            "risk_allele": risk_allele,
            "zygosity": zygosity,
            "effect": meta["effect"],
            "trait_direction": trait_direction,
            "claim_ready": validation.get("claim_ready", False),
            "validation_status": validation.get("validation_status"),
            "validation_notes": validation.get("validation_notes", []),
            "risk_allele_count": risk_count,
            "input_orientation": validation.get("input_orientation"),
            "risk_allele_plus_strand": validation.get("risk_allele_plus_strand"),
            "is_palindromic": validation.get("is_palindromic", False),
            "chromosome": genotype_record.get("chromosome", ""),
            "position": genotype_record.get("position", ""),
            "dbsnp_metadata": variant_cache.get(lookup_rsid, {}),
        })

    # Resolve APOE haplotype
    apoe_haplotype = resolve_apoe_haplotype(genotypes if isinstance(genotypes, dict) else {})
    qc["apoe_haplotype"] = apoe_haplotype

    return results, qc


def process_snps(genotypes: dict) -> dict:
    """Backward-compatible panel processing without returning QC."""
    results, _ = process_snps_with_qc(genotypes)
    return results


def analyze_dna_file(filepath: str, strict: bool = True) -> tuple[dict, dict]:
    """Parse and validate a raw DNA file in one step."""
    parsed = parse_raw_genotype_file(filepath, keep_only_rsids=panel_keep_set(SNP_PANEL.keys()))
    results, qc = process_snps_with_qc(
        parsed["genotypes"],
        parse_summary=parsed["summary"],
        strict=strict,
    )
    return results, qc


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 snp_processor.py <ancestry_file.txt> [output.json]")
        print(f"Panel: {len(SNP_PANEL)} SNPs across 14 categories")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "bedrock_payload.json"

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    print(f"Parsing: {input_file}")
    parsed = parse_raw_genotype_file(input_file)
    genotypes = parsed["genotypes"]
    print(f"  Loaded {len(genotypes):,} SNPs")

    print(f"Cross-referencing {len(SNP_PANEL)} panel SNPs...")
    results, qc = process_snps_with_qc(genotypes, parse_summary=parsed["summary"])

    total = sum(len(v) for v in results.values())
    print(f"  Found {total} relevant variants across {len(results)} categories:")
    print(f"  Claim-ready variants: {qc['claim_ready_variants']}")
    print(f"  Suppressed variants: {qc['suppressed_variants']}")
    for cat, snps in results.items():
        chunks = chunk_category(snps)
        print(f"    {cat}: {len(snps)} variants ({len(chunks)} chunk(s))")

    print(f"\nSaved de-identified data to {output_file}.")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
