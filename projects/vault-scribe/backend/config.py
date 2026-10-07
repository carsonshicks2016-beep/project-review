"""
Vault Scribe — Configuration
Central config for vault paths, folder mappings, classification keywords, and LLM settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Vault Location ──────────────────────────────────────────────────────────
# We assume the user runs this inside their carson-brain repo
VAULT_PATH = Path("/Users/REVIEW_USER/Desktop/carson-brain/carson-brain")

# ── LLM Configuration ──────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")  # "gemini", "anthropic", "openai", or ""
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "5"))  # requests per minute

# ── Folder Mappings ─────────────────────────────────────────────────────────
# Maps category slugs → relative paths within the vault
FOLDER_MAP: dict[str, str] = {
    # Genetics
    "snp":                  "knowledge/genetics/individual-snps",
    "pharmacogenomics":     "knowledge/genetics/pharmacogenomics",
    "genetics-studies":     "knowledge/genetics/studies",
    "genetics-dashboard":   "knowledge/genetics/Dashboards",

    # Reference Library
    "biochemistry":         "knowledge/reference/biochemistry",
    "anatomy":              "knowledge/reference/anatomy",
    "exercise-physiology":  "knowledge/reference/exercise-physiology",
    "pharmacology":         "knowledge/reference/pharmacology",
    "supplements":          "knowledge/reference/supplements",
    "immunology":           "knowledge/reference/immunology",
    "pathophysiology":      "knowledge/reference/pathophysiology",
    "epidemiology":         "knowledge/reference/epidemiology",
    "toxicology":           "knowledge/reference/toxicology",
    "microbiology":         "knowledge/reference/microbiology",
    "molecular-biology":    "knowledge/reference/molecular-biology",
    "gerontology":          "knowledge/reference/gerontology",
    "lab-values":           "knowledge/reference/lab-values",
    "neuroscience":         "knowledge/reference/neuroscience",
    "medical-entomology":   "knowledge/reference/medical-entomology",
    "pubmed":               "knowledge/reference/pubmed",

    # Knowledge Domains
    "fitness":              "knowledge/fitness/sessions",
    "neuroacoustics":       "knowledge/neuroacoustics",
    "nutrition":            "knowledge/nutrition",
    "graph":                "knowledge/graph",

    # Areas
    "lifestyle-coaching":   "areas/lifestyle-coaching",
    "academics":            "areas/academics",
    "ai-research":          "areas/AI-Research",

    # Projects
    "project":              "projects",

    # Default
    "inbox":                "inbox",
}

# ── Parent Note Mappings ────────────────────────────────────────────────────
# Maps category slugs → the Parent wikilink to use in blockquote metadata
PARENT_MAP: dict[str, str] = {
    "snp":                  "[[Full SNP Registry]]",
    "pharmacogenomics":     "[[Drug Interactions]]",
    "biochemistry":         "Biochemistry",
    "anatomy":              "[[Human Anatomy & Organ Systems]]",
    "exercise-physiology":  "[[Exercise Physiology]]",
    "pharmacology":         "[[Pharmacology Reference]]",
    "supplements":          "[[Supplement Evidence Base]]",
    "immunology":           "[[Immunology]]",
    "pathophysiology":      "[[Pathophysiology & Disease]]",
    "epidemiology":         "[[Epidemiology & Biostatistics]]",
    "toxicology":           "[[Toxicology]]",
    "microbiology":         "[[Microbiology & Infectious Disease]]",
    "molecular-biology":    "[[Molecular Biology Techniques]]",
    "gerontology":          "[[Gerontology — The Biology of Aging]]",
    "lab-values":           "[[Clinical Lab Values]]",
    "neuroscience":         "[[Neuroscience]]",
    "neuroacoustics":       "[[_Neuroacoustics Master Index]]",
    "nutrition":            "[[Nutrition Database Index]]",
    "fitness":              "[[workouts]]",
    "graph":                "Knowledge Graph",
    "lifestyle-coaching":   "Lifestyle Coaching",
    "academics":            "Academics",
}

# ── Classification Keywords ─────────────────────────────────────────────────
# Keyword patterns for rule-based classification. Each category has a list of
# (pattern, weight) tuples. The category with the highest weighted score wins.
CLASSIFICATION_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "snp": [
        (r"rs\d{4,}", 10.0),       # rsID pattern — strongest signal
        (r"genotype", 3.0),
        (r"allele", 3.0),
        (r"polymorphism", 3.0),
        (r"variant", 1.5),
        (r"SNP", 5.0),
        (r"evidence score", 4.0),
        (r"clinical significance", 2.0),
        (r"GWAS", 3.0),
        (r"dbSNP", 4.0),
    ],
    "pharmacogenomics": [
        (r"CYP\d\w+", 8.0),        # CYP enzyme pattern
        (r"metabolizer", 6.0),
        (r"drug interaction", 5.0),
        (r"CPIC", 6.0),
        (r"pharmacogenomic", 7.0),
        (r"drug class", 4.0),
        (r"poor metabolizer", 5.0),
        (r"rapid metabolizer", 5.0),
        (r"extensive metabolizer", 5.0),
    ],
    "biochemistry": [
        (r"mTORC[12]", 5.0),
        (r"AMPK", 5.0),
        (r"PI3K", 4.0),
        (r"signaling pathway", 4.0),
        (r"phosphorylation", 4.0),
        (r"kinase", 3.0),
        (r"enzyme", 2.0),
        (r"substrate", 2.5),
        (r"metabolic", 2.0),
        (r"receptor", 2.0),
        (r"ligand", 3.0),
        (r"transduction", 3.0),
        (r"transcription factor", 4.0),
        (r"ATP", 2.5),
        (r"mitochondria", 3.5),
        (r"oxidative phosphorylation", 5.0),
        (r"electron transport", 4.0),
        (r"GLUT4", 5.0),
        (r"insulin signaling", 5.0),
        (r"BDNF", 3.5),
        (r"PGC-1", 4.0),
        (r"autophagy", 4.0),
        (r"anandamide", 4.0),
        (r"cannabinoid", 3.0),
        (r"neurotransmitter", 3.0),
        (r"serotonin", 2.5),
        (r"dopamine", 2.5),
        (r"norepinephrine", 2.5),
        (r"GABA", 3.0),
        (r"glutamate", 3.0),
        (r"cortisol", 2.0),
        (r"HPA axis", 4.0),
        (r"catecholamine", 3.5),
        (r"glucocorticoid", 3.5),
    ],
    "anatomy": [
        (r"anatomy", 3.0),
        (r"organ system", 4.0),
        (r"skeletal", 2.5),
        (r"muscular system", 3.0),
        (r"cardiovascular system", 3.0),
        (r"nervous system", 3.0),
        (r"endocrine gland", 3.0),
    ],
    "exercise-physiology": [
        (r"VO2\s*max", 5.0),
        (r"lactate threshold", 5.0),
        (r"exercise physiology", 5.0),
        (r"cardiac output", 3.0),
        (r"muscle fiber type", 4.0),
        (r"type I fiber", 3.5),
        (r"type II fiber", 3.5),
        (r"aerobic capacity", 3.0),
        (r"anaerobic", 2.5),
    ],
    "pharmacology": [
        (r"pharmacokinetics", 5.0),
        (r"pharmacodynamics", 5.0),
        (r"half-life", 3.0),
        (r"bioavailability", 3.0),
        (r"dose-response", 4.0),
        (r"therapeutic index", 4.0),
        (r"drug metabolism", 4.0),
    ],
    "supplements": [
        (r"supplement", 3.0),
        (r"mg\s*(daily|per day|before bed|morning)", 4.0),
        (r"clinical trial", 2.0),
        (r"RDA", 3.0),
        (r"tolerable upper", 3.0),
        (r"ergogenic", 4.0),
        (r"nootropic", 3.5),
        (r"adaptogen", 3.5),
        (r"creatine", 2.5),
        (r"ashwagandha", 3.0),
        (r"magnesium", 2.0),
        (r"omega-3", 2.5),
        (r"vitamin D", 2.5),
    ],
    "immunology": [
        (r"immune", 2.5),
        (r"cytokine", 4.0),
        (r"T cell", 3.5),
        (r"B cell", 3.5),
        (r"antibody", 3.0),
        (r"antigen", 3.0),
        (r"inflammation", 2.0),
        (r"interleukin", 4.0),
        (r"TNF", 3.5),
        (r"autoimmune", 3.0),
    ],
    "neuroscience": [
        (r"neuroscience", 5.0),
        (r"synaptic plasticity", 5.0),
        (r"LTP", 4.0),
        (r"neurogenesis", 5.0),
        (r"neurotransmitter", 3.0),
        (r"synapse", 2.0),
        (r"action potential", 3.5),
        (r"dendritic", 3.0),
        (r"axon", 2.0),
        (r"cortex", 2.0),
    ],
    "neuroacoustics": [
        (r"Hz", 2.0),
        (r"binaural", 5.0),
        (r"psychoacoustics", 6.0),
        (r"auditory", 3.0),
        (r"frequency", 1.5),
        (r"brainwave", 4.0),
        (r"entrainment", 5.0),
        (r"isochronic", 5.0),
        (r"theta wave", 4.0),
        (r"alpha wave", 4.0),
        (r"beta wave", 4.0),
        (r"delta wave", 4.0),
        (r"gamma wave", 4.0),
        (r"genre", 1.5),
        (r"BPM", 2.0),
        (r"timbre", 3.0),
    ],
    "nutrition": [
        (r"per 100\s*g", 5.0),
        (r"calories", 2.0),
        (r"protein.*g\b", 2.0),
        (r"carbohydrate", 2.0),
        (r"fiber", 1.5),
        (r"macronutrient", 4.0),
        (r"micronutrient", 4.0),
        (r"nutritional profile", 5.0),
        (r"diet", 1.5),
    ],
    "fitness": [
        (r"workout", 4.0),
        (r"sets?\s*[x×]\s*\d+", 5.0),  # "3x10", "4 × 8"
        (r"reps?", 2.0),
        (r"volume\s*:", 3.0),
        (r"total volume", 4.0),
        (r"1RM", 5.0),
        (r"WHOOP", 3.0),
        (r"strain score", 4.0),
        (r"hypertrophy", 2.5),
        (r"RPE", 3.0),
    ],
    "graph": [
        (r"claim:", 3.0),
        (r"mechanism:", 3.0),
        (r"quantitative threshold", 5.0),
        (r"practical implication", 4.0),
    ],
    "epidemiology": [
        (r"epidemiology", 5.0),
        (r"prevalence", 3.0),
        (r"incidence", 3.0),
        (r"odds ratio", 4.0),
        (r"relative risk", 4.0),
        (r"confidence interval", 3.0),
        (r"cohort study", 4.0),
        (r"meta-analysis", 3.5),
    ],
    "gerontology": [
        (r"aging", 2.5),
        (r"senescence", 4.0),
        (r"telomere", 4.0),
        (r"longevity", 3.0),
        (r"lifespan", 3.0),
        (r"healthspan", 4.0),
        (r"gerontology", 5.0),
    ],
    "toxicology": [
        (r"toxicology", 5.0),
        (r"LD50", 5.0),
        (r"toxic", 2.0),
        (r"poison", 2.5),
        (r"carcinogen", 3.5),
        (r"mutagen", 3.5),
    ],
    "lab-values": [
        (r"lab value", 5.0),
        (r"reference range", 5.0),
        (r"normal range", 4.0),
        (r"mg/dL", 3.0),
        (r"mmol/L", 3.0),
        (r"diagnostic", 2.0),
    ],
    "microbiology": [
        (r"microbiome", 3.0),
        (r"bacteria", 2.0),
        (r"viral", 2.0),
        (r"pathogen", 3.0),
        (r"antibiotic", 3.0),
        (r"gram-positive", 4.0),
        (r"gram-negative", 4.0),
    ],
    "molecular-biology": [
        (r"PCR", 3.5),
        (r"CRISPR", 4.0),
        (r"gene expression", 3.5),
        (r"mRNA", 3.0),
        (r"DNA sequencing", 4.0),
        (r"Western blot", 4.0),
        (r"gel electrophoresis", 4.0),
    ],
}

# ── SNP Tag Categories ──────────────────────────────────────────────────────
# Maps trait keywords → the #tag used in SNP notes
SNP_TAG_MAP: dict[str, str] = {
    "athletic": "#athletic-performance",
    "sport": "#athletic-performance",
    "endurance": "#athletic-performance",
    "strength": "#athletic-performance",
    "autoimmune": "#autoimmune-and-inflammation",
    "inflammation": "#autoimmune-and-inflammation",
    "inflammatory": "#autoimmune-and-inflammation",
    "cardiovascular risk": "#cardiovascular-risk",
    "heart disease": "#cardiovascular-risk",
    "cardiovascular protect": "#cardiovascular-protection",
    "cardioprotect": "#cardiovascular-protection",
    "cognitive": "#cognitive-resilience",
    "cognition": "#cognitive-resilience",
    "brain": "#cognitive-resilience",
    "memory": "#cognitive-resilience",
    "detox": "#detoxification",
    "exercise": "#exercise-and-recovery",
    "recovery": "#exercise-and-recovery",
    "eye": "#eye-and-vision-health",
    "vision": "#eye-and-vision-health",
    "macular": "#eye-and-vision-health",
    "gut": "#gut-microbiome",
    "microbiome": "#gut-microbiome",
    "hormone": "#hormonal-health",
    "hormonal": "#hormonal-health",
    "testosterone": "#hormonal-advantage",
    "immune": "#immune-resilience",
    "longevity": "#longevity-and-aging",
    "aging": "#longevity-and-aging",
    "telomere": "#longevity-advantage",
    "mental health": "#mental-health-and-cognition",
    "depression": "#mental-health-and-cognition",
    "anxiety": "#mental-health-and-cognition",
    "nutrition": "#nutrition-and-metabolism",
    "metabolism": "#nutrition-and-metabolism",
    "vitamin": "#nutrition-and-metabolism",
    "pharmacogenomic": "#pharmacogenomics",
    "drug": "#pharmacogenomics",
    "CYP": "#pharmacogenomics",
    "metabolizer": "#pharmacogenomic-advantage",
    "skin": "#skin-and-dermatology",
    "dermatology": "#skin-and-dermatology",
    "sleep": "#sleep-and-circadian",
    "circadian": "#sleep-and-circadian",
    "melatonin": "#sleep-and-circadian",
}

# ── Fuzzy Matching Thresholds ───────────────────────────────────────────────
LINK_MATCH_THRESHOLD = 85       # minimum rapidfuzz score for auto-linking
DUPLICATE_TITLE_THRESHOLD = 80  # minimum score for duplicate title warning
DUPLICATE_CONTENT_THRESHOLD = 0.4  # minimum TF-IDF cosine similarity

# ── Server Configuration ───────────────────────────────────────────────────
HOST = os.getenv("SCRIBE_HOST", "127.0.0.1")
PORT = int(os.getenv("SCRIBE_PORT", "8484"))
