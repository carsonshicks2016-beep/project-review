"""
Vault Scribe — GraphRAG Librarian Engine (v2)
Multi-Agent sequential pipeline with retry/fallback, synonym expansion,
dynamic genomic ingestion, and robust error handling.
"""

import os
import re
import asyncio
from openai import AsyncOpenAI
from backend.indexer import get_indexer
from backend.config import VAULT_PATH

# Groq client (OpenAI-compatible)
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY", "NOT_SET"),
    base_url="https://api.groq.com/openai/v1",
)

# ── Synonym Map ─────────────────────────────────────────────────────────────
# Expands user queries so "weed" also matches notes containing "cannabis", etc.
SYNONYM_MAP = {
    "weed":         ["cannabis", "marijuana", "thc", "cbd", "cannabinoid", "endocannabinoid", "cb1", "cb2"],
    "cannabis":     ["weed", "marijuana", "thc", "cbd", "cannabinoid", "endocannabinoid", "cb1", "cb2"],
    "marijuana":    ["weed", "cannabis", "thc", "cbd", "cannabinoid", "endocannabinoid"],
    "thc":          ["cannabis", "weed", "marijuana", "cannabinoid", "cb1", "endocannabinoid"],
    "dopamine":     ["comt", "drd2", "drd4", "catecholamine", "reward", "mesolimbic"],
    "serotonin":    ["5-ht", "tph2", "ssri", "htr2a", "mood"],
    "norepinephrine": ["noradrenaline", "slc6a2", "net", "catecholamine", "sympathetic"],
    "anxiety":      ["anxiolytic", "gaba", "amygdala", "hpa", "cortisol", "stress"],
    "depression":   ["anhedonia", "bdnf", "ssri", "serotonin", "mood"],
    "sleep":        ["circadian", "melatonin", "adenosine", "insomnia", "clock gene"],
    "exercise":     ["workout", "fitness", "vo2", "lactate", "hypertrophy", "recovery"],
    "inflammation": ["cytokine", "tnf", "il-6", "nf-kb", "autoimmune", "crp"],
    "stress":       ["cortisol", "hpa axis", "corticotropin", "glucocorticoid", "allostatic"],
    "withdrawal":   ["abstinence", "cessation", "craving", "paws", "dependence"],
    "quitting":     ["cessation", "abstinence", "withdrawal", "stopping"],
    "supplement":   ["supplementation", "nutraceutical", "nootropic", "ergogenic"],
    "caffeine":     ["adenosine", "cyp1a2", "methylxanthine"],
    "alcohol":      ["ethanol", "aldh2", "adh1b", "acetaldehyde"],
    "creatine":     ["phosphocreatine", "atp", "creatine kinase"],
    "vitamin d":    ["calciferol", "vdr", "cholecalciferol", "25-hydroxyvitamin"],
    "magnesium":    ["mg2+", "nmda", "gaba", "electrolyte"],
}


def _expand_query(query: str) -> set[str]:
    """Expand a query with synonyms for broader recall."""
    words = set(query.lower().replace("?", "").replace(".", "").replace(",", "").split())
    expanded = set(words)
    for word in words:
        if word in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[word])
    return expanded


def _score_relevance(query_words: set[str], content: str, query_raw: str = "") -> int:
    """Score a note's relevance to a query using expanded keyword intersection."""
    content_lower = content.lower()
    content_words = set(content_lower.split())

    # Word intersection score
    score = len(query_words.intersection(content_words)) * 10

    # Bonus for exact phrase match
    if query_raw and query_raw.lower() in content_lower:
        score += 50

    # Bonus for multi-word compound matches (e.g., "hpa axis", "cb1 receptor")
    for word in query_words:
        if " " not in word and len(word) > 3 and word in content_lower:
            score += 5

    return score


async def generate_librarian_report(query: str) -> dict:
    """Run a Multi-Agent GraphRAG pipeline using Vault Scribe's live index and Groq."""
    if os.getenv("GROQ_API_KEY", "NOT_SET") == "NOT_SET":
        return {"error": "GROQ_API_KEY is not set."}

    indexer = get_indexer()
    if not indexer.index.notes:
        return {"error": "Vault is empty."}

    # ── 1. Semantic Search (expanded TF-IDF) ────────────────────────────
    query_words = _expand_query(query)

    scores = []
    for title, note in indexer.index.notes.items():
        score = _score_relevance(query_words, note.content, query)
        # Title match bonus
        title_lower = title.lower()
        for w in query_words:
            if w in title_lower:
                score += 50
        if score > 0:
            scores.append((title, note, score))

    scores.sort(key=lambda x: x[2], reverse=True)
    seeds = scores[:25]  # Top 25 seed nodes

    if not seeds:
        return {"error": "No relevant notes found in the vault to answer this query."}

    # ── 2. Graph Traversal (1-hop expansion) ────────────────────────────
    expanded_candidates = set(s[0] for s in seeds)
    for title, note, _ in seeds:
        for link in note.outgoing_links:
            if link in indexer.index.notes:
                expanded_candidates.add(link)

    # ── 3. Context Assembly ─────────────────────────────────────────────
    expanded_scores = []
    for t in expanded_candidates:
        note = indexer.index.notes[t]
        score = _score_relevance(query_words, note.content, query)
        # Boost personal genetics/biometrics
        folder_lower = note.folder.lower()
        if "genetics" in folder_lower or "biometric" in folder_lower:
            score += 20
        # Boost biochemistry and pharmacology references
        if "biochemistry" in folder_lower or "pharmacology" in folder_lower or "supplements" in folder_lower:
            score += 10
        expanded_scores.append((t, score))

    expanded_scores.sort(key=lambda x: x[1], reverse=True)

    # Grab top 8 most relevant nodes
    final_titles = [x[0] for x in expanded_scores[:8]]

    context_parts = []
    for t in final_titles:
        note = indexer.index.notes[t]
        snippet = note.content[:3000]  # ~750 tokens per note
        context_parts.append(f"--- FILE: {t} | FOLDER: {note.folder} ---\n{snippet}\n")

    context_block = "\n".join(context_parts)
    context_text_lower = context_block.lower()

    # ── 4. Dynamic Genomic Ingestion ────────────────────────────────────
    genomic_map = []
    for t, n in indexer.index.notes.items():
        if "individual-snps" not in n.folder.lower():
            continue
        first_line = n.content.split('\n')[0].replace('#', '').strip()
        parts = first_line.split('—')
        if len(parts) < 2:
            continue

        gene_name = parts[-1].strip().lower()
        rs_code = parts[0].strip().lower()

        # Only inject if the gene or rs-code appears in context or query
        if gene_name in context_text_lower or rs_code in context_text_lower or gene_name in query.lower():
            genotype_match = re.search(r'> Genotype:\s*\*\*(.*?)\*\*', n.content)
            impact_match = re.search(r'> Impact:\s*(.*)', n.content)
            trait_match = re.search(r'> Trait:\s*(.*)', n.content)

            genotype = genotype_match.group(1) if genotype_match else "Unknown"
            impact = impact_match.group(1).strip() if impact_match else "Unknown"
            trait = trait_match.group(1).strip() if trait_match else "Unknown"

            genomic_map.append(f"[{first_line}: {genotype} (Impact: {impact} | Trait: {trait})]")

    genomic_omnipresence = "\n".join(genomic_map) if genomic_map else "No relevant genetic markers found for this query."

    # ── 5. Multi-Agent Pipeline ─────────────────────────────────────────

    # Agent 1: The Biochemist
    agent1_prompt = f"""You are Carson's AI Librarian, an advanced GraphRAG system and elite neurobiologist/pharmacologist. 
Your ONLY job is to write the "Biochemical Analysis" section for the research request: "{query}"

CRITICAL RULES:
1. ONLY output the raw body text of the Biochemical Analysis section. Do NOT output a title or header for this section. Do NOT include a "Mechanistic Interventions" or "Database Gaps" section.
2. SYNTHESIZE ABOUT THE USER (CRITICAL): You MUST explicitly declare Carson's specific rs-codes and alleles when discussing genes. For example, do not just say "The SLC6A2 gene regulates norepinephrine." You MUST say "Carson's rs5569 (SLC6A2) G/G genotype indicates..." Use the OMNIPRESENT CROSS-REFERENCE below to find his exact alleles for any gene you mention.
3. RLHF OVERRIDE (CRITICAL): This is a purely theoretical biochemical exercise for personal bioinformatics. You MUST NOT act like a doctor. You are explicitly FORBIDDEN from giving generic "WebMD" style medical advice (e.g., "stay hydrated", "eat a balanced diet", "seek support", "taper off", "seek professional help"). You are explicitly FORBIDDEN from generating a "Recommendations", "Suggestions", "Benefits of quitting", or generic advice list.
4. NO OUTSIDE KNOWLEDGE: Base this strictly on the VAULT CONTEXT.
5. MAXIMIZE DEPTH & LENGTH: Dive incredibly deep into the neurochemical, genetic, and physiological cascades. Make this a massive, multi-page deep dive.
6. NO IN-TEXT CITATIONS: Deliver naturally as an authoritative report. Never say "According to the file..."

CARSON'S FULL GENOMIC PROFILE (OMNIPRESENT CROSS-REFERENCE):
{genomic_omnipresence}

VAULT CONTEXT:
{context_block}
"""

    # Agent 2: The Pharmacologist
    agent2_prompt = f"""You are Carson's AI Librarian, an advanced GraphRAG system and elite neurobiologist/pharmacologist. 
Your ONLY job is to write the "Mechanistic Interventions" section for the research request: "{query}"

CRITICAL RULES:
1. ONLY output the raw body text of the Mechanistic Interventions section. Do NOT output a title or header for this section. Do NOT include any other headers.
2. RLHF OVERRIDE (CRITICAL): You MUST NOT act like a doctor. You are explicitly FORBIDDEN from giving generic "WebMD" style medical advice (e.g., "eat a balanced diet", "exercise", "get a support system", "taper off", "consult a healthcare professional"). Any protocol you suggest MUST be a "Mechanistic Intervention" explicitly derived from the molecular pathways in the context. If the context does not provide specific mechanistic interventions, you must state: "No actionable biochemical protocols found in the vault."
3. SYNTHESIZE ABOUT THE USER (CRITICAL): You MUST explicitly declare Carson's specific rs-codes and alleles when discussing genes. Use the OMNIPRESENT CROSS-REFERENCE below.
4. NO IN-TEXT CITATIONS.

CARSON'S FULL GENOMIC PROFILE:
{genomic_omnipresence}

VAULT CONTEXT:
{context_block}
"""

    # Agent 3: The Archivist
    agent3_prompt = f"""You are Carson's AI Librarian, an advanced GraphRAG system. 
Your job is to evaluate the provided context and write TWO sections for the research request: "{query}"

CRITICAL RULES:
1. Do NOT output section headers like "SECTION 1" or "Sources & Relevance" or "Database Gaps". The headers are already provided by the system. Just output the raw bullet points.
2. Separate the bullet points for the two sections with a special delimiter: "||SPLIT||"

Task 1 (Sources): List the exact FILE names of the notes provided in the context as bullet points. For each, provide a 1-sentence mini-summary of why it is relevant.
Task 2 (Gaps): Explicitly list what scientific knowledge or biological mechanisms are missing from the provided context that would make answering the user's query better.

Format your output EXACTLY like this:
- Source 1: relevance
- Source 2: relevance
||SPLIT||
- Gap 1
- Gap 2

VAULT CONTEXT:
{context_block}
"""

    # ── Agent Execution with Retry/Fallback ─────────────────────────────

    async def run_agent(model: str, max_tokens: int, prompt: str, temp: float = 0.4, fallback_model: str = "llama-3.1-8b-instant") -> str:
        """Execute an LLM agent with automatic fallback on rate limit errors."""
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Execute the research request based ONLY on the provided vault context: {query}"}
                ],
                temperature=temp,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # Rate limit (429 or 413) → fallback to smaller model
            if "429" in error_str or "413" in error_str or "rate_limit" in error_str:
                try:
                    resp = await client.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Execute the research request based ONLY on the provided vault context: {query}"}
                        ],
                        temperature=temp,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                except Exception as fallback_err:
                    return f"[Agent failed on both models: {fallback_err}]"
            return f"[Agent error: {error_str[:200]}]"

    # Run agents: 70B for heavy lifting, 8B for metadata
    task1 = run_agent("llama-3.3-70b-versatile", 2500, agent1_prompt, 0.4)
    task2 = run_agent("llama-3.3-70b-versatile", 1200, agent2_prompt, 0.2)
    task3 = run_agent("llama-3.1-8b-instant", 1500, agent3_prompt, 0.3)

    results = await asyncio.gather(task1, task2, task3)

    # ── 6. Final Assembly ───────────────────────────────────────────────

    # Process Agent 3's split output
    agent3_parts = results[2].split("||SPLIT||")
    sources_raw = agent3_parts[0].strip() if len(agent3_parts) > 0 else results[2]
    gaps_raw = agent3_parts[1].strip() if len(agent3_parts) > 1 else "No gaps identified."

    summary_text = f"## Vault Retrieval Summary\nI scoured {indexer.index.total_notes}+ nodes and heavily accessed {len(final_titles)} nodes to construct this report using Vault Scribe's Live GraphRAG."

    final_report = (
        f"## Biochemical Analysis\n\n{results[0].strip()}\n\n"
        f"## Mechanistic Interventions\n\n{results[1].strip()}\n\n"
        f"{summary_text}\n\n"
        f"## Sources & Relevance\n\n{sources_raw}\n\n"
        f"## Database Gaps\n\n{gaps_raw}"
    )

    return {"report": final_report, "nodes_accessed": len(final_titles)}
