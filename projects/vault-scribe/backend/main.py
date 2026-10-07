"""
Vault Scribe — Main API Application
FastAPI server that ties together the indexer, classifier, linker, and ghost checker,
and serves the frontend web UI.
"""

from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import VAULT_PATH, HOST, PORT
from backend.indexer import get_indexer, rebuild_index
from backend.classifier import classify_content, suggest_title
from backend.linker import find_links
from backend.ghost_checker import check_ghosts, get_vault_ghosts
from backend.duplicate_detector import check_duplicates
from backend.retroactive import find_retroactive_links, apply_retroactive_patches
from backend.templates import TEMPLATE_REGISTRY
from backend.scraper import scrape_url
from backend.gap_detector import detect_knowledge_gaps
from backend.experiments import get_active_experiments, log_experiment_data
from backend.llm import chat_with_scribe, generate_insights
from backend.duplicate_detector import check_duplicates

# ── Application Setup ────────────────────────────────────────────────────────
app = FastAPI(title="Vault Scribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Models ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    content: str
    template_override: Optional[str] = None


class ScrapeRequest(BaseModel):
    url: str


class PreviewRequest(BaseModel):
    content: str
    title: str
    template: str
    folder: str
    selected_links: list[dict[str, Any]]
    retroactive_patches: list[dict[str, Any]] = []


class CommitRequest(BaseModel):
    title: str
    content: str
    folder: str
    template: str
    selected_links: list[dict[str, Any]]
    retroactive_patches: list[dict[str, Any]] = []


class InnervateScanRequest(BaseModel):
    title: str

class InnervateApplyRequest(BaseModel):
    patches: list[dict[str, Any]]


class LogExperimentRequest(BaseModel):
    title: str
    notes: str
    metric: str = "—"
    subjective: str = "—"

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


class ContradictionRequest(BaseModel):
    title: str
    content: str

class LibrarianRequest(BaseModel):
    query: str

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/index")
def get_vault_index_stats():
    """Return stats about the currently indexed vault."""
    indexer = get_indexer()
    return indexer.get_stats()


@app.get("/api/notes")
def get_all_notes():
    """Return a list of all indexed notes (titles)."""
    indexer = get_indexer()
    return {"notes": list(indexer.index.notes.keys())}


@app.get("/api/gaps")
def get_knowledge_gaps():
    """Return a list of ghost hubs (highly linked missing notes)."""
    gaps = detect_knowledge_gaps(min_mentions=2)
    return {"gaps": [g.__dict__ for g in gaps]}


@app.get("/api/experiments")
def list_experiments():
    """Return a list of active experiments."""
    return {"experiments": get_active_experiments()}


@app.post("/api/innervate/apply")
def apply_innervate(req: InnervateApplyRequest):
    """Apply selected retroactive patches."""
    try:
        results = apply_retroactive_patches(req.patches)
        rebuild_index()
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_vault(q: str):
    """Search the vault by title and content."""
    indexer = get_indexer()
    results = []
    seen = set()
    
    # Title matches
    for title, score in indexer.search_titles(q):
        note = indexer.get_note(title)
        if note:
            results.append({
                "title": title,
                "folder": note.folder,
                "preview": note.first_paragraph,
                "score": score,
                "type": "title"
            })
            seen.add(title)
            
    # Content matches
    term_lower = q.lower()
    for title, note in indexer.index.notes.items():
        if title not in seen:
            if term_lower in note.content.lower():
                results.append({
                    "title": title,
                    "folder": note.folder,
                    "preview": note.first_paragraph,
                    "score": 50,
                    "type": "content"
                })
    return {"results": results[:50]}

class SaveReportRequest(BaseModel):
    query: str
    report: str

@app.post("/api/librarian/save")
def save_librarian_report(req: SaveReportRequest):
    """Save a Librarian report to the vault."""
    import time
    from backend.config import VAULT_PATH
    
    clean_query = "".join(c if c.isalnum() else "-" for c in req.query).strip("-")[:50]
    filename = f"{clean_query}-{int(time.time())}.md"
    
    reports_dir = VAULT_PATH / "knowledge" / "reference" / "librarian-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = reports_dir / filename
    
    content = f"# Librarian Report: {req.query}\n\n> Parent: [[Reference Library]]\n> Type: Auto-Generated Report\n\n---\n\n{req.report}"
    file_path.write_text(content, encoding="utf-8")
    rebuild_index()
    return {"status": "success", "filepath": str(file_path.relative_to(VAULT_PATH))}

@app.post("/api/innervate/batch")
def batch_innervate(skip: int = 0, limit: int = 50):
    """Scan the entire vault for retroactive links with pagination (Fast version)."""
    indexer = get_indexer()
    index = indexer.index
    
    titles = sorted([t for t in index.all_titles if len(t) > 4], key=len, reverse=True)
    note_items = sorted(list(index.notes.items()), key=lambda x: x[0])
    
    found_count = 0
    all_patches = []
    import re
    
    for note_title, note in note_items:
        content = note.content
        content_lower = content.lower()
        
        # Fast string match to pre-filter titles that exist in this note
        possible_titles = [t for t in titles if t.lower() in content_lower and t not in note.outgoing_links and t != note_title]
        
        if not possible_titles:
            continue
            
        lines = content.split("\n")
        for t in possible_titles:
            pattern = re.compile(r'(?<!\[\[)' + re.escape(t) + r'(?!\]\])', re.IGNORECASE)
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith(">") or stripped.startswith("#") or stripped == "---":
                    continue
                match = pattern.search(line)
                if match:
                    before_match = line[:match.start()]
                    if before_match.count("[[") > before_match.count("]]"):
                        continue
                    
                    found_count += 1
                    if found_count > skip and len(all_patches) < limit:
                        original = match.group(0)
                        rep = f"[[{t}]]" if original == t else f"[[{t}|{original}]]"
                        all_patches.append({
                            "target_note_title": note_title,
                            "target_note_path": note.filepath,
                            "proposed_replacement": rep,
                            "context_line": line.strip(),
                            "line_number": line_num
                        })
                    
                    if len(all_patches) >= limit:
                        break
            if len(all_patches) >= limit:
                break
        if len(all_patches) >= limit:
            break
            
    return {"patches": all_patches, "skip": skip + len(all_patches), "has_more": len(all_patches) == limit}

@app.post("/api/ghosts/extinct")
def extinct_ghosts():
    """Remove all ghost links from the vault."""
    indexer = get_indexer()
    ghosts = get_vault_ghosts()
    
    patched_count = 0
    from backend.config import VAULT_PATH
    from backend.ghost_checker import remove_ghost_links
    
    for ghost_title, referencing_notes in ghosts.items():
        for note_title in referencing_notes:
            note = indexer.get_note(note_title)
            if note:
                note_path = note.filepath
                try:
                    content = note_path.read_text(encoding="utf-8")
                    new_content = remove_ghost_links(content, [ghost_title])
                    if new_content != content:
                        note_path.write_text(new_content, encoding="utf-8")
                        patched_count += 1
                except Exception:
                    pass
                    
    rebuild_index()
    return {"status": "success", "extincted_count": patched_count}


@app.post("/api/experiments/log")
def log_experiment(req: LogExperimentRequest):
    """Append a log entry to an active experiment."""
    try:
        new_row = log_experiment_data(req.title, req.notes, req.metric, req.subjective)
        return {"status": "success", "appended_row": new_row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def process_chat(req: ChatRequest):
    """Send conversation history to LLM and get response."""
    try:
        reply = await chat_with_scribe(req.messages)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/insights/generate")
async def process_insights():
    """Generate cross-domain insights."""
    try:
        insight = await generate_insights()
        return {"insight": insight}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index/rebuild")
def trigger_index_rebuild():
    """Force a rebuild of the vault index."""
    rebuild_index()
    return {"status": "success", "stats": get_indexer().get_stats()}


@app.post("/api/scrape")
def fetch_url_content(req: ScrapeRequest):
    """Fetch text content from a given URL."""
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is empty")
    
    result = scrape_url(req.url.strip())
    
    if result.error:
        raise HTTPException(status_code=500, detail=f"Failed to scrape: {result.error}")
        
    return {
        "title": result.title,
        "content": result.content,
        "source_url": result.source_url
    }


@app.post("/api/analyze")
def analyze_raw_content(req: AnalyzeRequest):
    """
    Core intake pipeline:
    1. Classify content → folder & template
    2. Suggest title
    3. Find potential wikilinks
    4. Check for duplicates
    5. Check for ghost links in the content
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content is empty")

    # 1. Classification
    classification = classify_content(req.content, req.template_override)

    # 2. Title Suggestion
    title = suggest_title(req.content, classification.category)

    # 3. Duplicate Detection
    duplicates_res = check_duplicates(title, req.content, classification.folder)

    # 4. Link Detection
    links_res = find_links(req.content, classification.category, exclude_title=title)

    # 5. Ghost Check (on raw content before auto-linking, but UI will check again after)
    ghosts_res = check_ghosts(req.content)

    # 6. Retroactive Link Check (What notes should link to this new note?)
    retro_res = find_retroactive_links(title)

    return {
        "suggested_title": title,
        "classification": classification.__dict__,
        "duplicates": [d.__dict__ for d in duplicates_res.duplicates],
        "has_duplicates": duplicates_res.has_duplicates,
        "suggested_links": [s.__dict__ for s in links_res.suggested_links],
        "parent_suggestion": links_res.parent_suggestion,
        "ghosts": [g.__dict__ for g in ghosts_res.ghost_links],
        "retroactive_patches": [p.__dict__ for p in retro_res.patches],
    }


@app.post("/api/analyze/contradictions")
async def process_contradictions(req: ContradictionRequest):
    """Check if the draft contradicts the vault."""
    try:
        from backend.llm import check_contradictions
        result = await check_contradictions(req.content, req.title)
        has_contradiction = "No clear contradictions found." not in result
        return {
            "has_contradiction": has_contradiction,
            "message": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/librarian/query")
async def query_librarian(req: LibrarianRequest):
    """Run GraphRAG Librarian pipeline."""
    try:
        from backend.librarian_engine import generate_librarian_report
        result = await generate_librarian_report(req.query)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preview")
def generate_preview(req: PreviewRequest):
    """Generate the final markdown as it will be written to disk."""
    if req.template not in TEMPLATE_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown template: {req.template}")

    # Build related notes from selected links
    related_notes = [link["target_title"] for link in req.selected_links if link.get("included", True)]

    # Get the template function
    template_func = TEMPLATE_REGISTRY[req.template]

    # Build kwargs based on template type
    kwargs = {}
    if req.template == "snp":
        # Extract rsid and gene from title "rs123 — GENE"
        parts = req.title.split(" — ")
        kwargs["rsid"] = parts[0] if parts else req.title
        kwargs["gene"] = parts[1] if len(parts) > 1 else "Unknown"
        kwargs["clinical_significance"] = req.content
        kwargs["related_notes"] = related_notes
    elif req.template == "biochemistry":
        kwargs["title"] = req.title
        kwargs["content"] = req.content
        kwargs["cross_references"] = related_notes
    elif req.template == "fitness":
        kwargs["summary"] = req.content
    elif req.template == "neuroacoustics":
        kwargs["title"] = req.title
        kwargs["content"] = req.content
        kwargs["cross_references"] = related_notes
    elif req.template == "nutrition":
        kwargs["title"] = req.title
        kwargs["content"] = req.content
    elif req.template == "pharmacogenomics":
        kwargs["title"] = req.title
        kwargs["metabolizer_status"] = req.content
        kwargs["related_notes"] = related_notes
    elif req.template == "graph":
        kwargs["claim_title"] = req.title
        kwargs["claim"] = req.title
        kwargs["mechanism"] = req.content
        kwargs["linked_notes"] = related_notes
    elif req.template == "experiment":
        # Extract experiment title from "Experiment: ..." prefix
        exp_title = req.title.replace("Experiment: ", "").strip() or req.title
        kwargs["title"] = exp_title
        kwargs["intervention"] = exp_title
        kwargs["hypothesis"] = req.content
        kwargs["linked_research"] = related_notes
    elif req.template == "hub":
        kwargs["title"] = req.title
        kwargs["description"] = req.content[:200] if req.content else ""
        kwargs["cross_references"] = related_notes
    else:
        # Generic reference fallback
        kwargs["title"] = req.title
        kwargs["content"] = req.content
        kwargs["cross_references"] = related_notes

    try:
        rendered = template_func(**kwargs)
        
        # Safety check: if template rendered mostly empty, fall back to raw content with a header
        stripped = rendered.replace("#", "").replace(">", "").replace("-", "").replace("*", "").replace("|", "").strip()
        content_lines = [l for l in stripped.split("\n") if l.strip() and len(l.strip()) > 5]
        
        if len(content_lines) < 3 and req.content.strip():
            # Template produced essentially nothing — use raw content as fallback
            rendered = f"# {req.title}\n\n{req.content}\n"
            if related_notes:
                rendered += "\n## Related Notes\n" + "\n".join(f"- [[{r}]]" for r in related_notes) + "\n"
        
        return {"rendered_markdown": rendered}
    except Exception as e:
        # If template crashes, still produce a usable note rather than failing
        fallback = f"# {req.title}\n\n{req.content}\n"
        if related_notes:
            fallback += "\n## Related Notes\n" + "\n".join(f"- [[{r}]]" for r in related_notes) + "\n"
        return {"rendered_markdown": fallback}


@app.post("/api/commit")
def commit_to_vault(req: CommitRequest):
    """Write the note to the vault and apply any retroactive patches."""
    try:
        # First generate the exact markdown to save
        preview_res = generate_preview(PreviewRequest(
            content=req.content,
            title=req.title,
            template=req.template,
            folder=req.folder,
            selected_links=req.selected_links,
            retroactive_patches=req.retroactive_patches
        ))
        markdown_content = preview_res["rendered_markdown"]

        # Ensure folder exists
        folder_path = VAULT_PATH / req.folder
        folder_path.mkdir(parents=True, exist_ok=True)

        # Write the file
        safe_title = req.title.replace("/", "-")
        filepath = folder_path / f"{safe_title}.md"
        
        # Check if exists to avoid accidental overwrite
        if filepath.exists():
            # append a suffix if it exists
            import time
            filepath = folder_path / f"{safe_title} ({int(time.time())}).md"

        filepath.write_text(markdown_content, encoding="utf-8")

        # Apply retroactive patches if requested
        patch_results = {}
        if req.retroactive_patches:
            # Filter to only included patches
            included_patches = [p for p in req.retroactive_patches if p.get("included", True)]
            if included_patches:
                patch_results = apply_retroactive_patches(included_patches)

        # Rebuild index in background
        import threading
        threading.Thread(target=rebuild_index).start()

        return {
            "status": "success",
            "filepath": str(filepath.relative_to(VAULT_PATH)),
            "retroactive_patched_count": len([r for r in patch_results.values() if r == "patched"]),
            "patch_results": patch_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Commit failed: {str(e)}")


@app.post("/api/innervate/scan")
def innervate_scan(req: InnervateScanRequest):
    """Scan vault for retroactive links to a specific note."""
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
        
    res = find_retroactive_links(req.title)
    return {
        "patches": [p.__dict__ for p in res.patches],
        "scanned_notes": res.scanned_notes,
        "total_patches": res.total_patches
    }


@app.post("/api/innervate/apply")
def innervate_apply(req: InnervateApplyRequest):
    """Apply a list of retroactive patches."""
    if not req.patches:
        return {"status": "success", "patched_count": 0, "results": {}}
        
    patch_results = apply_retroactive_patches(req.patches)
    
    # Rebuild index since we altered files
    import threading
    threading.Thread(target=rebuild_index).start()
    
    return {
        "status": "success",
        "patched_count": len([r for r in patch_results.values() if r == "patched"]),
        "results": patch_results
    }


def run():
    """Start the FastAPI server."""
    print(f"Starting Vault Scribe on http://{HOST}:{PORT}")
    print(f"Vault path: {VAULT_PATH}")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)

if __name__ == "__main__":
    run()
