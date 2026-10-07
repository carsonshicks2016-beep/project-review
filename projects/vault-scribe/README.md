# Vault Scribe

An intelligent note intake system for the `carson-brain` Obsidian vault. 
It analyzes unstructured text, automatically classifies it into the correct vault folder, suggests wikilinks based on your existing knowledge graph, prevents ghost links, detects duplicates, and formats everything exactly to your vault's conventions.

## Architecture

- **Backend**: Python / FastAPI
- **Frontend**: Vanilla HTML / CSS / JS (Single Page App)
- **Matching**: RapidFuzz (fuzzy text matching) & Scikit-learn (TF-IDF similarity)

## Installation

```bash
cd /Users/REVIEW_USER/Desktop/carson-brain/vault-scribe
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
source venv/bin/activate
python -m backend.main
```

Then open `http://127.0.0.1:8484` in your browser.

## Features (Phase 1)

1. **Auto-Classification**: Scans for keywords to determine if a note is genetics, biochemistry, fitness, etc.
2. **Auto-Linking**: Finds concepts you've mentioned that have existing notes and suggests `[[wikilinks]]`.
3. **Ghost Prevention**: Warns you if you create a link to a note that doesn't exist.
4. **Duplicate Detection**: Uses TF-IDF cosine similarity to warn if you're writing about something already in the vault.
5. **Exact Templating**: Replicates your blockquote metadata conventions perfectly.

*Phases 2-4 (Retroactive Innervation, Gaps, LLM Insights) coming soon.*
