#!/usr/bin/env python3
"""
SYNAPSE — Neural Knowledge Inserter
Fire-and-forget brain loader for the carson-brain Obsidian vault.

Usage:
    python3 synapse.py

Requirements:
    pip3 install flask google-generativeai requests beautifulsoup4
"""

import os, re, json, datetime, webbrowser, threading, time
from pathlib import Path
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
VAULT_PATH  = Path("/Users/REVIEW_USER/Desktop/carson-brain/carson-brain")
GOOGLE_KEY  = os.environ.get("GOOGLE_API_KEY", "")
MODEL       = "gemini-2.0-flash"
PORT        = 7337

VAULT_STRUCTURE = """
Vault root: /Users/REVIEW_USER/Desktop/carson-brain/carson-brain/

Folders (use exactly as written):
  inbox/                          <- catch-all; use when nothing else fits clearly
  areas/lifestyle-coaching/       <- fitness, health, wellness, coaching theory
  areas/academics/                <- university coursework (general)
  areas/academics/biochemistry/   <- biochem coursework and concepts
  areas/academics/PHYS-2414/      <- Physics 1 (completed)
  projects/HealthBridge/          <- HealthBridge platform: build logs, architecture, research
  knowledge/fitness/              <- training theory, exercise science, protocols, programming
  knowledge/neuroacoustics/       <- music, sound, neural entrainment, Spotify context
  knowledge/biochemistry/         <- biochemistry reference, pathways, enzymes
  knowledge/                      <- general educational/reference content

Owner context: Configure your own context locally; personal biography omitted from this export.
"""

ROUTE_PROMPT = """You are SYNAPSE, an intelligent routing system for a personal Obsidian knowledge vault.

{vault_structure}

Analyze the content below and decide how to file it.

Source type: {source_type}
Content:
---
{content}
---

Return ONLY valid JSON with no markdown fences, no preamble, no trailing text:
{{
  "folder": "relative/folder/path",
  "filename": "descriptive-slug.md",
  "content": "# Title\\n\\nFull markdown note body...",
  "summary": "One sentence: what this is and where it is going",
  "tags": ["tag1", "tag2", "tag3"]
}}

Rules:
- folder: pick the best match from the vault structure above; default to inbox/ if uncertain
- filename: kebab-case, no spaces, .md extension, specific and descriptive
- content: write a clean well-structured markdown note with appropriate headings
  - For URLs: extract and synthesize key info; include source URL and date at bottom
  - For raw text/notes: preserve intent, improve formatting, add useful structure
- tags: 2-4 lowercase kebab-case tags
"""

# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = Flask(__name__)
filing_history = []

def get_model():
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_KEY)
    return genai.GenerativeModel(MODEL)

def route_and_write(content: str, source_type: str, override_folder: str = "") -> dict:
    model = get_model()
    prompt = ROUTE_PROMPT.format(
        vault_structure=VAULT_STRUCTURE,
        source_type=source_type,
        content=content[:7000]
    )
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    result = json.loads(raw)

    folder = override_folder.strip() if override_folder.strip() else result["folder"]
    folder = folder.strip("/")

    target_dir = VAULT_PATH / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / result["filename"]

    if target_path.exists():
        stem, suffix = target_path.stem, target_path.suffix
        ts = datetime.datetime.now().strftime("%H%M%S")
        target_path = target_dir / f"{stem}-{ts}{suffix}"

    target_path.write_text(result["content"], encoding="utf-8")
    result["path"]        = str(target_path.relative_to(VAULT_PATH))
    result["timestamp"]   = datetime.datetime.now().strftime("%H:%M:%S")
    result["folder_used"] = folder
    return result


def scrape_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside","iframe","noscript"]):
        tag.decompose()
    title_tag  = soup.find("title") or soup.find("h1")
    title_text = title_tag.get_text(strip=True) if title_tag else url
    main       = soup.find("article") or soup.find("main") or soup.find(id=re.compile(r"content|main|article", re.I))
    body_soup  = main if main else soup
    body       = body_soup.get_text(separator="\n", strip=True)
    body       = "\n".join(line for line in body.splitlines() if line.strip())[:7000]
    return f"URL: {url}\nTitle: {title_text}\n\n{body}"


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/insert", methods=["POST"])
def insert():
    data     = request.json
    mode     = data.get("mode", "note")
    content  = data.get("content", "").strip()
    override = data.get("override_folder", "").strip()
    if not content:
        return jsonify({"error": "Nothing to insert."}), 400
    try:
        if mode == "url":
            raw, source_type = scrape_url(content), f"URL ({content})"
        else:
            raw, source_type = content, "raw text / note"
        result = route_and_write(raw, source_type, override)
        filing_history.insert(0, result)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    return jsonify(filing_history[:20])

@app.route("/vault-status")
def vault_status():
    return jsonify({"vault_exists": VAULT_PATH.exists(), "vault_path": str(VAULT_PATH)})


# ─────────────────────────────────────────────
#  HTML
# ─────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SYNAPSE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #04060f; --surface: #090d1e; --surface2: #0e1428; --border: #1c2840;
    --accent: #00c8ff; --accent2: #7b5ea7; --success: #00ff8c;
    --danger: #ff4d6a; --text: #c5d5ef; --muted: #3e5070; --dimtext: #6a80a0;
  }
  html, body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; font-size: 14px; min-height: 100vh; line-height: 1.6; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }
  .header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2.5rem; }
  .brand { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; letter-spacing: .18em; color: var(--accent); text-shadow: 0 0 28px rgba(0,200,255,.35); }
  .brand span { color: var(--accent2); font-weight: 400; }
  .vault-indicator { display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--dimtext); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); transition: background .4s; }
  .dot.live { background: var(--success); box-shadow: 0 0 8px rgba(0,255,140,.5); }
  .dot.err  { background: var(--danger); }
  .tabs { display: flex; margin-bottom: 1rem; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; width: fit-content; }
  .tab { padding: 7px 22px; font-family: 'DM Mono', monospace; font-size: 12px; letter-spacing: .1em; cursor: pointer; background: transparent; color: var(--dimtext); border: none; transition: all .2s; text-transform: uppercase; }
  .tab:hover { color: var(--text); background: var(--surface); }
  .tab.active { background: var(--surface2); color: var(--accent); border-bottom: 2px solid var(--accent); }
  .input-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: border-color .2s; }
  .input-panel:focus-within { border-color: rgba(0,200,255,.35); box-shadow: 0 0 0 3px rgba(0,200,255,.06); }
  #content { width: 100%; min-height: 130px; background: transparent; border: none; outline: none; color: var(--text); font-family: 'DM Mono', monospace; font-size: 14px; line-height: 1.7; padding: 18px 20px 12px; resize: vertical; caret-color: var(--accent); }
  #content::placeholder { color: var(--muted); font-style: italic; }
  .input-footer { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-top: 1px solid var(--border); gap: 12px; }
  .override-row { display: flex; align-items: center; gap: 8px; flex: 1; }
  .override-label { font-size: 11px; color: var(--dimtext); white-space: nowrap; }
  #override-folder { background: transparent; border: 1px solid var(--border); border-radius: 4px; color: var(--dimtext); font-family: 'DM Mono', monospace; font-size: 12px; padding: 4px 10px; outline: none; width: 180px; transition: all .2s; }
  #override-folder:focus { border-color: var(--accent2); color: var(--text); }
  #override-folder::placeholder { color: var(--muted); }
  .submit-btn { display: flex; align-items: center; gap: 8px; padding: 9px 22px; background: rgba(0,200,255,.08); border: 1px solid rgba(0,200,255,.3); border-radius: 6px; color: var(--accent); font-family: 'DM Mono', monospace; font-size: 12px; letter-spacing: .12em; text-transform: uppercase; cursor: pointer; transition: all .2s; white-space: nowrap; }
  .submit-btn:hover { background: rgba(0,200,255,.14); border-color: var(--accent); box-shadow: 0 0 16px rgba(0,200,255,.12); }
  .submit-btn:active { transform: scale(.97); }
  .submit-btn.loading { opacity: .6; pointer-events: none; }
  .kbd { font-size: 10px; color: var(--muted); border: 1px solid var(--muted); border-radius: 3px; padding: 1px 5px; }
  .status-bar { margin-top: .75rem; height: 18px; display: flex; align-items: center; }
  .status-text { font-size: 11px; color: var(--dimtext); transition: color .3s; }
  .status-text.working { color: var(--accent); }
  .status-text.ok      { color: var(--success); }
  .status-text.fail    { color: var(--danger); }
  .receipts-label { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 2rem 0 .75rem; }
  .receipts { display: flex; flex-direction: column; gap: 10px; }
  .receipt { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; opacity: 0; transform: translateY(-8px); animation: slideIn .3s ease forwards; position: relative; overflow: hidden; }
  .receipt::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(180deg, var(--accent), var(--accent2)); }
  @keyframes slideIn { to { opacity: 1; transform: translateY(0); } }
  .receipt-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
  .receipt-summary { font-size: 13px; color: var(--text); line-height: 1.5; }
  .receipt-ts { font-size: 10px; color: var(--muted); white-space: nowrap; flex-shrink: 0; }
  .receipt-path { font-size: 11px; color: var(--accent); word-break: break-all; margin-bottom: 6px; }
  .receipt-path-label { color: var(--muted); margin-right: 4px; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
  .tag { font-size: 10px; padding: 2px 8px; background: rgba(123,94,167,.15); border: 1px solid rgba(123,94,167,.3); border-radius: 10px; color: var(--accent2); }
  .empty { text-align: center; padding: 2.5rem 0; color: var(--muted); font-size: 12px; letter-spacing: .06em; }
  .empty-icon { font-size: 28px; margin-bottom: 8px; opacity: .3; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .pulsing { animation: pulse .9s ease infinite; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="brand">SYN<span>APSe</span></div>
    <div class="vault-indicator">
      <div class="dot" id="vault-dot"></div>
      <span id="vault-label">checking vault…</span>
    </div>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="setMode('note',this)">NOTE</button>
    <button class="tab" onclick="setMode('url',this)">URL</button>
  </div>
  <div class="input-panel">
    <textarea id="content"
      placeholder="Stream anything into the vault — a thought, a concept, a research note, a paste from elsewhere…"
      autofocus onkeydown="handleKey(event)"></textarea>
    <div class="input-footer">
      <div class="override-row">
        <span class="override-label">-> folder</span>
        <input id="override-folder" type="text" placeholder="auto-route" spellcheck="false">
      </div>
      <button class="submit-btn" id="submit-btn" onclick="submit()">
        FIRE <span class="kbd">cmd+enter</span>
      </button>
    </div>
  </div>
  <div class="status-bar"><span class="status-text" id="status">ready</span></div>
  <p class="receipts-label" id="history-label" style="display:none">FILED THIS SESSION</p>
  <div class="receipts" id="receipts">
    <div class="empty" id="empty-state">
      <div class="empty-icon">o</div>
      NOTHING FILED YET
    </div>
  </div>
</div>
<script>
let mode = 'note';
function setMode(m,el){mode=m;document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));el.classList.add('active');const ta=document.getElementById('content');ta.placeholder=m==='url'?'Paste any URL — SYNAPSE will scrape, extract, and file it automatically…':'Stream anything into the vault — a thought, a concept, a research note, a paste from elsewhere…';ta.focus();}
function handleKey(e){if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();submit();}}
function setStatus(msg,cls){const el=document.getElementById('status');el.textContent=msg;el.className='status-text '+(cls||'');}
async function submit(){
  const content=document.getElementById('content').value.trim();
  const override=document.getElementById('override-folder').value.trim();
  if(!content){setStatus('nothing to insert','fail');return;}
  const btn=document.getElementById('submit-btn');
  btn.classList.add('loading');btn.textContent='...';
  setStatus('routing signal…','working pulsing');
  try{
    const res=await fetch('/insert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,content,override_folder:override})});
    const data=await res.json();
    if(!res.ok||data.error){setStatus('error: '+(data.error||res.statusText),'fail');}
    else{setStatus('filed -> '+data.path,'ok');addReceipt(data);document.getElementById('content').value='';document.getElementById('override-folder').value='';}
  }catch(err){setStatus('connection error: '+err.message,'fail');}
  finally{btn.classList.remove('loading');btn.innerHTML='FIRE <span class="kbd">cmd+enter</span>';}
}
function addReceipt(d){
  document.getElementById('empty-state')?.remove();
  document.getElementById('history-label').style.display='';
  const tags=(d.tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join('');
  const card=document.createElement('div');card.className='receipt';
  card.innerHTML='<div class="receipt-top"><span class="receipt-summary">'+esc(d.summary||'')+'</span><span class="receipt-ts">'+d.timestamp+'</span></div><div class="receipt-path"><span class="receipt-path-label">path </span>'+esc(d.path)+'</div>'+(tags?'<div class="tags">'+tags+'</div>':'');
  document.getElementById('receipts').prepend(card);
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function checkVault(){try{const d=await(await fetch('/vault-status')).json();document.getElementById('vault-dot').className='dot '+(d.vault_exists?'live':'err');document.getElementById('vault-label').textContent=d.vault_exists?'vault connected':'vault not found';}catch(e){document.getElementById('vault-label').textContent='no connection';}}
checkVault();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────
def open_browser():
    time.sleep(0.9)
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    print(f"\n  o  SYNAPSE  --  powered by Gemini\n")
    print(f"  Vault : {VAULT_PATH}")
    print(f"  URL   : http://localhost:{PORT}")
    print(f"  Model : {MODEL}")
    print(f"\n  Press Ctrl+C to stop\n")
    if not VAULT_PATH.exists():
        print(f"  WARNING: Vault path not found: {VAULT_PATH}\n")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
