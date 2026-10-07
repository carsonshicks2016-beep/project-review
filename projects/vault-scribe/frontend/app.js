// Vault Scribe — Frontend Application Logic
// Handles UI interactions, API calls to the FastAPI backend, and dynamic rendering.

const API_BASE = "http://127.0.0.1:8484/api";

// ── DOM Elements ────────────────────────────────────────────────────────────
const els = {
    tabs: document.querySelectorAll('.nav-item'),
    panes: document.querySelectorAll('.tab-pane'),
    
    // Intake Panel Left
    noteTitle: document.getElementById('note-title'),
    rawContent: document.getElementById('raw-content'),
    templateOverride: document.getElementById('template-override'),
    btnAnalyze: document.getElementById('btn-analyze'),
    btnPreview: document.getElementById('btn-preview'),
    btnCommit: document.getElementById('btn-commit'),
    
    // Intake Panel Right
    placeholder: document.getElementById('analysis-placeholder'),
    results: document.getElementById('analysis-results'),
    
    resCategory: document.getElementById('res-category'),
    resConfidence: document.getElementById('res-confidence'),
    resFolder: document.getElementById('res-folder'),
    resTemplate: document.getElementById('res-template'),
    resReasoning: document.getElementById('res-reasoning'),
    
    // Contradictions
    cardContradictions: document.getElementById('card-contradictions'),
    resContradiction: document.getElementById('res-contradiction'),
    
    cardDuplicates: document.getElementById('card-duplicates'),
    listDuplicates: document.getElementById('list-duplicates'),
    
    cardGhosts: document.getElementById('card-ghosts'),
    listGhosts: document.getElementById('list-ghosts'),
    
    cardLinks: document.getElementById('card-links'),
    listLinks: document.getElementById('list-links'),
    emptyLinks: document.getElementById('empty-links'),
    
    cardRetro: document.getElementById('card-retro'),
    listRetro: document.getElementById('list-retro'),
    
    // URL Import
    urlInput: document.getElementById('url-input'),
    btnFetchUrl: document.getElementById('btn-fetch-url'),
    
    // Innervate
    innervateNoteInput: document.getElementById('innervate-note-input'),
    vaultNotesList: document.getElementById('vault-notes-list'),
    btnInnervateScan: document.getElementById('btn-innervate-scan'),
    innervateResults: document.getElementById('innervate-results'),
    innervateStats: document.getElementById('innervate-stats'),
    listInnervatePatches: document.getElementById('list-innervate-patches'),
    btnInnervateApply: document.getElementById('btn-innervate-apply'),
    
    // Gaps
    btnGapsScan: document.getElementById('btn-gaps-scan'),
    gapsResults: document.getElementById('gaps-results'),
    listGaps: document.getElementById('list-gaps'),
    
    // Experiments
    btnNewExperiment: document.getElementById('btn-new-experiment'),
    listActiveExperiments: document.getElementById('list-active-experiments'),
    logModal: document.getElementById('log-modal'),
    btnCloseLogModal: document.getElementById('btn-close-log-modal'),
    btnCancelLogModal: document.getElementById('btn-cancel-log-modal'),
    btnSubmitLog: document.getElementById('btn-submit-log'),
    logModalTitle: document.getElementById('log-modal-title'),
    logExperimentTitle: document.getElementById('log-experiment-title'),
    logNotes: document.getElementById('log-notes'),
    logMetric: document.getElementById('log-metric'),
    logSubjective: document.getElementById('log-subjective'),
    
    // Chat
    chatHistory: document.getElementById('chat-history'),
    chatInput: document.getElementById('chat-input'),
    btnChatSend: document.getElementById('btn-chat-send'),
    
    // Insights
    btnGenerateInsight: document.getElementById('btn-generate-insight'),
    insightResults: document.getElementById('insight-results'),
    insightContent: document.getElementById('insight-content'),
    
    // Librarian
    btnLibrarianSearch: document.getElementById('btn-librarian-search'),
    inputLibrarianQuery: document.getElementById('librarian-query'),
    librarianLoading: document.getElementById('librarian-loading'),
    librarianResults: document.getElementById('librarian-results'),
    
    // Modal
    modal: document.getElementById('preview-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    btnCancelModal: document.getElementById('btn-cancel-modal'),
    btnModalCommit: document.getElementById('btn-modal-commit'),
    markdownOutput: document.getElementById('markdown-output'),
    
    // Search
    searchInput: document.getElementById('search-input'),
    btnSearch: document.getElementById('btn-search'),
    listSearchResults: document.getElementById('list-search-results'),
    
    // Batch Innervate
    btnInnervateBatch: document.getElementById('btn-innervate-batch'),
    btnInnervateLoadMore: document.getElementById('btn-innervate-load-more'),
    
    // Ghost Extinct
    btnExtinctGhosts: document.getElementById('btn-extinct-ghosts'),
    
    // Save Librarian
    btnSaveLibrarian: document.getElementById('btn-save-librarian'),
    librarianActions: document.getElementById('librarian-actions'),

    // Stats & Status
    statNotes: document.getElementById('stat-notes'),
    statLinks: document.getElementById('stat-links'),
    rebuildBtn: document.getElementById('rebuild-btn'),
    indexStatus: document.getElementById('index-status'),
    statusText: document.querySelector('#index-status .status-text'),
    statusDot: document.querySelector('#index-status .dot'),
    
    toastContainer: document.getElementById('toast-container'),
};

// ── State ───────────────────────────────────────────────────────────────────
let currentState = {
    analyzedContent: "",
    suggestedTitle: "",
    classification: null,
    links: [],
    ghosts: [],
    duplicates: [],
    retroactive: [],
    
    // User selections
    selectedLinks: [],
    selectedGhosts: [],
    selectedRetro: [],
    
    // Innervate State
    innervatePatches: [],
    innervateSkip: 0,
    innervateHasMore: false,
    
    // Chat State
    chatMessages: []
};


// ── Initialization ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initEventListeners();
    fetchIndexStats();
    fetchAllNotes();
    fetchActiveExperiments();
});

function initTabs() {
    els.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            
            // Update active nav
            els.tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Update active pane
            els.panes.forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });
}

function initEventListeners() {
    els.btnAnalyze.addEventListener('click', handleAnalyze);
    els.btnPreview.addEventListener('click', handlePreview);
    els.btnCommit.addEventListener('click', handleCommit);
    els.rebuildBtn.addEventListener('click', handleRebuildIndex);
    
    els.btnCloseModal.addEventListener('click', closeModal);
    els.btnCancelModal.addEventListener('click', closeModal);
    els.btnModalCommit.addEventListener('click', handleCommit);
    
    if (els.btnFetchUrl) {
        els.btnFetchUrl.addEventListener('click', handleFetchUrl);
    }
    
    if (els.btnInnervateScan) {
        els.btnInnervateScan.addEventListener('click', handleInnervateScan);
        els.btnInnervateApply.addEventListener('click', handleInnervateApply);
    }
    
    if (els.btnGapsScan) {
        els.btnGapsScan.addEventListener('click', handleGapsScan);
    }
    
    if (els.btnNewExperiment) {
        els.btnNewExperiment.addEventListener('click', () => {
            els.noteTitle.value = "Experiment: ";
            els.templateOverride.value = "experiment";
            document.querySelector('.nav-item[data-tab="intake"]').click();
            showToast("Fill out the raw details to draft the experiment.", "info");
        });
        
        if (els.btnCloseLogModal) els.btnCloseLogModal.addEventListener('click', closeLogModal);
        if (els.btnCancelLogModal) els.btnCancelLogModal.addEventListener('click', closeLogModal);
        if (els.btnSubmitLog) els.btnSubmitLog.addEventListener('click', submitLogEntry);
    }
    
    if (els.btnChatSend) {
        els.btnChatSend.addEventListener('click', handleChatSend);
        els.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSend();
            }
        });
    }
    
    if (els.btnGenerateInsight) {
        els.btnGenerateInsight.addEventListener('click', handleGenerateInsight);
    }
    
    if (els.btnLibrarianSearch) {
        els.btnLibrarianSearch.addEventListener('click', handleLibrarianSearch);
        els.inputLibrarianQuery.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleLibrarianSearch();
        });
    }
    
    if (els.btnSearch) {
        els.btnSearch.addEventListener('click', handleSearch);
        els.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSearch();
        });
    }
    
    if (els.btnInnervateBatch) {
        els.btnInnervateBatch.addEventListener('click', () => {
            currentState.innervateSkip = 0;
            currentState.innervatePatches = [];
            handleInnervateBatch();
        });
    }
    
    if (els.btnInnervateLoadMore) {
        els.btnInnervateLoadMore.addEventListener('click', handleInnervateBatch);
    }
    
    if (els.btnExtinctGhosts) {
        els.btnExtinctGhosts.addEventListener('click', handleExtinctGhosts);
    }
    
    if (els.btnSaveLibrarian) {
        els.btnSaveLibrarian.addEventListener('click', handleSaveLibrarian);
    }
    
    // Enable analyze button on typing
    els.rawContent.addEventListener('input', () => {
        if (els.rawContent.value.trim().length > 0) {
            els.btnAnalyze.disabled = false;
        } else {
            els.btnAnalyze.disabled = true;
            els.btnPreview.disabled = true;
            els.btnCommit.disabled = true;
        }
    });
}


// ── API Calls & Handlers ────────────────────────────────────────────────────

async function handleFetchUrl() {
    const url = els.urlInput.value.trim();
    if (!url) return;
    
    els.btnFetchUrl.innerHTML = '⚡ Fetching...';
    els.btnFetchUrl.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/scrape`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Scrape failed");
        }
        
        const data = await res.json();
        
        // Populate Intake tab
        els.rawContent.value = data.content;
        els.noteTitle.value = data.title;
        els.btnAnalyze.disabled = false;
        
        // Switch to Intake tab
        document.querySelector('.nav-item[data-tab="intake"]').click();
        showToast("URL Extracted. Ready to analyze.", "success");
        
    } catch (e) {
        console.error(e);
        showToast("Scrape error: " + e.message, "error");
    } finally {
        els.btnFetchUrl.innerHTML = '⚡ Fetch & Analyze';
        els.btnFetchUrl.disabled = false;
    }
}

async function fetchAllNotes() {
    try {
        const res = await fetch(`${API_BASE}/notes`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (els.vaultNotesList) {
            els.vaultNotesList.innerHTML = data.notes.map(note => `<option value="${note}">`).join('');
        }
    } catch (e) {
        console.error("Failed to fetch notes list", e);
    }
}

async function handleInnervateScan() {
    const title = els.innervateNoteInput.value.trim();
    if (!title) return;
    
    els.btnInnervateScan.innerHTML = '🔍 Scanning...';
    els.btnInnervateScan.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/innervate/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        
        if (!res.ok) throw new Error("Scan failed");
        
        const data = await res.json();
        currentState.innervatePatches = data.patches.map(p => ({ ...p, included: true }));
        
        // Update UI
        els.innervateStats.textContent = `Found ${data.total_patches} mentions across ${data.scanned_notes} notes.`;
        
        els.listInnervatePatches.innerHTML = currentState.innervatePatches.map((p, idx) => `
            <li>
                <input type="checkbox" id="global-retro-${idx}" checked data-idx="${idx}">
                <div>
                    <label for="global-retro-${idx}" class="value">Inject into <strong>${p.target_note_title}</strong></label>
                    <span class="link-context mono">${p.proposed_replacement}</span>
                </div>
            </li>
        `).join('');
        
        // Listeners
        document.querySelectorAll('#list-innervate-patches input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                currentState.innervatePatches[idx].included = e.target.checked;
            });
        });
        
        els.innervateResults.classList.remove('hidden');
        els.btnInnervateApply.disabled = data.total_patches === 0;
        
    } catch (e) {
        console.error(e);
        showToast("Scan failed", "error");
    } finally {
        els.btnInnervateScan.innerHTML = '🔍 Scan Vault';
        els.btnInnervateScan.disabled = false;
    }
}

async function handleInnervateApply() {
    const patches = currentState.innervatePatches.filter(p => p.included);
    if (patches.length === 0) return;
    
    els.btnInnervateApply.innerHTML = '⚡ Injecting...';
    els.btnInnervateApply.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/innervate/apply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patches })
        });
        
        if (!res.ok) throw new Error("Apply failed");
        
        const data = await res.json();
        showToast(`Successfully patched ${data.patched_count} notes!`, "success");
        
        // Reset UI
        els.innervateResults.classList.add('hidden');
        els.innervateNoteInput.value = '';
        
        setTimeout(fetchIndexStats, 2000); // Index is rebuilding
        
    } catch (e) {
        console.error(e);
        showToast("Apply failed", "error");
        els.btnInnervateApply.disabled = false;
        els.btnInnervateApply.innerHTML = '⚡ Inject Links';
    }
}

async function handleGapsScan() {
    els.btnGapsScan.innerHTML = '🔍 Scanning Graph...';
    els.btnGapsScan.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/gaps`);
        if (!res.ok) throw new Error("Gaps scan failed");
        
        const data = await res.json();
        
        if (data.gaps.length === 0) {
            els.listGaps.innerHTML = '<li><span class="value">No major knowledge gaps found! Your graph is solid.</span></li>';
        } else {
            els.listGaps.innerHTML = data.gaps.map((gap, idx) => `
                <li style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="value mono" style="font-size: 1.1em; color: var(--error);">[[${gap.title}]]</span>
                        <div class="link-context" style="margin-top: 0.5rem;">
                            Missing note referenced in <strong>${gap.mention_count}</strong> existing notes:
                            <br>
                            <span style="opacity: 0.7; font-size: 0.9em;">${gap.mentioned_in.slice(0, 5).join(', ')}${gap.mentioned_in.length > 5 ? '...' : ''}</span>
                        </div>
                    </div>
                    <button class="btn btn-secondary btn-create-gap" data-title="${gap.title}" style="padding: 0.3rem 0.6rem; font-size: 0.8rem;">📝 Create Note</button>
                </li>
            `).join('');
            
            // Add listeners to "Create Note" buttons
            document.querySelectorAll('.btn-create-gap').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const title = e.target.dataset.title;
                    els.noteTitle.value = title;
                    els.rawContent.focus();
                    document.querySelector('.nav-item[data-tab="intake"]').click();
                    showToast(`Drafting new note: ${title}`, "info");
                });
            });
        }
        
        els.gapsResults.classList.remove('hidden');
        
    } catch (e) {
        console.error(e);
        showToast("Gaps scan failed", "error");
    } finally {
        els.btnGapsScan.innerHTML = '🔍 Scan Graph for Gaps';
        els.btnGapsScan.disabled = false;
    }
}

async function fetchActiveExperiments() {
    try {
        const res = await fetch(`${API_BASE}/experiments`);
        if (!res.ok) return;
        const data = await res.json();
        
        if (els.listActiveExperiments) {
            if (data.experiments.length === 0) {
                els.listActiveExperiments.innerHTML = '<li><span class="value">No active experiments found.</span></li>';
            } else {
                els.listActiveExperiments.innerHTML = data.experiments.map(exp => `
                    <li style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="value mono">[[${exp.title}]]</span>
                            <div class="sub-text">Intervention: ${exp.intervention}</div>
                        </div>
                        <button class="btn btn-secondary btn-log-data" data-title="${exp.title}">📝 Log Data</button>
                    </li>
                `).join('');
                
                document.querySelectorAll('.btn-log-data').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const title = e.target.dataset.title;
                        openLogModal(title);
                    });
                });
            }
        }
    } catch (e) {
        console.error("Failed to fetch experiments", e);
    }
}

function openLogModal(title) {
    els.logExperimentTitle.value = title;
    els.logModalTitle.textContent = `Log Data: ${title.replace('Experiment: ', '')}`;
    els.logNotes.value = '';
    els.logMetric.value = '';
    els.logSubjective.value = '';
    els.logModal.classList.remove('hidden');
}

function closeLogModal() {
    els.logModal.classList.add('hidden');
}

async function submitLogEntry() {
    const title = els.logExperimentTitle.value;
    const notes = els.logNotes.value.trim();
    if (!notes) {
        showToast("Please enter some notes.", "error");
        return;
    }
    
    els.btnSubmitLog.disabled = true;
    els.btnSubmitLog.textContent = 'Saving...';
    
    try {
        const res = await fetch(`${API_BASE}/experiments/log`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title,
                notes: notes,
                metric: els.logMetric.value.trim() || '—',
                subjective: els.logSubjective.value.trim() || '—'
            })
        });
        
        if (!res.ok) throw new Error("Failed to log data");
        
        showToast("Data logged successfully!", "success");
        closeLogModal();
    } catch (e) {
        console.error(e);
        showToast("Failed to log data", "error");
    } finally {
        els.btnSubmitLog.disabled = false;
        els.btnSubmitLog.textContent = 'Save Log Entry';
    }
}

function appendChatMessage(role, text) {
    const isBot = role === 'assistant' || role === 'system';
    const align = isBot ? 'flex-start' : 'flex-end';
    const bg = isBot ? 'rgba(124, 58, 237, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    const border = isBot ? 'var(--primary)' : 'var(--success)';
    
    const div = document.createElement('div');
    div.style.cssText = `align-self: ${align}; background: ${bg}; border: 1px solid ${border}; padding: 1rem; border-radius: 8px; max-width: 80%; white-space: pre-wrap;`;
    div.textContent = text;
    
    els.chatHistory.appendChild(div);
    els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

async function handleChatSend() {
    const text = els.chatInput.value.trim();
    if (!text) return;
    
    els.chatInput.value = '';
    
    // Add user message to UI and State
    appendChatMessage('user', text);
    currentState.chatMessages.push({ role: 'user', content: text });
    
    els.btnChatSend.disabled = true;
    els.btnChatSend.textContent = '...';
    
    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: currentState.chatMessages })
        });
        
        if (!res.ok) throw new Error("Chat failed");
        
        const data = await res.json();
        const replyText = data.reply;
        
        currentState.chatMessages.push({ role: 'assistant', content: replyText });
        
        // Check if LLM emitted the commit JSON
        if (replyText.includes('```json') && replyText.includes('"action": "COMMIT"')) {
            try {
                const jsonStr = replyText.split('```json')[1].split('```')[0].trim();
                const cmd = JSON.parse(jsonStr);
                
                if (cmd.action === "COMMIT") {
                    appendChatMessage('assistant', `Note successfully drafted! I'm transferring you to the Intake tab to review and save.`);
                    
                    els.noteTitle.value = cmd.title;
                    els.rawContent.value = cmd.content;
                    if (cmd.template && els.templateOverride.querySelector(`option[value="${cmd.template}"]`)) {
                        els.templateOverride.value = cmd.template;
                    }
                    
                    setTimeout(() => {
                        document.querySelector('.nav-item[data-tab="intake"]').click();
                    }, 1500);
                    
                    return;
                }
            } catch (err) {
                console.error("Failed to parse LLM JSON", err);
            }
        }
        
        appendChatMessage('assistant', replyText);
        
    } catch (e) {
        console.error(e);
        appendChatMessage('assistant', "❌ Error connecting to LLM. Check backend logs.");
    } finally {
        els.btnChatSend.disabled = false;
        els.btnChatSend.textContent = 'Send ➔';
        els.chatInput.focus();
    }
}

async function handleGenerateInsight() {
    els.btnGenerateInsight.disabled = true;
    els.btnGenerateInsight.textContent = '✨ Synthesizing...';
    els.insightResults.classList.add('hidden');
    els.insightContent.textContent = '';
    
    try {
        const res = await fetch(`${API_BASE}/insights/generate`, { method: 'POST' });
        if (!res.ok) throw new Error("Failed to generate insight");
        
        const data = await res.json();
        
        els.insightContent.textContent = data.insight;
        els.insightResults.classList.remove('hidden');
        
    } catch (e) {
        console.error(e);
        showToast("Error generating insight.", "error");
    } finally {
        els.btnGenerateInsight.disabled = false;
        els.btnGenerateInsight.textContent = '✨ Generate Insight';
    }
}

async function handleLibrarianSearch() {
    const query = els.inputLibrarianQuery.value.trim();
    if (!query) return;

    els.librarianLoading.classList.remove('hidden');
    els.librarianResults.innerHTML = '';
    els.btnLibrarianSearch.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/librarian/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Librarian query failed");
        
        // Render markdown using marked.js
        currentState.librarianReport = data.report;
        els.librarianResults.innerHTML = marked.parse(data.report);
        els.librarianActions.classList.remove('hidden');
    } catch (e) {
        console.error(e);
        showToast("Librarian Error: " + e.message, "error");
        els.librarianResults.innerHTML = `<p style="color:red; text-align:center; font-weight: bold;">Error: ${e.message}</p>`;
        els.librarianActions.classList.add('hidden');
    } finally {
        els.librarianLoading.classList.add('hidden');
        els.btnLibrarianSearch.disabled = false;
    }
}

async function fetchIndexStats() {
    try {
        const res = await fetch(`${API_BASE}/index`);
        if (!res.ok) throw new Error("Failed to fetch stats");
        const data = await res.json();
        
        if (els.statNotes) els.statNotes.textContent = data.total_notes.toLocaleString();
        if (els.statLinks) els.statLinks.textContent = data.total_links.toLocaleString();
        if (els.statGhosts) els.statGhosts.textContent = data.total_ghosts.toLocaleString();
        if (els.statOrphans) els.statOrphans.textContent = data.orphan_notes.toLocaleString();
        if (els.statWords) els.statWords.textContent = (data.total_words / 1000).toFixed(1) + 'k';
        
        if (els.statIndexed && data.last_indexed) {
            const date = new Date(data.last_indexed + 'Z');
            els.statIndexed.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        
        setIndexStatus("ready", "Vault Indexed");
    } catch (e) {
        setIndexStatus("error", "Index Offline");
        console.error(e);
    }
}

async function handleRebuildIndex() {
    setIndexStatus("indexing", "Rebuilding Index...");
    try {
        const res = await fetch(`${API_BASE}/index/rebuild`, { method: 'POST' });
        const data = await res.json();
        els.statNotes.textContent = data.stats.total_notes.toLocaleString();
        els.statLinks.textContent = data.stats.total_links.toLocaleString();
        setIndexStatus("ready", "Index Rebuilt");
        showToast("Vault index rebuilt successfully", "success");
    } catch (e) {
        setIndexStatus("error", "Rebuild Failed");
        showToast("Failed to rebuild index", "error");
    }
}

async function handleAnalyze() {
    const content = els.rawContent.value.trim();
    if (!content) return;
    
    const templateOverride = els.templateOverride.value;
    
    // UI Loading state
    els.btnAnalyze.innerHTML = '<span class="spinner">↻</span> Analyzing...';
    els.btnAnalyze.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, template_override: templateOverride || null })
        });
        
        if (!res.ok) throw new Error("Analysis failed");
        
        const data = await res.json();
        updateStateAndUI(data, content);
        
        showToast("Analysis complete", "success");
        
        els.btnPreview.disabled = false;
        els.btnCommit.disabled = false;
        
        // Trigger Contradiction Radar in the background
        els.cardContradictions.classList.remove('hidden');
        els.resContradiction.innerHTML = '<span class="pulse-ring" style="display:inline-block;width:12px;height:12px;margin-right:8px;position:relative;top:2px;"></span> <i>Scanning graph for contradictions...</i>';
        els.resContradiction.style.color = 'var(--text-secondary)';
        
        checkContradictions(data.suggested_title, content);

    } catch (e) {
        console.error(e);
        showToast("Analysis failed: " + e.message, "error");
    } finally {
        els.btnAnalyze.innerHTML = '🔍 Analyze';
        els.btnAnalyze.disabled = false;
    }
}

function updateStateAndUI(data, rawContent) {
    currentState = {
        analyzedContent: rawContent,
        suggestedTitle: data.suggested_title,
        classification: data.classification,
        links: data.suggested_links,
        ghosts: data.ghosts,
        duplicates: data.duplicates,
        retroactive: data.retroactive_patches,
        
        // Default selections
        selectedLinks: data.suggested_links.map(l => ({ ...l, included: true })),
        selectedRetro: data.retroactive_patches.map(p => ({ ...p, included: true }))
    };
    
    // Title
    els.noteTitle.value = data.suggested_title;
    
    // Classification Card
    els.resCategory.textContent = data.classification.category;
    els.resFolder.textContent = data.classification.folder;
    els.resTemplate.textContent = data.classification.template;
    els.resReasoning.textContent = data.classification.reasoning;
    
    // Confidence badge
    els.resConfidence.textContent = data.classification.confidence;
    els.resConfidence.className = `badge ${data.classification.confidence}`;
    
    // Duplicates Card
    if (data.has_duplicates) {
        els.cardDuplicates.classList.remove('hidden');
        els.listDuplicates.innerHTML = data.duplicates.map(d => `
            <li>
                <div class="row" style="margin-bottom:0">
                    <span class="value mono">${d.title}</span>
                    <span class="badge ${d.title_similarity > 90 ? 'error' : 'warning'}">
                        ${d.title_similarity.toFixed(0)}% match
                    </span>
                </div>
            </li>
        `).join('');
    } else {
        els.cardDuplicates.classList.add('hidden');
    }
    
    // Ghosts Card
    if (data.ghosts && data.ghosts.length > 0) {
        els.cardGhosts.classList.remove('hidden');
        els.listGhosts.innerHTML = data.ghosts.map(g => `
            <li>
                <div>
                    <span class="value mono">[[${g.target}]]</span>
                    <span class="link-context">Line ${g.line_number}</span>
                </div>
                <div class="ghost-actions">
                    <select class="ghost-action-select" data-target="${g.target}">
                        <option value="keep">Keep Link</option>
                        <option value="remove">Remove Link</option>
                    </select>
                </div>
            </li>
        `).join('');
    } else {
        els.cardGhosts.classList.add('hidden');
    }
    
    // Links Card
    if (data.suggested_links.length > 0) {
        els.listLinks.classList.remove('hidden');
        els.emptyLinks.classList.add('hidden');
        els.listLinks.innerHTML = data.suggested_links.map((l, idx) => `
            <li>
                <input type="checkbox" id="link-${idx}" checked data-idx="${idx}">
                <div>
                    <label for="link-${idx}" class="value">
                        [[${l.target_title}]]
                        <span class="${l.match_type === 'exact' ? 'match-exact' : 'match-fuzzy'}">
                            (${l.match_type})
                        </span>
                    </label>
                    <span class="link-context">"${l.context}"</span>
                </div>
            </li>
        `).join('');
        
        // Add event listeners to link checkboxes
        document.querySelectorAll('#list-links input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                currentState.selectedLinks[idx].included = e.target.checked;
            });
        });
        
    } else {
        els.listLinks.classList.add('hidden');
        els.emptyLinks.classList.remove('hidden');
    }
    
    // Retroactive Card
    if (data.retroactive_patches && data.retroactive_patches.length > 0) {
        els.cardRetro.classList.remove('hidden');
        els.listRetro.innerHTML = data.retroactive_patches.map((p, idx) => `
            <li>
                <input type="checkbox" id="retro-${idx}" checked data-idx="${idx}">
                <div>
                    <label for="retro-${idx}" class="value">Inject into <strong>${p.target_note_title}</strong></label>
                    <span class="link-context mono">${p.proposed_replacement}</span>
                </div>
            </li>
        `).join('');
        
        // Listeners
        document.querySelectorAll('#list-retro input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                currentState.selectedRetro[idx].included = e.target.checked;
            });
        });
    } else {
        els.cardRetro.classList.add('hidden');
    }
    
    // Show results
    els.placeholder.classList.add('hidden');
    els.results.classList.remove('hidden');
}


async function handlePreview() {
    const title = els.noteTitle.value.trim() || currentState.suggestedTitle;
    
    try {
        const res = await fetch(`${API_BASE}/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: currentState.analyzedContent,
                title: title,
                template: currentState.classification.template,
                folder: currentState.classification.folder,
                selected_links: currentState.selectedLinks.filter(l => l.included),
                retroactive_patches: currentState.selectedRetro.filter(p => p.included)
            })
        });
        
        if (!res.ok) throw new Error("Preview generation failed");
        
        const data = await res.json();
        
        // Simple escape for display
        const escaped = data.rendered_markdown
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
            
        els.markdownOutput.innerHTML = escaped;
        openModal();
        
    } catch (e) {
        console.error(e);
        showToast("Preview failed", "error");
    }
}

async function checkContradictions(title, content) {
    try {
        const res = await fetch(`${API_BASE}/analyze/contradictions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, content })
        });
        
        if (!res.ok) throw new Error("Contradiction check failed");
        
        const data = await res.json();
        
        if (data.has_contradiction) {
            els.resContradiction.textContent = data.message;
            els.resContradiction.style.color = 'var(--error)';
        } else {
            els.cardContradictions.classList.add('hidden');
        }
    } catch (e) {
        console.error("Contradiction Radar error:", e);
        els.cardContradictions.classList.add('hidden');
    }
}


async function handleCommit() {
    const title = els.noteTitle.value.trim() || currentState.suggestedTitle;
    
    els.btnCommit.innerHTML = 'Commiting...';
    els.btnCommit.disabled = true;
    els.btnModalCommit.innerHTML = 'Commiting...';
    els.btnModalCommit.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/commit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: currentState.analyzedContent,
                title: title,
                template: currentState.classification.template,
                folder: currentState.classification.folder,
                selected_links: currentState.selectedLinks.filter(l => l.included),
                retroactive_patches: currentState.selectedRetro.filter(p => p.included)
            })
        });
        
        if (!res.ok) throw new Error("Commit failed");
        
        const data = await res.json();
        
        showToast(`Note saved to ${data.filepath}`, "success");
        if (data.retroactive_patched_count > 0) {
            showToast(`Applied ${data.retroactive_patched_count} retroactive links!`, "info");
        }
        
        closeModal();
        resetUI();
        
        // Update stats quietly
        setTimeout(fetchIndexStats, 2000);
        
    } catch (e) {
        console.error(e);
        showToast("Commit failed: " + e.message, "error");
        els.btnCommit.disabled = false;
        els.btnModalCommit.disabled = false;
        els.btnCommit.innerHTML = '✅ Commit to Vault';
        els.btnModalCommit.innerHTML = 'Commit to Vault';
    }
}


// ── Utilities ───────────────────────────────────────────────────────────────

function setIndexStatus(state, text) {
    els.statusDot.className = 'dot ' + state;
    els.statusText.textContent = text;
}

function openModal() {
    els.modal.classList.add('active');
}

function closeModal() {
    els.modal.classList.remove('active');
}

function resetUI() {
    els.rawContent.value = '';
    els.noteTitle.value = '';
    els.btnAnalyze.disabled = true;
    els.btnPreview.disabled = true;
    els.btnCommit.disabled = true;
    els.btnCommit.innerHTML = '✅ Commit to Vault';
    els.btnModalCommit.innerHTML = 'Commit to Vault';
    
    els.placeholder.classList.remove('hidden');
    els.results.classList.add('hidden');
    
    currentState = {
        analyzedContent: "",
        suggestedTitle: "",
        classification: null,
        links: [],
        ghosts: [],
        duplicates: [],
        retroactive: [],
        selectedLinks: [],
        selectedGhosts: [],
        selectedRetro: []
    };
}

function showToast(message, type = "info") {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    els.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ── ASCII Aquarium Background ───────────────────────────────────────────────
function initAquarium() {
    const bg = document.getElementById('aquarium-bg');
    if (!bg) return;

    const fishTypes = [
        "><(((('>", 
        "><>", 
        "<°)))><", 
        "<=><",
        "¸.·´¯`·.´¯`·.¸¸.·´¯`·.¸><(((º>"
    ];

    function spawnFish() {
        const fish = document.createElement('div');
        fish.className = 'ascii-entity';
        
        // Random fish type
        const type = fishTypes[Math.floor(Math.random() * fishTypes.length)];
        fish.textContent = type;
        
        // Is it facing left or right?
        const isLeft = type.startsWith('<');
        
        // Start position
        const yPos = 10 + Math.random() * 80; // 10% to 90% height
        fish.style.top = `${yPos}vh`;
        
        if (isLeft) {
            fish.style.left = '100vw';
            fish.style.transform = 'scaleX(1)';
        } else {
            fish.style.left = '-10vw';
            fish.style.transform = 'scaleX(1)';
        }
        
        bg.appendChild(fish);
        
        // Animate
        const duration = 15000 + Math.random() * 20000; // 15-35s to cross screen
        const startTime = Date.now();
        
        function animate() {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / duration;
            
            if (progress >= 1) {
                fish.remove();
                return;
            }
            
            // Wobble up and down slightly
            const wobble = Math.sin(progress * Math.PI * 4) * 2;
            
            if (isLeft) {
                const x = 100 - (progress * 120); // 100vw to -20vw
                fish.style.transform = `translate(${x}vw, ${wobble}vh)`;
            } else {
                const x = -10 + (progress * 120); // -10vw to 110vw
                fish.style.transform = `translate(${x}vw, ${wobble}vh)`;
            }
            
            // Random bubbles
            if (Math.random() < 0.005) {
                spawnBubble(fish.getBoundingClientRect());
            }
            
            requestAnimationFrame(animate);
        }
        
        requestAnimationFrame(animate);
    }
    
    function spawnBubble(rect) {
        const bubble = document.createElement('div');
        bubble.className = 'ascii-entity ascii-bubble';
        bubble.textContent = Math.random() > 0.5 ? 'o' : 'O';
        
        bubble.style.left = `${rect.left + (rect.width / 2)}px`;
        bubble.style.top = `${rect.top}px`;
        
        bg.appendChild(bubble);
        
        const startTime = Date.now();
        const duration = 5000 + Math.random() * 5000;
        const startY = rect.top;
        
        function animateBubble() {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / duration;
            
            if (progress >= 1) {
                bubble.remove();
                return;
            }
            
            const wobble = Math.sin(progress * Math.PI * 10) * 10;
            const y = startY - (progress * 200);
            
            bubble.style.transform = `translate(${wobble}px, ${y - startY}px)`;
            bubble.style.opacity = 1 - progress;
            
            requestAnimationFrame(animateBubble);
        }
        
        requestAnimationFrame(animateBubble);
    }

    // Initial spawn
    for(let i=0; i<5; i++) {
        setTimeout(spawnFish, Math.random() * 5000);
    }
    
    // Continuous spawn
    setInterval(spawnFish, 4000);
}

// Call on load
document.addEventListener('DOMContentLoaded', initAquarium);

// ── New Handlers ────────────────────────────────────────────────────────────

async function handleSearch() {
    const query = els.searchInput.value.trim();
    if (!query) return;
    
    els.btnSearch.innerHTML = 'Searching...';
    els.listSearchResults.innerHTML = '';
    
    try {
        const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        
        if (data.results.length === 0) {
            els.listSearchResults.innerHTML = '<li style="text-align: center; color: var(--text-muted);">No results found.</li>';
        } else {
            els.listSearchResults.innerHTML = data.results.map(r => `
                <li class="duplicate-item" style="margin-bottom: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                        <strong style="color: var(--primary); font-size: 1.1rem;">${r.title}</strong>
                        <span style="font-size: 0.8rem; color: var(--text-muted); background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">${r.folder}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Match: ${r.type}</div>
                    <div style="font-size: 0.95rem; line-height: 1.4;">${r.preview}...</div>
                </li>
            `).join('');
        }
    } catch (e) {
        showToast("Search failed", "error");
    } finally {
        els.btnSearch.innerHTML = 'Search';
    }
}

async function handleInnervateBatch() {
    const isLoadMore = currentState.innervateSkip > 0;
    
    if (!isLoadMore) {
        els.btnInnervateBatch.innerHTML = '⚡ Scanning Entire Vault... (This may take a moment)';
        els.btnInnervateBatch.disabled = true;
    } else {
        els.btnInnervateLoadMore.innerHTML = 'Loading...';
        els.btnInnervateLoadMore.disabled = true;
    }
    
    try {
        const res = await fetch(`${API_BASE}/innervate/batch?skip=${currentState.innervateSkip}&limit=50`, { method: 'POST' });
        const data = await res.json();
        
        if (data.patches && data.patches.length > 0) {
            const newPatches = data.patches.map(p => ({ ...p, included: true }));
            currentState.innervatePatches = [...currentState.innervatePatches, ...newPatches];
            currentState.innervateSkip = data.skip;
            currentState.innervateHasMore = data.has_more;
            
            renderInnervatePatches();
            
            if (!isLoadMore) {
                showToast(`Found potential retroactive links!`, "success");
            }
        } else if (!isLoadMore) {
            els.innervateResults.classList.remove('hidden');
            els.innervateResults.innerHTML = '<p style="text-align: center; color: var(--success); padding: 2rem;">Vault is fully innervated! No missing links found.</p>';
        } else {
            currentState.innervateHasMore = false;
            renderInnervatePatches();
        }
    } catch (e) {
        showToast("Batch scan failed", "error");
    } finally {
        if (!isLoadMore) {
            els.btnInnervateBatch.innerHTML = '⚡ Scan Entire Vault';
            els.btnInnervateBatch.disabled = false;
        } else {
            els.btnInnervateLoadMore.innerHTML = 'Load Next 50';
            els.btnInnervateLoadMore.disabled = false;
        }
    }
}

function renderInnervatePatches() {
    els.innervateResults.classList.remove('hidden');
    
    const listEl = document.getElementById('list-innervate-patches');
    if (!listEl) return;
    
    const statsEl = document.getElementById('innervate-stats');
    if (statsEl) {
        statsEl.textContent = `Found ${currentState.innervatePatches.length} mentions across multiple notes.`;
    }
    
    listEl.innerHTML = currentState.innervatePatches.map((p, idx) => `
        <li>
            <input type="checkbox" id="patch-${idx}" ${p.included ? 'checked' : ''} data-idx="${idx}">
            <div>
                <label for="patch-${idx}" class="value">Inject into <strong>${p.target_note_title}</strong></label>
                <span class="link-context mono">${p.proposed_replacement}</span>
            </div>
        </li>
    `).join('');
    
    // Add event listeners to checkboxes
    document.querySelectorAll('#list-innervate-patches input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const idx = parseInt(e.target.dataset.idx);
            currentState.innervatePatches[idx].included = e.target.checked;
        });
    });
    
    // Update Load More button visibility
    if (els.btnInnervateLoadMore) {
        if (currentState.innervateHasMore) {
            els.btnInnervateLoadMore.classList.remove('hidden');
        } else {
            els.btnInnervateLoadMore.classList.add('hidden');
        }
    }
}

async function handleExtinctGhosts() {
    if (!confirm("Are you sure you want to remove ALL ghost links from the entire vault? This cannot be undone.")) return;
    
    els.btnExtinctGhosts.innerHTML = '💀 Extincting...';
    els.btnExtinctGhosts.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/ghosts/extinct`, { method: 'POST' });
        const data = await res.json();
        
        showToast(`Extincted ghosts across ${data.extincted_count} notes.`, "success");
        setTimeout(() => location.reload(), 1500);
    } catch (e) {
        showToast("Extinction failed", "error");
        els.btnExtinctGhosts.innerHTML = '💀 Extinct All Ghosts';
        els.btnExtinctGhosts.disabled = false;
    }
}

async function handleSaveLibrarian() {
    const query = els.inputLibrarianQuery.value.trim();
    if (!query) return;
    
    els.btnSaveLibrarian.innerHTML = '📥 Saving...';
    els.btnSaveLibrarian.disabled = true;
    
    try {
        // Find the original raw markdown text from the response (we saved it when we rendered it)
        // Since we don't have it easily accessible, we'll grab the raw text content or we can re-send to API
        // Wait, earlier we used `marked.parse(data.report)`. So `data.report` is what we need.
        // Let's store it on `currentState.librarianReport` when generated.
        
        if (!currentState.librarianReport) {
            showToast("No report to save.", "error");
            return;
        }
        
        const res = await fetch(`${API_BASE}/librarian/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, report: currentState.librarianReport })
        });
        const data = await res.json();
        showToast(`Saved to ${data.filepath}`, "success");
        els.btnSaveLibrarian.innerHTML = '✅ Saved';
    } catch (e) {
        showToast("Save failed", "error");
        els.btnSaveLibrarian.innerHTML = '📥 Commit Report to Vault';
        els.btnSaveLibrarian.disabled = false;
    }
}
