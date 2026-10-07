/**
 * CHECKPOINTS / HALL OF FAME — dashboard panel.
 *
 * Lists HoF slots (latest, best, cleanest, fastest, furthest) for a run via
 * GET /api/runs/{id}/checkpoints. Actions: live eval, copy path, evaluate CLI.
 */

import {
  discoverRunIds,
  evaluateCommand,
  fetchRunCheckpoints,
  HOF_SLOTS,
  indexBySlot,
  isHofSlot,
  setLiveCheckpoint,
  SLOT_BLURB,
  type CheckpointEntry,
  type HofSlot,
} from "../api/checkpointsClient";
import { registerPanel } from "../registry";
import type { PanelContext } from "../types";
import { btn, codeBlock } from "../ui";
import "./checkpoints.css";

const RUN_PREF_KEY = "rallyai.dashboard.checkpoints.runId";

function fmtInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

function fmtFloat(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n) || n < 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function fmtMtime(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  try {
    return new Date(sec * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

function shortHash(h: string | null | undefined): string {
  if (!h) return "—";
  return h.length > 12 ? `${h.slice(0, 10)}…` : h;
}

function specialtyMetric(slot: HofSlot, ck: CheckpointEntry): { label: string; value: string } {
  switch (slot) {
    case "best":
      return { label: "Return", value: fmtFloat(ck.mean_return, 2) };
    case "cleanest":
      return {
        label: "Clean",
        value:
          ck.clean_rate != null && Number.isFinite(ck.clean_rate)
            ? `${(ck.clean_rate * 100).toFixed(0)}%`
            : "—",
      };
    case "fastest":
      return {
        label: "Finish",
        value:
          ck.mean_finish_time != null && Number.isFinite(ck.mean_finish_time)
            ? `${fmtFloat(ck.mean_finish_time, 1)} s`
            : "—",
      };
    case "furthest":
      return {
        label: "Progress",
        value:
          ck.max_progress != null && Number.isFinite(ck.max_progress)
            ? `${(ck.max_progress * 100).toFixed(0)}%`
            : "—",
      };
    default:
      return { label: "Stage", value: ck.stage ?? "—" };
  }
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

function loadPreferredRun(): string | null {
  try {
    return localStorage.getItem(RUN_PREF_KEY);
  } catch {
    return null;
  }
}

function savePreferredRun(runId: string): void {
  try {
    localStorage.setItem(RUN_PREF_KEY, runId);
  } catch {
    /* ignore */
  }
}

/**
 * Mount hall-of-fame / checkpoint browser.
 * Section id: `checkpoints`.
 */
export function mountCheckpoints(
  el: HTMLElement,
  ctx: PanelContext,
): () => void {
  el.id = "checkpoints";
  el.classList.add("ck-panel");

  el.innerHTML = `
    <header class="ck-head">
      <div>
        <div class="ck-kicker">Evidence · disk truth</div>
        <h2 class="ck-title">HALL OF FAME</h2>
      </div>
      <p class="ck-sub">Five named slots beside each run. Latest always writes; specialty slots update when they improve.</p>
    </header>
    <div class="ck-toolbar">
      <div class="ck-field">
        <label for="ck-run-select">Run</label>
        <select id="ck-run-select" aria-label="Training run"></select>
      </div>
      <div class="ck-field">
        <label for="ck-run-manual">Or run id</label>
        <input id="ck-run-manual" type="text" spellcheck="false" placeholder="foundation_01" autocomplete="off" />
      </div>
      <button type="button" class="dash-btn" id="ck-refresh">Refresh</button>
      <button type="button" class="dash-btn" id="ck-load-manual">Load</button>
      <button type="button" class="dash-btn armed" id="ck-eval-now" disabled>Evaluate Now</button>
    </div>
    <div class="ck-status" id="ck-status">Connecting to control room…</div>
    <section class="dash-section" id="ck-hof-section">
      <div class="dash-section-hd">
        <span class="dash-title">Slots</span>
        <span class="dash-meta" id="ck-hof-meta">—</span>
      </div>
      <div class="ck-hof" id="ck-hof"></div>
    </section>
    <section class="dash-section" id="ck-files-section">
      <div class="dash-section-hd">
        <span class="dash-title">Run directory · .pt files</span>
        <span class="dash-meta" id="ck-files-meta">—</span>
      </div>
      <div class="ck-files" id="ck-files"></div>
    </section>
    <section class="dash-section ck-eval" id="ck-eval">
      <div class="dash-section-hd">
        <span class="dash-title">Evaluate command</span>
        <span class="dash-meta" id="ck-eval-meta"></span>
      </div>
      <div id="ck-eval-body"></div>
    </section>
    <div class="ck-toast" id="ck-toast" role="status" aria-live="polite"></div>
  `;

  const runSelect = el.querySelector<HTMLSelectElement>("#ck-run-select")!;
  const runManual = el.querySelector<HTMLInputElement>("#ck-run-manual")!;
  const statusEl = el.querySelector<HTMLElement>("#ck-status")!;
  const hofEl = el.querySelector<HTMLElement>("#ck-hof")!;
  const hofMeta = el.querySelector<HTMLElement>("#ck-hof-meta")!;
  const filesEl = el.querySelector<HTMLElement>("#ck-files")!;
  const filesMeta = el.querySelector<HTMLElement>("#ck-files-meta")!;
  const evalSection = el.querySelector<HTMLElement>("#ck-eval")!;
  const evalBody = el.querySelector<HTMLElement>("#ck-eval-body")!;
  const evalMeta = el.querySelector<HTMLElement>("#ck-eval-meta")!;
  const toastEl = el.querySelector<HTMLElement>("#ck-toast")!;
  const evalNowBtn = el.querySelector<HTMLButtonElement>("#ck-eval-now")!;

  const abort = new AbortController();
  let alive = true;
  let currentRun: string | null = null;
  let checkpoints: CheckpointEntry[] = [];
  let toastTimer: number | null = null;
  let openEvalPath: string | null = null;

  const setStatus = (msg: string, kind: "" | "ok" | "err" = ""): void => {
    statusEl.textContent = msg;
    statusEl.classList.toggle("ok", kind === "ok");
    statusEl.classList.toggle("err", kind === "err");
  };

  const toast = (msg: string): void => {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    if (toastTimer != null) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toastEl.classList.remove("show");
      toastTimer = null;
    }, 1600);
  };

  const showEval = (path: string): void => {
    openEvalPath = path;
    evalSection.classList.add("open");
    evalMeta.textContent = path.split(/[/\\]/).pop() ?? path;
    evalBody.replaceChildren();
    const cmd = evaluateCommand(path);
    evalBody.appendChild(codeBlock({ code: cmd, lang: "sh" }));
    const copyBtn = btn({
      label: "Copy command",
      onClick: () => {
        void copyText(cmd).then((ok) => toast(ok ? "Command copied" : "Copy failed"));
      },
    });
    const row = document.createElement("div");
    row.className = "ck-actions";
    row.appendChild(copyBtn);
    evalBody.appendChild(row);
  };

  const useLive = (path: string): void => {
    setLiveCheckpoint(path);
    toast("Checkpoint armed for LIVE");
    ctx.navigate("live");
  };

  const renderSlotCard = (slot: HofSlot, ck: CheckpointEntry | undefined): HTMLElement => {
    const card = document.createElement("article");
    card.className = `ck-slot${ck ? "" : " empty"}`;
    card.dataset.slot = slot;

    const hd = document.createElement("div");
    hd.className = "ck-slot-hd";
    hd.innerHTML = `
      <span class="ck-slot-name">${slot}</span>
      <span class="ck-slot-blurb">${SLOT_BLURB[slot]}</span>
    `;
    card.appendChild(hd);

    if (!ck) {
      const empty = document.createElement("div");
      empty.className = "ck-empty";
      empty.style.padding = "0.85rem 0.25rem";
      empty.style.border = "none";
      empty.textContent = "Not written yet";
      card.appendChild(empty);
      return card;
    }

    const spec = specialtyMetric(slot, ck);
    const metrics = document.createElement("div");
    metrics.className = "ck-metrics";
    const rows: Array<[string, string]> = [
      ["Timesteps", fmtInt(ck.timesteps)],
      ["Tier", ck.tier != null ? String(ck.tier) : "—"],
      [spec.label, spec.value],
      ["Hash", shortHash(ck.policy_sha256)],
    ];
    for (const [label, value] of rows) {
      const m = document.createElement("div");
      m.className = "ck-metric";
      m.innerHTML = `<span>${label}</span><span title="${value}">${value}</span>`;
      metrics.appendChild(m);
    }
    card.appendChild(metrics);

    const pathEl = document.createElement("pre");
    pathEl.className = "ck-path";
    pathEl.title = ck.path;
    pathEl.textContent = ck.path;
    card.appendChild(pathEl);

    const actions = document.createElement("div");
    actions.className = "ck-actions";
    actions.append(
      btn({
        label: "Live eval",
        variant: "armed",
        onClick: () => useLive(ck.path),
      }),
      btn({
        label: "Copy path",
        onClick: () => {
          void copyText(ck.path).then((ok) => toast(ok ? "Path copied" : "Copy failed"));
        },
      }),
      btn({
        label: "Evaluate",
        onClick: () => showEval(ck.path),
      }),
    );
    card.appendChild(actions);
    return card;
  };

  const renderHof = (): void => {
    const bySlot = indexBySlot(checkpoints);
    hofEl.replaceChildren();
    let filled = 0;
    for (const slot of HOF_SLOTS) {
      const ck = bySlot.get(slot);
      if (ck) filled += 1;
      hofEl.appendChild(renderSlotCard(slot, ck));
    }
    hofMeta.textContent = currentRun
      ? `${filled}/${HOF_SLOTS.length} · ${currentRun}`
      : "—";
  };

  const renderFiles = (): void => {
    filesEl.replaceChildren();
    if (!checkpoints.length) {
      const empty = document.createElement("div");
      empty.className = "ck-empty";
      empty.textContent = currentRun
        ? `No .pt files for ${currentRun}. Start a run or check packages/sim/runs/.`
        : "Select a run to browse checkpoints.";
      filesEl.appendChild(empty);
      filesMeta.textContent = "0 files";
      return;
    }

    filesMeta.textContent = `${checkpoints.length} file${checkpoints.length === 1 ? "" : "s"}`;

    const table = document.createElement("table");
    table.innerHTML = `
      <thead>
        <tr>
          <th>Name</th>
          <th>Slot</th>
          <th>Steps</th>
          <th>Tier</th>
          <th>Size</th>
          <th>Modified</th>
          <th></th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody")!;

    for (const ck of checkpoints) {
      const tr = document.createElement("tr");
      if (openEvalPath === ck.path) tr.classList.add("selected");
      const slotCell = isHofSlot(ck.slot)
        ? `<span class="ck-slot-tag">${ck.slot}</span>`
        : `<span class="dash-faint">${ck.slot ?? "—"}</span>`;
      tr.innerHTML = `
        <td title="${ck.path}">${ck.name}</td>
        <td>${slotCell}</td>
        <td>${fmtInt(ck.timesteps)}</td>
        <td>${ck.tier != null ? ck.tier : "—"}</td>
        <td>${fmtBytes(ck.size)}</td>
        <td>${fmtMtime(ck.mtime)}</td>
        <td></td>
      `;
      const cell = tr.lastElementChild as HTMLTableCellElement;
      const rowActions = document.createElement("div");
      rowActions.className = "ck-actions";
      rowActions.append(
        btn({
          label: "Live",
          variant: "armed",
          onClick: () => useLive(ck.path),
        }),
        btn({
          label: "Copy",
          onClick: () => {
            void copyText(ck.path).then((ok) => toast(ok ? "Path copied" : "Copy failed"));
          },
        }),
        btn({
          label: "Eval",
          onClick: () => {
            showEval(ck.path);
            renderFiles();
          },
        }),
      );
      cell.appendChild(rowActions);
      tbody.appendChild(tr);
    }

    filesEl.appendChild(table);
  };

  const fillRunSelect = (ids: string[], selected: string | null): void => {
    runSelect.replaceChildren();
    if (!ids.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No runs found";
      runSelect.appendChild(opt);
      return;
    }
    for (const id of ids) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      runSelect.appendChild(opt);
    }
    if (selected && ids.includes(selected)) {
      runSelect.value = selected;
    } else {
      runSelect.selectedIndex = 0;
    }
  };

  const loadRun = async (runId: string): Promise<void> => {
    const id = runId.trim();
    if (!id) {
      setStatus("Enter a run id", "err");
      return;
    }
    currentRun = id;
    savePreferredRun(id);
    runManual.value = id;
    setStatus(`Loading checkpoints for ${id}…`);
    try {
      const res = await fetchRunCheckpoints(ctx.apiBase, id, abort.signal);
      if (!alive) return;
      checkpoints = res.checkpoints;
      currentRun = res.run_id;
      const n = checkpoints.length;
      setStatus(
        n
          ? `${n} checkpoint${n === 1 ? "" : "s"} · ${res.run_id}`
          : `Run ${res.run_id} has no .pt files yet`,
        n ? "ok" : "",
      );
      renderHof();
      renderFiles();
      if (openEvalPath && !checkpoints.some((c) => c.path === openEvalPath)) {
        evalSection.classList.remove("open");
        openEvalPath = null;
      }
      } catch (e) {
      if (!alive || (e instanceof DOMException && e.name === "AbortError")) return;
      checkpoints = [];
      renderHof();
      renderFiles();
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(
        `Failed: ${msg} — needs control room GET /api/runs/{id}/checkpoints`,
        "err",
      );
    } finally {
      evalNowBtn.disabled = !currentRun;
    }
  };

  const refreshRuns = async (prefer?: string | null): Promise<void> => {
    setStatus("Discovering runs…");
    try {
      const ids = await discoverRunIds(ctx.apiBase, abort.signal);
      if (!alive) return;
      const preferred = prefer ?? loadPreferredRun() ?? ids[0] ?? null;
      fillRunSelect(ids, preferred);
      if (preferred) {
        await loadRun(preferred);
      } else {
        currentRun = null;
        checkpoints = [];
        renderHof();
        renderFiles();
        setStatus(
          "No runs on disk. Start training or point control at packages/sim/runs/.",
          "",
        );
      }
    } catch (e) {
      if (!alive || (e instanceof DOMException && e.name === "AbortError")) return;
      const msg = e instanceof Error ? e.message : String(e);
      fillRunSelect([], null);
      setStatus(`Control unreachable: ${msg}`, "err");
      // Still allow manual load.
      const manual = (prefer ?? loadPreferredRun() ?? "").trim();
      if (manual) {
        runManual.value = manual;
        await loadRun(manual);
      } else {
        checkpoints = [];
        renderHof();
        renderFiles();
      }
    }
  };

  el.querySelector("#ck-refresh")!.addEventListener("click", () => {
    void refreshRuns(runSelect.value || runManual.value || currentRun);
  });
  el.querySelector("#ck-load-manual")!.addEventListener("click", () => {
    void loadRun(runManual.value || runSelect.value);
  });
  runSelect.addEventListener("change", () => {
    if (runSelect.value) void loadRun(runSelect.value);
  });
  runManual.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      void loadRun(runManual.value);
    }
  });

  evalNowBtn.addEventListener("click", async () => {
    if (!currentRun) return;
    toast("Starting evaluation...");
    try {
      const base = ctx.apiBase.trim().replace(/\/+$/, "");
      const path = `/api/runs/${encodeURIComponent(currentRun)}/evaluate`;
      const url = base ? `${base}${path}` : path;
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      toast("Evaluation started");
    } catch (e) {
      toast("Eval failed: " + (e instanceof Error ? e.message : String(e)));
    }
  });

  // Initial HoF empty grid so the composition is visible while loading.
  renderHof();
  renderFiles();
  void refreshRuns(loadPreferredRun());

  return () => {
    alive = false;
    abort.abort();
    if (toastTimer != null) window.clearTimeout(toastTimer);
  };
}

registerPanel("checkpoints", mountCheckpoints);
