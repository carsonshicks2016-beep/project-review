document.addEventListener("DOMContentLoaded", () => {
  // Tabs
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");

  tabs.forEach(t => {
    t.addEventListener("click", () => {
      tabs.forEach(x => x.classList.remove("active"));
      panels.forEach(x => x.classList.remove("active"));
      
      t.classList.add("active");
      document.getElementById(`tab-${t.dataset.tab}`).classList.add("active");
    });
  });

  // Range inputs
  const bindRange = (id, valId) => {
    const r = document.getElementById(id);
    const v = document.getElementById(valId);
    if (!r || !v) return;
    r.addEventListener("input", () => v.innerText = r.value);
  };
  bindRange("input-gens", "val-gens");
  bindRange("input-pop", "val-pop");
  bindRange("input-iters", "val-iters");

  // Segment buttons
  document.querySelectorAll(".seg").forEach(seg => {
    const btns = seg.querySelectorAll("button");
    btns.forEach(btn => {
      btn.addEventListener("click", () => {
        btns.forEach(b => b.classList.remove("on"));
        btn.classList.add("on");
        seg.dataset.val = btn.dataset.v || btn.innerText.toLowerCase();
      });
    });
  });

  // Global state
  let currentEventSource = null;

  function fetchStatus() {
    fetch("/api/status")
      .then(r => r.json())
      .then(data => {
        const container = document.getElementById("ops-active-runs");
        const chip = document.getElementById("task-chip");
        chip.innerText = `${data.runs.length} running`;
        
        container.innerHTML = "";
        if (data.runs.length === 0) {
          container.innerHTML = `<div style="padding: 20px; color: #888;">No active runs.</div>`;
          return;
        }

        data.runs.forEach(r => {
          const div = document.createElement("div");
          div.className = "ops-run";
          div.innerHTML = `
            <i class="ops-run-state developing"></i>
            <div><b>Co-Evolution Task [${r.pid}]</b><span>${r.cmd}</span></div>
            <em class="ops-pill">${r.uptime}s</em>
            <button class="kill-btn" data-pid="${r.pid}" style="background:#ff3b30; color:white; border:none; padding:4px 8px; border-radius:4px; cursor:pointer;">Kill</button>
            <button class="stream-btn" data-pid="${r.pid}" style="background:#090A0F; color:#00F2FE; border:1px solid #00F2FE; padding:4px 8px; border-radius:4px; cursor:pointer; margin-left:5px;">Watch</button>
          `;
          container.appendChild(div);
        });

        // Bind kill buttons
        container.querySelectorAll(".kill-btn").forEach(btn => {
          btn.addEventListener("click", (e) => {
            const pid = e.target.dataset.pid;
            fetch(`/api/stop/${pid}`, { method: "POST" }).then(() => fetchStatus());
          });
        });

        // Bind stream buttons
        container.querySelectorAll(".stream-btn").forEach(btn => {
          btn.addEventListener("click", (e) => {
            const pid = e.target.dataset.pid;
            streamLogs(pid);
          });
        });
      });
  }

  function streamLogs(pid) {
    if (currentEventSource) {
      currentEventSource.close();
    }
    
    const out = document.getElementById("console-output");
    out.innerHTML = `Connected to stream [${pid}]...\n`;
    
    currentEventSource = new EventSource(`/api/stream/${pid}`);
    currentEventSource.onmessage = (e) => {
      out.innerHTML += e.data + "\n";
      out.scrollTop = out.scrollHeight; // auto-scroll
    };
    currentEventSource.onerror = () => {
      out.innerHTML += "\n[Stream Ended]\n";
      currentEventSource.close();
      currentEventSource = null;
    };
  }

  // Launch button
  document.getElementById("btn-launch").addEventListener("click", () => {
    const data = {
      gens: document.getElementById("input-gens").value,
      pop: document.getElementById("input-pop").value,
      iters: document.getElementById("input-iters").value,
      track: document.getElementById("input-track").dataset.val,
      out: document.getElementById("input-out").value
    };

    fetch("/api/launch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(res => {
      // Switch to ops tab automatically
      tabs.forEach(x => x.classList.remove("active"));
      panels.forEach(x => x.classList.remove("active"));
      
      document.querySelector('[data-tab="ops"]').classList.add("active");
      document.getElementById('tab-ops').classList.add("active");
      
      fetchStatus();
      streamLogs(res.pid);
    });
  });

  document.getElementById("ops-refresh").addEventListener("click", fetchStatus);

  // Poll status occasionally
  setInterval(fetchStatus, 3000);
  fetchStatus();
});
