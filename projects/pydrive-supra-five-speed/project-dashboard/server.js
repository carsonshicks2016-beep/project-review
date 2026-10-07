import express from 'express';
import { spawn, exec } from 'child_process';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Store active background processes
// Key: PID (number), Value: { pid, command, cwd, proc, logs: [], onDataListeners: [] }
const activeProcesses = new Map();

// Helper to expand user homedir ~/ to full path
const expandHomeDir = (p) => {
  if (p.startsWith('~/')) {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
};

// Generic metrics parser for extraction from stdout
const extractMetrics = (text, command) => {
  const metrics = {};
  let found = false;

  // Pattern 0: PPO race/drift training prints: "it    5  ret    -9.6  laps 0.07  diff 0.15  pi -0.000  vf 2.84  ent 1.84  1011 sps"
  const ppoMatch = text.match(/it\s+(\d+)\s+ret\s+([+-]?[\d.]+)\s+(laps|drift)\s+([+-]?[\d.]+)\s+diff\s+([+-]?[\d.]+)\s+pi\s+([+-]?[\d.]+)\s+vf\s+([+-]?[\d.]+)\s+ent\s+([+-]?[\d.]+)\s+(\d+)\s+sps/i);
  if (ppoMatch) {
    metrics['generation'] = parseInt(ppoMatch[1]); // Maps iteration to generation for the chart X-axis
    metrics['return'] = parseFloat(ppoMatch[2]);
    metrics[ppoMatch[3].toLowerCase()] = parseFloat(ppoMatch[4]);
    metrics['difficulty'] = parseFloat(ppoMatch[5]);
    metrics['policy_loss'] = parseFloat(ppoMatch[6]);
    metrics['value_loss'] = parseFloat(ppoMatch[7]);
    metrics['entropy'] = parseFloat(ppoMatch[8]);
    metrics['speed_sps'] = parseInt(ppoMatch[9]);
    return metrics;
  }

  // Pattern 1: GA evolution prints: "  gen  15  best 1.25 laps  mean 0.90"
  const gaMatch = text.match(/gen\s+(\d+)\s+best\s+([\d.]+)\s+laps\s+mean\s+([\d.]+)/i);
  if (gaMatch) {
    metrics['generation'] = parseInt(gaMatch[1]);
    metrics['best_lap'] = parseFloat(gaMatch[2]);
    metrics['mean_lap'] = parseFloat(gaMatch[3]);
    return metrics;
  }

  // Pattern 2: Stable Baselines3 table lines: "|    ep_rew_mean     | -42.5    |"
  // We match: | key | value |
  const sb3Match = text.match(/\|\s*([a-zA-Z0-9_/_-]+)\s*\|\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\|/);
  if (sb3Match) {
    const fullKey = sb3Match[1].trim();
    const shortKey = fullKey.includes('/') ? fullKey.split('/').pop() : fullKey;
    metrics[shortKey] = parseFloat(sb3Match[2]);
    return metrics;
  }

  // Pattern 3: Generic key: value or key=value patterns (ignores dates, timestamps, brackets)
  // e.g. "loss: 0.12  entropy: -1.2"
  const matches = text.matchAll(/\b([a-zA-Z_][a-zA-Z0-9_-]*)\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g);
  for (const m of matches) {
    metrics[m[1].toLowerCase()] = parseFloat(m[2]);
    found = true;
  }

  return found ? metrics : null;
};

// Start a background process
app.post('/api/run/background', (req, res) => {
  let { command, cwd } = req.body;
  if (!command || !cwd) {
    return res.status(400).json({ error: 'Command and cwd are required' });
  }

  cwd = expandHomeDir(cwd);

  console.log(`[Dashboard] Launching background task: "${command}" in ${cwd}`);

  try {
    const proc = spawn(command, {
      cwd,
      shell: true,
      detached: true, // Allow process group management
      env: {
        ...process.env,
        FORCE_COLOR: '1', // Request ANSI colors from supporting CLI tools
        PYTHONUNBUFFERED: '1' // Force python output to be real-time
      }
    });

    const pid = proc.pid;
    const pInfo = {
      pid,
      command,
      cwd,
      proc,
      logs: [],
      metrics: [], // Store history of { t: timestamp, d: parsedMetrics }
      onDataListeners: [],
      startTime: Date.now()
    };

    activeProcesses.set(pid, pInfo);

    let lineBuffer = '';

    // Capture standard output
    proc.stdout.on('data', (data) => {
      const str = data.toString();
      pInfo.logs.push(str);
      // Keep logs list trimmed to avoid memory leaks
      if (pInfo.logs.length > 5000) pInfo.logs.shift();
      
      // Parse lines for metrics
      lineBuffer += str;
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop(); // Keep the last incomplete line in buffer
      
      const parsedBatch = [];
      lines.forEach(line => {
        const metrics = extractMetrics(line, command);
        if (metrics) {
          const entry = { t: Date.now(), d: metrics };
          pInfo.metrics.push(entry);
          if (pInfo.metrics.length > 5000) pInfo.metrics.shift(); // Prevent memory leak
          parsedBatch.push(metrics);
        }
      });
      
      // Notify active listeners
      pInfo.onDataListeners.forEach(listener => listener({ text: str, metrics: parsedBatch }));
    });

    // Capture error output
    proc.stderr.on('data', (data) => {
      const str = data.toString();
      pInfo.logs.push(str);
      if (pInfo.logs.length > 5000) pInfo.logs.shift();
      pInfo.onDataListeners.forEach(listener => listener({ text: str, metrics: [] }));
    });

    proc.on('close', (code) => {
      console.log(`[Dashboard] Process ${pid} exited with code ${code}`);
      const exitMsg = `\n--- Process exited with code ${code} ---\n`;
      pInfo.logs.push(exitMsg);
      pInfo.onDataListeners.forEach(listener => listener({ text: exitMsg, metrics: [] }));
      
      // Keep the logs in history, but clean up the process reference
      pInfo.proc = null;
    });

    proc.on('error', (err) => {
      console.error(`[Dashboard] Process ${pid} error:`, err);
      const errMsg = `\n--- Process Error: ${err.message} ---\n`;
      pInfo.logs.push(errMsg);
      pInfo.onDataListeners.forEach(listener => listener({ text: errMsg, metrics: [] }));
    });

    res.json({
      success: true,
      pid,
      command,
      cwd
    });
  } catch (error) {
    console.error('[Dashboard] Spawn failed:', error);
    res.status(500).json({ error: error.message });
  }
});

// Run a command in native macOS Terminal using AppleScript
app.post('/api/run/native', (req, res) => {
  let { command, cwd } = req.body;
  if (!command || !cwd) {
    return res.status(400).json({ error: 'Command and cwd are required' });
  }

  cwd = expandHomeDir(cwd);
  console.log(`[Dashboard] Launching native Terminal: "${command}" in ${cwd}`);

  // Escape command quotes for AppleScript double-quotes
  const escapedCmd = command
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/'/g, "'\\''");

  const appleScript = `
    tell application "Terminal"
      activate
      do script "cd '${cwd}' && clear && ${escapedCmd}"
    end tell
  `.trim();

  exec(`osascript -e "${appleScript.replace(/"/g, '\\"')}"`, (err) => {
    if (err) {
      console.error('[Dashboard] AppleScript Terminal launch failed:', err);
      return res.status(500).json({ error: err.message });
    }
    res.json({ success: true });
  });
});

// Open VS Code or Cursor at the directory
app.post('/api/open/vscode', (req, res) => {
  let { cwd } = req.body;
  if (!cwd) return res.status(400).json({ error: 'cwd is required' });

  cwd = expandHomeDir(cwd);
  console.log(`[Dashboard] Opening in Editor: ${cwd}`);

  // Tries Cursor first, then standard VS Code, then macOS Bundle identifier fallback
  const cmd = `cursor "${cwd}" || code "${cwd}" || open -b com.microsoft.VSCode "${cwd}"`;
  exec(cmd, (err) => {
    if (err) {
      console.error('[Dashboard] Open editor failed:', err);
      return res.status(500).json({ error: err.message });
    }
    res.json({ success: true });
  });
});

// Open Folder in Finder
app.post('/api/open/folder', (req, res) => {
  let { cwd } = req.body;
  if (!cwd) return res.status(400).json({ error: 'cwd is required' });

  cwd = expandHomeDir(cwd);
  console.log(`[Dashboard] Opening Finder: ${cwd}`);

  exec(`open "${cwd}"`, (err) => {
    if (err) {
      console.error('[Dashboard] Open Finder failed:', err);
      return res.status(500).json({ error: err.message });
    }
    res.json({ success: true });
  });
});

// Open shell in native Terminal cd'd to project path
app.post('/api/open/terminal', (req, res) => {
  let { cwd } = req.body;
  if (!cwd) return res.status(400).json({ error: 'cwd is required' });

  cwd = expandHomeDir(cwd);
  console.log(`[Dashboard] Opening blank Terminal: ${cwd}`);

  const appleScript = `
    tell application "Terminal"
      activate
      do script "cd '${cwd}' && clear"
    end tell
  `.trim();

  exec(`osascript -e "${appleScript.replace(/"/g, '\\"')}"`, (err) => {
    if (err) {
      console.error('[Dashboard] Open Terminal failed:', err);
      return res.status(500).json({ error: err.message });
    }
    res.json({ success: true });
  });
});

// Stream real-time logs via SSE
app.get('/api/stream-logs/:pid', (req, res) => {
  const pid = parseInt(req.params.pid);
  const pInfo = activeProcesses.get(pid);

  if (!pInfo) {
    return res.status(404).send('Process not found');
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable proxy buffering (Nginx, etc.)

  console.log(`[Dashboard] Client subscribed to logs for process PID ${pid}`);

  // Send history logs first along with metrics history
  const payload = {
    text: pInfo.logs.join(''),
    historyMetrics: pInfo.metrics
  };
  res.write(`data: ${JSON.stringify(payload)}\n\n`);

  const listener = ({ text, metrics }) => {
    res.write(`data: ${JSON.stringify({ text, metrics })}\n\n`);
  };

  pInfo.onDataListeners.push(listener);

  req.on('close', () => {
    console.log(`[Dashboard] Client disconnected from log stream for PID ${pid}`);
    pInfo.onDataListeners = pInfo.onDataListeners.filter(l => l !== listener);
  });
});

// Kill process
app.post('/api/stop/:pid', (req, res) => {
  const pid = parseInt(req.params.pid);
  const pInfo = activeProcesses.get(pid);

  if (!pInfo) {
    // Fallback: try to kill the PID directly at the OS level in case it's a process that survived a server restart
    try {
      console.log(`[Dashboard] Fallback stop: Attempting to kill PID ${pid} via OS signal`);
      process.kill(pid, 'SIGINT');
      
      // Schedule a SIGKILL fallback after 4 seconds
      const killTimer = setTimeout(() => {
        try {
          console.log(`[Dashboard] Fallback stop: Process ${pid} resisted SIGINT, force killing (SIGKILL)...`);
          process.kill(pid, 'SIGKILL');
        } catch (err) {
          // Process might have already exited
        }
      }, 4000);
      killTimer.unref();

      return res.json({ success: true, message: 'OS SIGINT sent to PID (fallback)' });
    } catch (e) {
      console.warn(`[Dashboard] Fallback kill failed for PID ${pid}:`, e.message);
      return res.status(404).json({ error: `Process ${pid} not found in dashboard active list, and OS fallback signal failed: ${e.message}` });
    }
  }

  console.log(`[Dashboard] Request to stop PID ${pid} (${pInfo.command})`);

  if (pInfo.proc) {
    try {
      // Send SIGINT first to let programs checkpoint (like training scripts!)
      pInfo.proc.kill('SIGINT');
      
      // If it doesn't die in 4 seconds, force SIGKILL
      const killTimer = setTimeout(() => {
        const checkInfo = activeProcesses.get(pid);
        if (checkInfo && checkInfo.proc) {
          console.log(`[Dashboard] Process ${pid} resisted SIGINT, force killing (SIGKILL)...`);
          try {
            // Kill the negative PID of the detached process group
            process.kill(-pid, 'SIGKILL');
          } catch (e) {
            try { checkInfo.proc.kill('SIGKILL'); } catch (err) {}
          }
        }
      }, 4000);

      killTimer.unref();

      res.json({ success: true, message: 'SIGINT sent' });
    } catch (e) {
      console.error(`[Dashboard] Error killing process ${pid}:`, e);
      res.status(500).json({ error: e.message });
    }
  } else {
    // Process is already dead
    activeProcesses.delete(pid);
    res.json({ success: true, message: 'Process was already stopped' });
  }
});

// Clear dead process logs from memory
app.post('/api/clear-logs/:pid', (req, res) => {
  const pid = parseInt(req.params.pid);
  const pInfo = activeProcesses.get(pid);
  if (pInfo) {
    if (pInfo.proc) {
      pInfo.logs = [];
      res.json({ success: true, message: 'Active logs cleared' });
    } else {
      activeProcesses.delete(pid);
      res.json({ success: true, message: 'Stopped process logs cleared from history' });
    }
  } else {
    res.status(404).json({ error: 'Process logs not found' });
  }
});

// Get status list and system stats
app.get('/api/status', (req, res) => {
  const processList = [];
  activeProcesses.forEach((val, key) => {
    processList.push({
      pid: val.pid,
      command: val.command,
      cwd: val.cwd,
      startTime: val.startTime,
      active: val.proc !== null
    });
  });

  // Calculate memory usage
  const freeMem = os.freemem();
  const totalMem = os.totalmem();
  const usedMem = totalMem - freeMem;
  const memoryUsagePercent = Math.round((usedMem / totalMem) * 100);

  // System load (1 min average)
  const systemLoad = os.loadavg()[0];
  const cpusCount = os.cpus().length;
  const cpuPercent = Math.round(Math.min((systemLoad / cpusCount) * 100, 100));

  res.json({
    processes: processList,
    system: {
      cpu: cpuPercent,
      memory: memoryUsagePercent,
      uptime: Math.round(os.uptime()),
      hostname: os.hostname(),
      platform: os.platform()
    }
  });
});

app.listen(PORT, () => {
  console.log(`\n======================================================`);
  console.log(`  Carson's Project Dashboard Control Center started!`);
  console.log(`  URL: http://localhost:${PORT}`);
  console.log(`======================================================\n`);
});
