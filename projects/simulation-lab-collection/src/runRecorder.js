// Persistent record of the best runs ever flown.
//
// A run is only meaningful alongside the track it was flown on, so each record carries a
// snapshot of the gates. Trajectories are Float32Array; JSON would store each number as
// text and roughly triple the size, so they are base64'd instead.

const STORAGE_KEY = 'drone_best_runs';
const MAX_RUNS = 5;
const VERSION = 1;

function encodeTrajectory(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  // Chunked: String.fromCharCode.apply on a 16KB array overflows the argument stack.
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function decodeTrajectory(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function loadRuns() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(r => r && r.version === VERSION && r.trajectory);
  } catch (err) {
    console.warn('Could not read saved runs:', err);
    return [];
  }
}

export function getRun(index = 0) {
  const runs = loadRuns();
  const run = runs[index];
  if (!run) return null;
  return { ...run, trajectory: decodeTrajectory(run.trajectory) };
}

// Keeps the top MAX_RUNS by fitness. Returns true if this run made the cut.
export function saveRun({ trajectory, fitness, gatesPassed, topSpeed, generation, startGateIdx, gates, tracksideCamera }) {
  if (!trajectory || trajectory.length < 14) return false;

  const record = {
    version: VERSION,
    recordedAt: new Date().toISOString(),
    fitness,
    gatesPassed,
    totalGates: gates.length,
    topSpeed,
    generation,
    startGateIdx: startGateIdx || 0,
    // Deep copy: the live track keeps mutating as the editor is used.
    gates: JSON.parse(JSON.stringify(gates)),
    tracksideCamera: tracksideCamera ? JSON.parse(JSON.stringify(tracksideCamera)) : null,
    trajectory: encodeTrajectory(trajectory)
  };

  const runs = loadRuns();
  runs.push(record);
  runs.sort((a, b) => b.fitness - a.fitness);
  runs.length = Math.min(runs.length, MAX_RUNS);

  const kept = runs.indexOf(record) !== -1;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
  } catch (err) {
    // Quota exceeded: drop to a single run rather than losing the record entirely.
    console.warn('Run storage full, keeping only the best run.', err);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(runs.slice(0, 1)));
    } catch (_) {
      return false;
    }
  }

  return kept;
}

export function clearRuns() {
  localStorage.removeItem(STORAGE_KEY);
}
