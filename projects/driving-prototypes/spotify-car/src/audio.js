/* ============================================================
   Audio Engine — Web Audio API frequency analysis + beat detection
   ============================================================ */

let audioContext = null;
let analyser = null;
let dataArray = null;
let source = null;
let stream = null;

// Smoothed band values
const bands = {
  bass: 0,        // 20-200 Hz
  lowMids: 0,     // 200-800 Hz
  mids: 0,        // 800-4000 Hz
  highs: 0,       // 4000-16000 Hz
  rms: 0,         // overall energy
  beat: false,     // beat detected this frame
  beatIntensity: 0 // 0-1 beat strength
};

// Beat detection state
let beatHistory = [];
const BEAT_HISTORY_SIZE = 60;
let lastBeatTime = 0;
const BEAT_COOLDOWN = 200; // ms

const SMOOTHING = 0.15; // lower = smoother

/**
 * Initialize audio from system audio capture (getDisplayMedia)
 */
export async function initSystemAudio() {
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      audio: true,
      video: true // required by spec, we just ignore it
    });

    // Stop the video track immediately — we only need audio
    stream.getVideoTracks().forEach(t => t.stop());

    // Check that we actually got audio tracks
    if (stream.getAudioTracks().length === 0) {
      console.error('No audio track in shared stream. Make sure to check "Share audio".');
      return false;
    }

    setupAnalyser(stream);
    return true;
  } catch (err) {
    console.error('System audio capture failed:', err);
    return false;
  }
}

/**
 * Initialize audio from microphone
 */
export async function initMicAudio() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });
    setupAnalyser(stream);
    return true;
  } catch (err) {
    console.error('Mic capture failed:', err);
    return false;
  }
}

function setupAnalyser(mediaStream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.8;

  source = audioContext.createMediaStreamSource(mediaStream);
  source.connect(analyser);

  dataArray = new Uint8Array(analyser.frequencyBinCount);
}

/**
 * Get frequency bin index for a given frequency in Hz
 */
function freqToBin(freq) {
  if (!audioContext) return 0;
  const nyquist = audioContext.sampleRate / 2;
  return Math.round((freq / nyquist) * analyser.frequencyBinCount);
}

/**
 * Compute average energy in a frequency range
 */
function bandAverage(low, high) {
  const lowBin = freqToBin(low);
  const highBin = freqToBin(high);
  let sum = 0;
  let count = 0;
  for (let i = lowBin; i <= highBin && i < dataArray.length; i++) {
    sum += dataArray[i];
    count++;
  }
  return count > 0 ? sum / count / 255 : 0;
}

/**
 * Simple beat detection using energy spike tracking
 */
function detectBeat(currentEnergy) {
  beatHistory.push(currentEnergy);
  if (beatHistory.length > BEAT_HISTORY_SIZE) {
    beatHistory.shift();
  }

  const avg = beatHistory.reduce((a, b) => a + b, 0) / beatHistory.length;
  const now = performance.now();
  const threshold = avg * 1.4 + 0.05;

  if (currentEnergy > threshold && now - lastBeatTime > BEAT_COOLDOWN) {
    lastBeatTime = now;
    return Math.min((currentEnergy - avg) / avg, 1);
  }
  return 0;
}

/**
 * Update audio analysis — call once per frame
 */
export function updateAudio() {
  if (!analyser) return bands;

  analyser.getByteFrequencyData(dataArray);

  // Compute raw band values
  const rawBass = bandAverage(20, 200);
  const rawLowMids = bandAverage(200, 800);
  const rawMids = bandAverage(800, 4000);
  const rawHighs = bandAverage(4000, 16000);

  // Smooth
  bands.bass = bands.bass + (rawBass - bands.bass) * SMOOTHING;
  bands.lowMids = bands.lowMids + (rawLowMids - bands.lowMids) * SMOOTHING;
  bands.mids = bands.mids + (rawMids - bands.mids) * SMOOTHING;
  bands.highs = bands.highs + (rawHighs - bands.highs) * SMOOTHING;

  // RMS over full spectrum
  let sumSq = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 255;
    sumSq += v * v;
  }
  const rawRMS = Math.sqrt(sumSq / dataArray.length);
  bands.rms = bands.rms + (rawRMS - bands.rms) * SMOOTHING;

  // Beat detection (bass-focused)
  const beatStrength = detectBeat(rawBass);
  bands.beat = beatStrength > 0;
  bands.beatIntensity = bands.beatIntensity * 0.85 + (beatStrength > 0 ? beatStrength : 0) * 0.15;

  return bands;
}

/**
 * Get current band values without updating
 */
export function getBands() {
  return bands;
}

/**
 * Check if audio is active
 */
export function isAudioActive() {
  return analyser !== null;
}
