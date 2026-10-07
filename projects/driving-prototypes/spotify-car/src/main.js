/* ============================================================
   Night Cruise — Main Entry Point
   ============================================================ */

import './style.css';
import { initScene, renderFrame } from './scene.js';
import { initSystemAudio, initMicAudio, updateAudio, isAudioActive } from './audio.js';
import { startLogin, handleCallback, isLoggedIn, startPolling } from './spotify.js';

// ——— DOM refs ———
const modal = document.getElementById('start-modal');
const btnSpotify = document.getElementById('btn-spotify-login');
const btnSystemAudio = document.getElementById('btn-system-audio');
const btnMicAudio = document.getElementById('btn-mic-audio');
const modalStatus = document.getElementById('modal-status');
const hud = document.getElementById('hud');
const hudArt = document.getElementById('hud-art');
const hudTrack = document.getElementById('hud-track');
const hudArtist = document.getElementById('hud-artist');

// ——— State ———
let animating = false;
let startTime = 0;
let lastFrameTime = 0;
let spotifyReady = false;

// Demo bands for when no audio is connected (gentle idle animation)
const demoBands = {
  bass: 0, lowMids: 0, mids: 0, highs: 0, rms: 0,
  beat: false, beatIntensity: 0,
};

// ——— Init ———
async function init() {
  initScene();

  // Check for Spotify callback
  if (window.location.search.includes('code=') || window.location.pathname === '/callback') {
    setStatus('Connecting to Spotify...');
    const success = await handleCallback();
    if (success) {
      onSpotifyReady();
    } else {
      setStatus('Spotify connection failed. Try again.');
    }
  }

  // Check if we already have a token (page reload scenario)
  if (isLoggedIn()) {
    onSpotifyReady();
  }

  // Start render loop immediately (shows idle scene behind modal)
  startAnimation();

  // ——— Button handlers ———
  btnSpotify.addEventListener('click', () => {
    setStatus('Redirecting to Spotify...');
    startLogin();
  });

  btnSystemAudio.addEventListener('click', async () => {
    setStatus('Requesting system audio access...');
    btnSystemAudio.disabled = true;
    btnMicAudio.disabled = true;
    const success = await initSystemAudio();
    if (success) {
      closeModal();
    } else {
      setStatus('System audio failed. Try microphone instead.');
      btnSystemAudio.disabled = false;
      btnMicAudio.disabled = false;
    }
  });

  btnMicAudio.addEventListener('click', async () => {
    setStatus('Requesting microphone access...');
    btnSystemAudio.disabled = true;
    btnMicAudio.disabled = true;
    const success = await initMicAudio();
    if (success) {
      closeModal();
    } else {
      setStatus('Microphone access denied. Please allow and try again.');
      btnSystemAudio.disabled = false;
      btnMicAudio.disabled = false;
    }
  });
}

function onSpotifyReady() {
  spotifyReady = true;
  btnSpotify.textContent = '✓ Connected to Spotify';
  btnSpotify.disabled = true;
  btnSpotify.style.background = '#1a7a3a';
  btnSpotify.style.borderColor = '#1a7a3a';

  // Enable audio buttons
  btnSystemAudio.disabled = false;
  btnMicAudio.disabled = false;

  setStatus('Spotify connected! Now choose an audio source.');

  // Start track polling
  startPolling(onTrackChange);
}

function onTrackChange(track) {
  if (track) {
    hudTrack.textContent = track.name;
    hudArtist.textContent = track.artist;
    if (track.artUrl) {
      hudArt.src = track.artUrl;
    }
    hud.classList.remove('hidden');
  } else {
    hud.classList.add('hidden');
  }
}

function closeModal() {
  modal.classList.add('hidden');
}

function setStatus(msg) {
  modalStatus.textContent = msg;
}

// ——— Animation Loop ———
function startAnimation() {
  if (animating) return;
  animating = true;
  startTime = performance.now() / 1000;
  lastFrameTime = startTime;
  requestAnimationFrame(loop);
}

function loop(timestamp) {
  const now = timestamp / 1000;
  const time = now - startTime;
  const dt = Math.min(now - lastFrameTime, 0.05); // Cap dt to avoid jumps
  lastFrameTime = now;

  // Get audio data or use gentle idle animation
  let bands;
  if (isAudioActive()) {
    bands = updateAudio();
  } else {
    // Idle animation — lively enough to show the scene is alive
    const t = time;
    demoBands.bass = 0.18 + Math.sin(t * 0.6) * 0.1 + Math.sin(t * 1.1) * 0.05;
    demoBands.lowMids = 0.15 + Math.sin(t * 0.8 + 1) * 0.08 + Math.sin(t * 1.5) * 0.04;
    demoBands.mids = 0.12 + Math.sin(t * 1.0 + 2) * 0.07 + Math.sin(t * 2.1) * 0.03;
    demoBands.highs = 0.08 + Math.sin(t * 1.4 + 3) * 0.05;
    demoBands.rms = 0.15 + Math.sin(t * 0.5) * 0.08;
    // Occasional fake beats for taillight pulses
    demoBands.beat = Math.sin(t * 1.8) > 0.95;
    demoBands.beatIntensity = demoBands.beat ? 0.6 : demoBands.beatIntensity * 0.92;
    bands = demoBands;
  }

  renderFrame(bands, time, dt);
  requestAnimationFrame(loop);
}

// ——— Start ———
init();
