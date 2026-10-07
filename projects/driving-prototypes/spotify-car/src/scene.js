import { renderStars, resizeStars } from './elements/stars.js';
import { renderSun, resizeSun } from './elements/sun.js';
import { renderMountains, resizeMountains } from './elements/mountains.js';
import { renderRoad, resizeRoad } from './elements/road.js';
import { renderCar, resizeCar } from './elements/car.js';

let canvas = null;
let ctx = null;
let width = 0;
let height = 0;
let dpr = 1;

export function initScene() {
  canvas = document.getElementById('scene-canvas');
  ctx = canvas.getContext('2d');
  dpr = window.devicePixelRatio || 1;
  handleResize();
  window.addEventListener('resize', handleResize);
}

function handleResize() {
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  resizeStars(width, height);
  resizeSun();
  resizeMountains(width, height);
  resizeRoad();
  resizeCar();
}

/**
 * Render one frame.
 * @param {object} bands - Audio frequency bands from audio.js
 * @param {number} time - Time in seconds since start
 * @param {number} dt - Delta time in seconds
 */
export function renderFrame(bands, time, dt) {
  // Clear entirely to Black before drawing
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  // Render layers back-to-front
  renderStars(ctx, width, height, bands, time, dt);
  renderSun(ctx, width, height, bands, time);
  renderMountains(ctx, width, height, bands, time, dt);
  renderRoad(ctx, width, height, bands, time, dt);
  renderCar(ctx, width, height, bands, time);

  // Post-processing vignette (optional, darker around edges)
  renderVignette(ctx, width, height);
}

function renderVignette(ctx, w, h) {
  ctx.globalCompositeOperation = 'multiply';
  const grad = ctx.createRadialGradient(w / 2, h / 2, w * 0.4, w / 2, h / 2, Math.max(w, h));
  grad.addColorStop(0, 'rgba(255, 255, 255, 1)'); 
  grad.addColorStop(1, 'rgba(100, 100, 100, 1)'); // Darken edges
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = 'source-over'; // reset
}
