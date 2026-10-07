let scrollX = 0;
const peaks = [];
let initialized = false;

function initMountains(w, h) {
  peaks.length = 0;
  // Generate a procedural mountain range
  let x = 0;
  while (x < w * 3) {
    const width = Math.random() * 150 + 100;
    const height = Math.random() * 100 + 40;
    peaks.push({ x, w: width, h: height });
    x += width * 0.7; // overlap
  }
  initialized = true;
}

export function renderMountains(ctx, w, h, bands, time, dt) {
  if (!initialized) initMountains(w, h);
  
  const horizonY = h * 0.45;
  // Scroll opposite to the road or leftwards
  const speed = 10 + bands.rms * 50; 
  scrollX += speed * dt;
  
  // Wrap scroll
  if (scrollX > w) {
    scrollX -= w;
    // For a real continuous loop we'd pop/push peaks, but wrapping the whole array works if w*3 is wide enough
  }

  ctx.save();
  ctx.beginPath();
  
  // Start below the horizon
  ctx.moveTo(0, horizonY + 10);
  
  let started = false;
  
  for (const peak of peaks) {
    const worldX = peak.x - scrollX;
    
    // Cull off-screen
    if (worldX + peak.w < -100 || worldX > w + 100) continue;

    if (!started) {
      ctx.lineTo(worldX, horizonY);
      started = true;
    }
    
    // Draw mountain peak (triangle)
    ctx.lineTo(worldX + peak.w * 0.5, horizonY - peak.h);
    ctx.lineTo(worldX + peak.w, horizonY);
  }
  
  ctx.lineTo(w, horizonY + 10);
  ctx.closePath();
  
  // Fill with solid dark color so it blocks the stars/sun
  ctx.fillStyle = '#0a0512'; 
  ctx.fill();
  
  // Neon wireframe outline
  ctx.lineWidth = 2;
  const pulse = 0.5 + bands.mids * 1.5;
  
  // Glowing cyan or magenta outline
  ctx.strokeStyle = `rgba(179, 136, 255, ${0.4 * pulse})`;
  ctx.shadowBlur = 15;
  ctx.shadowColor = '#b388ff';
  
  ctx.stroke();
  
  // Base line on the horizon to separate from road
  ctx.beginPath();
  ctx.moveTo(0, horizonY);
  ctx.lineTo(w, horizonY);
  ctx.lineWidth = 3;
  ctx.strokeStyle = `rgba(240, 98, 146, ${0.8 + bands.rms})`;
  ctx.shadowColor = '#f06292';
  ctx.stroke();
  
  ctx.restore();
}

export function resizeMountains(w, h) {
  initMountains(w, h);
}
