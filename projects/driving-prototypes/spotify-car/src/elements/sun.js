export function renderSun(ctx, w, h, bands, time) {
  const horizonY = h * 0.45;
  const centerX = w * 0.5;
  const baseRadius = Math.min(w * 0.25, h * 0.4);
  const pulse = bands.bass * baseRadius * 0.15 + (bands.beat ? bands.beatIntensity * baseRadius * 0.1 : 0);
  const radius = baseRadius + pulse;
  const centerY = horizonY - radius * 0.4; // Sun sinks a bit into the horizon

  ctx.save();
  
  // Cut out scanlines using clip
  ctx.beginPath();
  // We want to fill the sun area, but NOT the scanlines. 
  // An easy way is to draw the sun, then use destination-out for the scanlines.
  
  // Create Sun Gradient
  const sunGrad = ctx.createLinearGradient(0, centerY - radius, 0, centerY + radius);
  sunGrad.addColorStop(0, '#ffe53b'); // Bright yellow at top
  sunGrad.addColorStop(0.5, '#ff2525'); // Bright red/pink middle
  sunGrad.addColorStop(1, '#ff007f'); // Hot pink bottom
  
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fillStyle = sunGrad;
  
  // Add a massive glow
  ctx.shadowBlur = 80 + bands.bass * 100;
  ctx.shadowColor = '#ff2525';
  ctx.fill();
  
  // Clear shadow before composition
  ctx.shadowBlur = 0;
  
  // Scanlines (cut out using destination-out)
  ctx.globalCompositeOperation = 'destination-out';
  
  const numLines = 15;
  // Scanlines are thicker at the bottom, thinner at the top
  for (let i = 0; i < numLines; i++) {
    const t = i / numLines; // 0 to 1
    // Exponential distribution so lines are denser/closer at the bottom
    const lineY = centerY + radius - (Math.pow(t, 1.5) * radius * 2);
    // Line thickness grows as it goes further down
    const thickness = 2 + (1 - t) * 12;
    
    // Add an offset so the scanlines slowly sink down
    const offsetY = (time * 20) % (radius * 0.1);
    
    ctx.fillRect(centerX - radius - 10, lineY + offsetY, radius * 2 + 20, thickness);
  }
  
  ctx.restore();
}

export function resizeSun() {
  // Nothing to cache
}
