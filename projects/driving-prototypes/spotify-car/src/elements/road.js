let offsetZ = 0;

export function renderRoad(ctx, w, h, bands, time, dt) {
  const horizonY = h * 0.45;
  const roadY = horizonY;
  const roadMaxY = h;
  const planeHeight = roadMaxY - roadY;
  const fov = 300;
  
  // Perspective parameters
  const linesZ = 25; // spacing of horizontal lines in 3D depth
  const numVLines = 30; // vertical lines radiating from center
  const speed = 100 + bands.rms * 600; // grid scroll speed
  
  offsetZ = (offsetZ + speed * dt) % linesZ;

  // Background road color (dark)
  ctx.fillStyle = '#05020c';
  ctx.fillRect(0, roadY, w, planeHeight);

  ctx.save();
  // We want a glowing neon grid
  const pulse = 0.5 + bands.lowMids * 1.5;
  
  ctx.strokeStyle = `rgba(240, 98, 146, 1)`; // Hot pink
  ctx.lineWidth = 2;
  ctx.globalCompositeOperation = 'screen';
  ctx.shadowBlur = 10;
  ctx.shadowColor = '#f06292';
  
  ctx.beginPath();

  // --- Horizontal Lines (Depth) ---
  // We draw lines starting from far away (z large) to close (z small)
  for (let i = 40; i > 0; i--) {
    const z = (i * linesZ) - offsetZ;
    if (z <= 0) continue;
    
    // Project Z to Y on screen
    const depthRatio = fov / (fov + z);
    const y = roadY + (planeHeight * (1 - depthRatio)); 
    
    // Alpha falls off in the distance
    const alpha = Math.min(1, Math.max(0.1, y - roadY) / (planeHeight * 0.5)) * pulse;
    
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }

  // Draw horizontals
  ctx.stroke();

  // --- Vertical Lines (Perspective) ---
  ctx.beginPath();
  ctx.strokeStyle = `rgba(79, 195, 247, 1)`; // Neon cyan for verticals
  ctx.shadowColor = '#4fc3f7';

  const vSpacing = 120; // width spacing
  const startX = w / 2 - (Math.floor(numVLines / 2) * vSpacing);

  for (let i = 0; i < numVLines; i++) {
    const lineStartX = startX + (i * vSpacing);
    
    // Bottom of screen coordinates
    const bottomX = (lineStartX - w/2) + w/2; 
    
    // The vanishing point is roughly camera center (w/2, roadY)
    const VPX = w/2;
    
    ctx.moveTo(bottomX, roadMaxY);
    ctx.lineTo(VPX, roadY);
  }
  
  ctx.stroke();

  // Optional: Inner glowing reflection streaks from the car taillights
  // Bass makes the reflections intense
  const rWidth = w * 0.15;
  const tlGlowGrad = ctx.createLinearGradient(0, h * 0.7, 0, h);
  tlGlowGrad.addColorStop(0, `rgba(255, 20, 20, ${0.3 * bands.bass})`);
  tlGlowGrad.addColorStop(1, 'transparent');
  
  // Left reflection
  ctx.fillStyle = tlGlowGrad;
  ctx.fillRect(w * 0.35 - rWidth/2, h * 0.7, rWidth, h * 0.3);
  // Right reflection
  ctx.fillRect(w * 0.65 - rWidth/2, h * 0.7, rWidth, h * 0.3);

  ctx.restore();
}

export function resizeRoad() {
  offsetZ = 0;
}
