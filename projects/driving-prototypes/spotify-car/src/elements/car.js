let bounceVel = 0;
let bouncePos = 0;
let swayPos = 0;
let swayVel = 0;

export function renderCar(ctx, w, h, bands, time) {
  const baseY = h * 0.72;
  const cx = w * 0.5;
  const carWidth = Math.min(w * 0.35, 400); // Much wider than before
  const carHeight = carWidth * 0.3; // Much lower than before (Flatter)

  // --- Oscillations and Audio-reactive physics ---
  const rideOsc = Math.sin(time * 3.5) * 2 + Math.sin(time * 5) * 1.5;
  const swayOsc = Math.sin(time * 1.5) * 1.5;

  const bounceForce = bands.bass * 30 + (bands.beat ? bands.beatIntensity * 50 : 0);
  bounceVel += (bounceForce - bouncePos * 0.3 - bounceVel * 0.15) * 0.5;
  bouncePos += bounceVel;
  bouncePos = Math.max(-25, Math.min(20, bouncePos));

  const swayTarget = (bands.lowMids - 0.15) * 35;
  swayVel += (swayTarget - swayPos) * 0.08 - swayVel * 0.1;
  swayPos += swayVel;
  swayPos = Math.max(-25, Math.min(25, swayPos));

  const carY = baseY - bouncePos - rideOsc;
  const carX = cx + swayPos + swayOsc;
  const tilt = swayVel * 0.005;

  ctx.save();
  ctx.translate(carX, carY);
  ctx.rotate(tilt);

  const hw = carWidth / 2;
  const hh = carHeight;

  // ---- Underglow ----
  const underglowIntensity = 0.5 + bands.lowMids * 0.8;
  const ugGrad = ctx.createRadialGradient(0, hh * 0.2, carWidth * 0.2, 0, hh * 0.2, carWidth * 0.8);
  ugGrad.addColorStop(0, `rgba(240, 98, 146, ${underglowIntensity * 0.8})`); // Hot pink
  ugGrad.addColorStop(0.5, `rgba(79, 195, 247, ${underglowIntensity * 0.3})`); // Cyan
  ugGrad.addColorStop(1, `rgba(0, 0, 0, 0)`);
  
  ctx.fillStyle = ugGrad;
  // Blend underglow heavily
  ctx.globalCompositeOperation = 'screen';
  ctx.fillRect(-carWidth, -hh*0.2, carWidth * 2, hh * 1.5);
  ctx.globalCompositeOperation = 'source-over'; // reset

  // ---- Car Silhouette (Sharp, aggressive 80s supercar) ----
  ctx.beginPath();
  
  // Start bottom center
  ctx.moveTo(0, 0);
  
  // Bottom chassis / diffuser
  ctx.lineTo(hw * 0.4, 0);
  ctx.lineTo(hw * 0.45, -hh * 0.1); 
  ctx.lineTo(hw * 0.8, -hh * 0.1); // Lower bumper
  
  // Rear wheel arch
  ctx.lineTo(hw * 0.85, 0); 
  ctx.lineTo(hw * 0.95, 0);
  ctx.lineTo(hw, -hh * 0.3); // Sharp vertical cut to shoulder
  
  // Shoulder / flank
  ctx.lineTo(hw * 0.8, -hh * 0.5); // Sloping in
  
  // C-Pillar
  ctx.lineTo(hw * 0.5, -hh * 0.55);
  
  // Roofline
  ctx.lineTo(hw * 0.35, -hh * 0.9);
  ctx.lineTo(-hw * 0.35, -hh * 0.9); // Top roof
  
  // Left side mirrored
  ctx.lineTo(-hw * 0.5, -hh * 0.55);
  ctx.lineTo(-hw * 0.8, -hh * 0.5);
  ctx.lineTo(-hw, -hh * 0.3);
  ctx.lineTo(-hw * 0.95, 0);
  ctx.lineTo(-hw * 0.85, 0);
  ctx.lineTo(-hw * 0.8, -hh * 0.1);
  ctx.lineTo(-hw * 0.45, -hh * 0.1);
  ctx.lineTo(-hw * 0.4, 0);
  ctx.closePath();

  // Very dark base fill
  ctx.fillStyle = '#020108';
  ctx.fill();

  // Edge highlights (synthwave neon grid reflections on the car body)
  ctx.strokeStyle = `rgba(79, 195, 247, ${0.1 + bands.mids * 0.3})`; // Cyan edge
  ctx.lineWidth = 1;
  ctx.stroke();

  // Louvers / Rear Engine Cover (Iconic 80s supercar detail)
  ctx.beginPath();
  const louverWidth = hw * 0.7;
  const louverStartY = -hh * 0.85;
  const louverEndY = -hh * 0.55;
  for (let i = 0; i < 4; i++) {
    const y = louverStartY + ((louverEndY - louverStartY) * (i / 4));
    const stepWidth = louverWidth * (0.6 + (i * 0.1));
    ctx.moveTo(-stepWidth, y);
    ctx.lineTo(stepWidth, y);
  }
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 2;
  ctx.stroke();

  // ---- Giant retro lightbar (Taillights) ----
  const tlPulse = bands.beat ? 1 : 0.6 + bands.bass * 0.5;
  const tlAlpha = 0.8 * tlPulse;
  
  // The panel that holds the lights
  ctx.fillStyle = '#000000';
  ctx.fillRect(-hw * 0.85, -hh * 0.4, hw * 1.7, hh * 0.15);

  // Left Lightbar
  drawLightbar(ctx, -hw * 0.8, -hh * 0.35, hw * 0.65, hh * 0.08, tlAlpha, tlPulse);
  
  // Right Lightbar
  drawLightbar(ctx, hw * 0.15, -hh * 0.35, hw * 0.65, hh * 0.08, tlAlpha, tlPulse);

  // Center Plate / Text 
  ctx.fillStyle = '#111';
  ctx.fillRect(-hw * 0.1, -hh * 0.35, hw * 0.2, hh * 0.08);
  ctx.fillStyle = `rgba(240, 98, 146, ${0.4 + bands.highs * 0.5})`;
  ctx.font = `bold ${hh*0.04}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText("CRUISE", 0, -hh * 0.31);

  ctx.restore();
}

function drawLightbar(ctx, x, y, w, h, alpha, pulse) {
  // Glow effect setup
  ctx.save();
  // Intensive red/pink core
  ctx.fillStyle = `rgba(255, 50, 80, ${alpha})`;
  ctx.shadowBlur = 30 * pulse;
  ctx.shadowColor = '#ff2525';
  
  ctx.fillRect(x, y, w, h);
  
  // Super bright core line
  ctx.fillStyle = `rgba(255, 255, 255, ${0.8 * alpha})`;
  ctx.shadowBlur = 10;
  ctx.fillRect(x, y + h * 0.3, w, h * 0.4);
  
  // segmented grills over the light
  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  ctx.shadowBlur = 0;
  for (let i = 1; i < 6; i++) {
    ctx.fillRect(x + (w / 6) * i, y, 2, h);
  }
  ctx.restore();
}

export function resizeCar() {
  bouncePos = 0;
  bounceVel = 0;
  swayPos = 0;
  swayVel = 0;
}
