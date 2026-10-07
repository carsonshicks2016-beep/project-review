const stars = [];
const STAR_COUNT = 300;
let initialized = false;

function initStars(w, h) {
  stars.length = 0;
  for (let i = 0; i < STAR_COUNT; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * (h * 0.45),
      z: Math.random() * 1000 + 100, // Z depth for perspective
      size: Math.random() * 1.5 + 0.5,
      color: Math.random() > 0.8 ? '#4fc3f7' : '#ffffff' // Occasional neon blue star
    });
  }
  initialized = true;
}

export function renderStars(ctx, w, h, bands, time, dt) {
  if (!initialized || stars.length === 0) initStars(w, h);
  
  const horizonY = h * 0.45;
  const speed = 20 + bands.rms * 300; // Warp speed when intense

  // Dark sky gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
  skyGrad.addColorStop(0, '#040b16');
  skyGrad.addColorStop(1, '#1b0b2e'); // Deep synthwave purple near horizon
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, w, horizonY);

  ctx.save();
  // Optional: add a slight global glow for stars
  ctx.shadowBlur = 5;
  ctx.shadowColor = '#ffffff';

  for (let i = 0; i < STAR_COUNT; i++) {
    const star = stars[i];
    
    // Move stars towards the left side (simulating wind/speed) and slightly towards the camera
    star.x -= (speed * dt * (1000 / star.z)) * 0.5;
    
    // Wrap around screen
    if (star.x < 0) {
      star.x = w;
      star.y = Math.random() * horizonY;
    }

    const alpha = Math.min(1, 150 / star.z) * (0.3 + bands.highs * 0.7);
    
    // Draw star
    ctx.beginPath();
    ctx.fillStyle = star.color;
    ctx.globalAlpha = alpha;
    
    // If fast enough, draw as streak
    if (speed > 100) {
      const streakLength = speed * dt * (1000 / star.z) * 0.5;
      ctx.fillRect(star.x, star.y, streakLength + star.size, star.size);
    } else {
      ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  ctx.restore();
}

export function resizeStars(w, h) {
  initStars(w, h);
}
