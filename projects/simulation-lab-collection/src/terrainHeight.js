// Terrain elevation, kept in its own module with NO imports.
//
// This function has to run in headless workers, and module workers do not inherit the
// page's importmap -- so anything that (even transitively) imports three by bare specifier
// cannot be loaded there. Living alongside the three.js mesh builders in outdoorTerrain.js
// silently made this unusable in a worker.
export function getOutdoorElevation(x, z) {
  const dist = Math.hypot(x, z);
  
  // Inner racing valley (r < 150m): gentle rolling hills between 0.5m and 6m
  // Outer perimeter (r > 150m): dramatic alpine mountain peaks rising to 70m+
  const valleyFactor = Math.min(1.0, Math.max(0.0, (dist - 120) / 140));
  
  // Smooth rolling hills in the valley
  const hValley = 
    Math.sin(x * 0.02 + 0.3) * Math.cos(z * 0.025) * 2.5 +
    Math.sin(x * 0.05 + 1.2) * Math.cos(z * 0.045 + 0.8) * 1.5 +
    Math.sin(x * 0.01 - z * 0.015) * 2.0 + 3.0;

  // Mountain ridges and jagged peaks for the perimeter
  const hMountains =
    Math.sin(x * 0.012) * Math.cos(z * 0.012) * 25.0 +
    Math.sin(x * 0.024 + 1.5) * Math.cos(z * 0.028 - 0.5) * 16.0 +
    Math.sin(x * 0.05 - 0.7) * Math.cos(z * 0.05 + 1.1) * 8.0 +
    Math.pow(Math.max(0, dist - 150) / 45, 1.35) * 6.0;

  // Blend valley into surrounding mountains
  let h = (1.0 - valleyFactor) * hValley + valleyFactor * (hValley * 0.5 + hMountains);

  // Guarantee minimum floor so water depressions stay above y = 0
  return Math.max(0.0, h);
}

// ── Procedural Low-Poly Terrain Mesh ──
