/**
 * Automated Verification Suite for ApexFlock
 * Tests WebGL canvas, Three.js scene, neural forward passes,
 * multi-agent simulation physics, and captures screenshot artifacts.
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';

const PORT = 8085;
const ROOT_DIR = process.cwd();

// Simple static file server for testing
function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let filePath = path.join(ROOT_DIR, req.url.split('?')[0]);
      if (filePath.endsWith('/')) filePath = path.join(filePath, 'boids.html');

      fs.readFile(filePath, (err, data) => {
        if (err) {
          res.writeHead(404);
          res.end('Not found');
          return;
        }
        let contentType = 'text/plain';
        if (filePath.endsWith('.html')) contentType = 'text/html';
        else if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) contentType = 'application/javascript';
        else if (filePath.endsWith('.css')) contentType = 'text/css';
        else if (filePath.endsWith('.json')) contentType = 'application/json';
        else if (filePath.endsWith('.png')) contentType = 'image/png';

        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
      });
    });

    server.listen(PORT, () => {
      console.log(`Test server running at http://localhost:${PORT}`);
      resolve(server);
    });
  });
}

async function runTests() {
  const server = await startServer();
  console.log("Launching headless browser...");

  const browser = await puppeteer.launch({
    headless: true
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const pageErrors = [];
  page.on('pageerror', err => {
    console.error("PAGE ERROR:", err.message);
    pageErrors.push(err.message);
  });

  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error("CONSOLE ERROR:", msg.text());
    }
  });

  console.log("Navigating to boids.html...");
  await page.goto(`http://localhost:${PORT}/boids.html`, { waitUntil: 'networkidle0' });

  // Wait for simulation initialization
  await page.waitForFunction(() => window.app && window.app.ecosystem && window.app.ecosystem.boids.length > 0, { timeout: 10000 });

  console.log("Checking simulation status...");
  const status = await page.evaluate(() => {
    const eco = window.app.ecosystem;
    return {
      boidCount: eco.boids.length,
      predCount: eco.predators.length,
      foodCount: eco.foods.length,
      selectedType: window.app.brainViz.agent?.type,
      selectedId: window.app.brainViz.agent?.id,
      boidBrainInputs: window.app.brainViz.agent?.brain?.activations[0]?.length,
      boidBrainOutputs: window.app.brainViz.agent?.brain?.activations[3]?.length
    };
  });

  console.log("Initial state:", status);

  if (status.boidCount < 50) throw new Error(`Expected at least 50 boids, got ${status.boidCount}`);
  if (status.predCount < 3) throw new Error(`Expected at least 3 predators, got ${status.predCount}`);
  if (status.foodCount < 20) throw new Error(`Expected at least 20 foods, got ${status.foodCount}`);
  if (!status.selectedType) throw new Error("Expected an agent to be auto-selected for Brain HUD inspection");

  console.log("Simulating 60 physics cycles...");
  await page.evaluate(() => {
    for (let i = 0; i < 60; i++) {
      window.app.ecosystem.update(0.016);
    }
  });

  // Test Radiation Mutation pulse
  console.log("Testing Radiation Pulse (mass mutation)...");
  await page.evaluate(() => {
    const btn = document.getElementById('btn-mutate');
    if (btn) btn.click();
  });

  // Wait for 1.5 seconds of live animation in Orbit mode
  await new Promise(r => setTimeout(r, 1500));

  // Capture Orbit overview screenshot
  const screenshotPath = path.join(ROOT_DIR, 'boids_simulation_overview.png');
  await page.screenshot({ path: screenshotPath });
  console.log(`Saved screenshot to ${screenshotPath}`);

  // Test Camera switching to Chase mode
  console.log("Testing Chase Cam...");
  await page.evaluate(() => {
    const btnChase = document.querySelector('[data-cam="chase"]');
    if (btnChase) btnChase.click();
  });
  await new Promise(r => setTimeout(r, 1200));

  const chaseScreenshotPath = path.join(ROOT_DIR, 'boids_chase_cam.png');
  await page.screenshot({ path: chaseScreenshotPath });
  console.log(`Saved Chase Cam screenshot to ${chaseScreenshotPath}`);

  // Switch to first person cam
  console.log("Testing First-Person Cam...");
  await page.evaluate(() => {
    const btn1st = document.querySelector('[data-cam="firstPerson"]');
    if (btn1st) btn1st.click();
  });
  await new Promise(r => setTimeout(r, 1000));

  const fpvScreenshotPath = path.join(ROOT_DIR, 'boids_first_person.png');
  await page.screenshot({ path: fpvScreenshotPath });
  console.log(`Saved FPV screenshot to ${fpvScreenshotPath}`);

  // Test Sumi-e Theme
  console.log("Testing Sumi-e Theme...");
  await page.evaluate(() => {
    const btnSumi = document.querySelector('[data-theme="sumie"]');
    if (btnSumi) btnSumi.click();
    const btnOrbit = document.querySelector('[data-cam="orbit"]');
    if (btnOrbit) btnOrbit.click();
  });
  await new Promise(r => setTimeout(r, 1000));
  const sumiPath = path.join(ROOT_DIR, 'boids_theme_sumie.png');
  await page.screenshot({ path: sumiPath });
  console.log(`Saved Sumi-e screenshot to ${sumiPath}`);

  // Test Alabaster Theme
  console.log("Testing Alabaster Theme...");
  await page.evaluate(() => {
    const btnAla = document.querySelector('[data-theme="alabaster"]');
    if (btnAla) btnAla.click();
  });
  await new Promise(r => setTimeout(r, 1000));
  const alaPath = path.join(ROOT_DIR, 'boids_theme_alabaster.png');
  await page.screenshot({ path: alaPath });
  console.log(`Saved Alabaster screenshot to ${alaPath}`);

  // Test Murmuration Scenario
  console.log("Testing Murmuration Scenario...");
  await page.evaluate(() => {
    const btnMurm = document.querySelector('[data-scenario="murmuration"]');
    if (btnMurm) btnMurm.click();
    const btnAbyss = document.querySelector('[data-theme="abyssal"]');
    if (btnAbyss) btnAbyss.click();
  });
  await new Promise(r => setTimeout(r, 1500));
  const murmPath = path.join(ROOT_DIR, 'boids_murmuration_scenario.png');
  await page.screenshot({ path: murmPath });
  console.log(`Saved Murmuration screenshot to ${murmPath}`);

  // Verify Phase Plane has updated
  const historyLen = await page.evaluate(() => window.app.ecosystem.history.length);
  console.log(`Ecological history recorded: ${historyLen} snapshots`);
  if (historyLen === 0) throw new Error("Expected history telemetry to be recorded");

  if (pageErrors.length > 0) {
    throw new Error(`Test failed with ${pageErrors.length} unhandled page errors: ${pageErrors.join('; ')}`);
  }

  console.log("All automated verification checks passed successfully!");
  await browser.close();
  server.close();
  process.exit(0);
}

runTests().catch(err => {
  console.error("TEST FAILED:", err);
  process.exit(1);
});
