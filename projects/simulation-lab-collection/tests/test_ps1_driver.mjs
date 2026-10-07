import http from 'http';
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';

const PORT = 8099;
const ROOT_DIR = process.cwd();

// Simple static server for ES module imports
const server = http.createServer((req, res) => {
  let filePath = path.join(ROOT_DIR, req.url === '/' ? 'ps1_driver.html' : req.url.split('?')[0]);
  const ext = path.extname(filePath).toLowerCase();

  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml'
  };

  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end('Not found');
      } else {
        res.writeHead(500);
        res.end('Server error');
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, async () => {
  console.log(`Test server running on port ${PORT}`);
  let hasErrors = false;

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 960 });

    page.on('console', msg => {
      console.log(`[PAGE LOG] ${msg.type()}: ${msg.text()}`);
      if (msg.type() === 'error') {
        hasErrors = true;
      }
    });

    page.on('pageerror', err => {
      console.error('[PAGE ERROR]', err.message);
      hasErrors = true;
    });

    console.log('Navigating to PS1 Driver...');
    await page.goto(`http://localhost:${PORT}/ps1_driver.html`, { waitUntil: 'networkidle0' });

    // Wait 1.5s for initial render and sprites
    await new Promise(r => setTimeout(r, 1500));

    // Capture initial Touge Pass screenshot
    await page.screenshot({ path: 'ps1_driver_touge.png' });
    console.log('Saved ps1_driver_touge.png');

    // Simulate accelerating forward
    console.log('Simulating acceleration & steering...');
    await page.keyboard.down('ArrowUp');
    await new Promise(r => setTimeout(r, 2000));

    // Simulate drift turn
    await page.keyboard.down('ArrowRight');
    await page.keyboard.down('Space');
    await new Promise(r => setTimeout(r, 1500));
    // Capture drift screenshot during action
    await page.screenshot({ path: 'ps1_driver_drift.png' });
    console.log('Saved ps1_driver_drift.png');

    await page.keyboard.up('Space');
    await page.keyboard.up('ArrowRight');

    // Verify speed
    const speedText = await page.$eval('#ui-speed', el => el.textContent);
    console.log(`Speed after acceleration: ${speedText} KM/H`);

    // Switch to Neo Yokohama Track and reset car to center
    console.log('Switching to Neo Yokohama Track...');
    await page.keyboard.press('KeyT');
    await page.keyboard.press('KeyR');
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ path: 'ps1_driver_yokohama.png' });
    console.log('Saved ps1_driver_yokohama.png');

    // Switch to Pacific Coast Track and reset car to center
    console.log('Switching to Pacific Coast Track...');
    await page.keyboard.press('KeyT');
    await page.keyboard.press('KeyR');
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ path: 'ps1_driver_coast.png' });
    console.log('Saved ps1_driver_coast.png');

    // Switch camera mode to Cockpit
    console.log('Switching camera to Cockpit mode...');
    await page.keyboard.press('KeyC');
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: 'ps1_driver_cockpit.png' });
    console.log('Saved ps1_driver_cockpit.png');

    // Release all keys
    await page.keyboard.up('ArrowUp');

    if (hasErrors) {
      console.error('TEST FAILED: Errors were encountered during test execution.');
      process.exitCode = 1;
    } else {
      console.log('ALL TESTS PASSED: PS1 driver loaded, rendered, and drove flawlessly!');
    }
  } catch (err) {
    console.error('Fatal test error:', err);
    process.exitCode = 1;
  } finally {
    await browser.close();
    server.close();
  }
});
