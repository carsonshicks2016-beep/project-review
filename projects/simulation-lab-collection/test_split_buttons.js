import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });

  page.on('console', msg => console.log(`[PAGE] ${msg.text()}`));
  page.on('pageerror', err => {
    console.error(`[PAGE ERROR] ${err.message}`);
    process.exit(1);
  });

  await page.goto('http://localhost:8080');
  await new Promise(r => setTimeout(r, 2000));

  // Enter editor mode so buttons are enabled
  console.log('Entering editor mode...');
  await page.click('#btn-mode-editor');
  await new Promise(r => setTimeout(r, 500));

  // Test Save Outdoor
  console.log('Testing Save Outdoor...');
  await page.click('#btn-save-track-outdoor');
  const outdoorSaved = await page.evaluate(() => localStorage.getItem('track_custom_outdoor'));
  if (!outdoorSaved) throw new Error('Outdoor track not found in localStorage!');
  console.log('Save Outdoor OK, length:', outdoorSaved.length);

  // Test Save LiDAR
  console.log('Testing Save LiDAR...');
  await page.click('#btn-save-track-lidar');
  const lidarSaved = await page.evaluate(() => localStorage.getItem('track_custom_lidar'));
  if (!lidarSaved) throw new Error('LiDAR track not found in localStorage!');
  console.log('Save LiDAR OK, length:', lidarSaved.length);

  // Test Load Outdoor
  console.log('Testing Load Outdoor...');
  await page.click('#btn-load-track-outdoor');
  await new Promise(r => setTimeout(r, 500));

  // Test Load LiDAR
  console.log('Testing Load LiDAR...');
  await page.click('#btn-load-track-lidar');
  await new Promise(r => setTimeout(r, 500));

  // Capture screenshot of track studio with split buttons
  await page.screenshot({ path: 'test_split_buttons.png' });
  console.log('Saved test_split_buttons.png');

  await browser.close();
  console.log('ALL SPLIT BUTTON TESTS PASSED!');
})();
