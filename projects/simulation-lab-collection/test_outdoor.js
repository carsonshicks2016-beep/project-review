import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });

  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      if (!msg.text().includes('404') && !msg.text().includes('favicon')) {
        errors.push(`[CONSOLE ERROR] ${msg.text()}`);
      }
    } else {
      console.log(`[PAGE] ${msg.text()}`);
    }
  });

  page.on('pageerror', err => {
    errors.push(`[PAGE ERROR] ${err.message}`);
  });

  console.log('Loading http://localhost:8080...');
  await page.goto('http://localhost:8080');

  // Wait for initial render and first simulation steps
  await new Promise(r => setTimeout(r, 4000));
  await page.screenshot({ path: 'test_outdoor_render.png' });
  console.log('Saved test_outdoor_render.png');

  // Switch to LiDAR Cyber
  console.log('Clicking LiDAR Cyber toggle...');
  await page.click('#btn-env-lidar');
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: 'test_cyber_render.png' });
  console.log('Saved test_cyber_render.png');

  // Switch back to Outdoor
  console.log('Switching back to Outdoor...');
  await page.click('#btn-env-outdoor');
  await new Promise(r => setTimeout(r, 1500));

  // Switch to PPO trainer
  console.log('Switching to PPO trainer...');
  await page.click('#btn-trainer-ppo');
  await new Promise(r => setTimeout(r, 5000));
  await page.screenshot({ path: 'test_ppo_outdoor_render.png' });
  console.log('Saved test_ppo_outdoor_render.png');

  await browser.close();

  if (errors.length > 0) {
    console.error('Test completed with errors:');
    errors.forEach(e => console.error(e));
    process.exit(1);
  } else {
    console.log('All browser verification tests PASSED cleanly!');
  }
})();
