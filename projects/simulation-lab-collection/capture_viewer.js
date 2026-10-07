import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:8080/drift_viewer.html');
  await new Promise(r => setTimeout(r, 1000));
  await page.click('#btn-track-gymkhana');
  // Wait 4 seconds for aggressive drifting around arena obstacles
  await new Promise(r => setTimeout(r, 4500));
  await page.screenshot({ path: 'drift_viewer_gymkhana.png' });
  await browser.close();
  console.log("Drift viewer screenshot captured!");
})();
