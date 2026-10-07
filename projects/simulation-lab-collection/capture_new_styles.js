import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  // 1. Capture Akina Touge
  await page.goto('http://localhost:8080/drift_viewer.html');
  await new Promise(r => setTimeout(r, 4500));
  await page.screenshot({ path: 'touge_tokyo_style.png' });
  console.log("Touge Tokyo style captured!");

  // 2. Capture Neo Docks Gymkhana
  await page.click('#btn-track-gymkhana');
  await new Promise(r => setTimeout(r, 4500));
  await page.screenshot({ path: 'neodocks_tokyo_style.png' });
  console.log("Neo Docks Tokyo style captured!");

  await browser.close();
})();
