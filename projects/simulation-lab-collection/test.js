import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:\n', err.stack));
  
  await page.goto('http://localhost:8080');
  await new Promise(r => setTimeout(r, 3000));
  
  await browser.close();
})();
