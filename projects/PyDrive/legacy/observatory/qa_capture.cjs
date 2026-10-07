const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const URL = 'http://localhost:8770/observatory/?edition=919&checkpoint=active-best&mode=replay';
const SCREENSHOT_DIR = path.join(__dirname, 'qa_screenshots');

if (!fs.existsSync(SCREENSHOT_DIR)){
    fs.mkdirSync(SCREENSHOT_DIR);
}

(async () => {
  console.log('Launching Puppeteer...');
  const browser = await puppeteer.launch({
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Set a large viewport for high-quality screenshots
  await page.setViewport({ width: 1920, height: 1080 });
  
  console.log(`Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: 'networkidle2' });
  
  console.log('Waiting 10 seconds for initial world loading and rendering...');
  await new Promise(resolve => setTimeout(resolve, 10000));
  
  const totalShots = 12;
  const intervalMs = 5000;
  
  console.log(`Starting capture sequence: ${totalShots} screenshots at ${intervalMs/1000}s intervals.`);
  
  for (let i = 1; i <= totalShots; i++) {
    const filename = path.join(SCREENSHOT_DIR, `qa_shot_${i.toString().padStart(2, '0')}.png`);
    await page.screenshot({ path: filename });
    console.log(`Saved screenshot ${i}/${totalShots}: ${filename}`);
    
    if (i < totalShots) {
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
  }
  
  console.log('Capture sequence complete. Closing browser.');
  await browser.close();
})();
