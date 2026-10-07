import puppeteer from 'puppeteer';
(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    await page.goto('http://localhost:8080');
    await new Promise(r => setTimeout(r, 6000));
    await page.screenshot({ path: 'test_render.png' });
    await browser.close();
    console.log("Screenshot saved.");
})();
