import puppeteer from 'puppeteer';
(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    page.on('console', msg => {
        if (msg.type() === 'error') {
            console.log('BROWSER ERROR:', msg.text());
        }
    });
    page.on('pageerror', err => {
        console.log('PAGE ERROR:', err.message);
    });
    await page.goto('http://localhost:8080');
    await new Promise(r => setTimeout(r, 6000));
    await browser.close();
})();
