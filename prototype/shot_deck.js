const puppeteer = require('puppeteer-core');
const { pathToFileURL } = require('url');
const fs = require('fs');
const path = require('path');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const FILE = path.join(__dirname, '..', '少走弯路_FlowMatching.html');
const URL = pathToFileURL(FILE).href;
const OUT = path.join(__dirname, 'shots');
const only = process.argv.slice(2).map(Number);

(async () => {
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: false,
    args: [
      '--window-size=1920,1200',
      '--window-position=-2400,-100',
      '--no-sandbox',
      '--disable-gpu-sandbox',
      '--autoplay-policy=no-user-gesture-required',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.log('PAGEERROR:', e.message));
  page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE:', m.text()); });

  await page.goto(URL, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__deckReady === true', { timeout: 180000 });
  await new Promise(r => setTimeout(r, 900));
  fs.mkdirSync(OUT, { recursive: true });

  const list = only.length ? only : [0,1,2,3,4,5,6,7,8,9,10];
  for (const i of list) {
    if (i !== 0) {
      await page.evaluate(n => window.__go(n), i);
    }
    await new Promise(r => setTimeout(r, i >= 5 ? 2600 : 1300));
    const name = path.join(OUT, `deck_${String(i).padStart(2, '0')}.png`);
    await page.screenshot({ path: name });
    console.log('shot', name);
  }

  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
