const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const URL = pathToFileURL(path.join(__dirname, 'proto_standalone.html')).href + '?clean=1&t=0';

const FPS = 30;
const DURATION = 12;
const W = 1920, H = 1080;
const TOTAL = FPS * DURATION;
const OUT = path.join(__dirname, 'frames_proto');

(async () => {
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: false,
    args: [
      '--window-size=1920,1200',
      '--window-position=-2400,-100',
      '--no-sandbox',
      '--disable-gpu-sandbox',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.log('PAGEERROR:', e.message));

  await page.goto(URL, { waitUntil: 'load', timeout: 90000 });
  await page.waitForFunction('window.__ready === true', { timeout: 90000 });
  await new Promise(r => setTimeout(r, 2000));

  for (let i = 0; i < TOTAL; i++) {
    const t = i / (TOTAL - 1);
    await page.evaluate(t => window.__seek(t), t);
    await page.screenshot({ path: path.join(OUT, `f${String(i).padStart(4, '0')}.png`) });
    if (i % 30 === 0) console.log(`frame ${i}/${TOTAL}`);
  }

  await browser.close();
  console.log('done');
})().catch(e => { console.error(e); process.exit(1); });
