// Captures real product UI screenshots + brand tokens from ../index.html.
// Usage: node motion-studio/scripts/capture-brand.mjs
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const studio = path.resolve(here, '..');
const app = path.resolve(studio, '..', 'index.html');
const screens = path.join(studio, 'brand', 'screens');
fs.mkdirSync(screens, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1200 }, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(app).href);
await page.addStyleTag({ content: "body{font-family:Inter,'Segoe UI',sans-serif}" });

// Sample job: a bathroom tile repair.
await page.fill('.mat-name', 'Porcelain tile');
await page.fill('.mat-cost', '4.25');
await page.fill('.mat-qty', '60');
await page.click('.btn-add');
const rows = page.locator('.material-row');
await rows.nth(1).locator('.mat-name').fill('Thinset + grout');
await rows.nth(1).locator('.mat-cost').fill('38');
await rows.nth(1).locator('.mat-qty').fill('2');
await page.fill('#hourlyRate', '45');
await page.fill('#estimatedHours', '10');
await page.fill('#numWorkers', '1');
await page.fill('#travelDistance', '34');
await page.fill('#foodCost', '18');

const sections = page.locator('.section');
await sections.nth(0).screenshot({ path: path.join(screens, 'materials.png') });
await sections.nth(1).screenshot({ path: path.join(screens, 'labor.png') });
await page.locator('.btn-calculate').screenshot({ path: path.join(screens, 'generate-button.png') });

await page.click('.btn-calculate');
await page.locator('#results .section').nth(0).screenshot({ path: path.join(screens, 'breakdown.png') });
await page.locator('#results .section').nth(1).screenshot({ path: path.join(screens, 'tiers.png') });

// Numbers the film shows must match what the app really produces.
const numbers = await page.evaluate(() => ({
  tiers: [...document.querySelectorAll('.tier-card .price')].map(e => e.textContent),
  breakdown: [...document.querySelectorAll('#breakdown .result-row')].map(r => ({
    label: r.querySelector('.label').textContent, value: r.querySelector('.value').textContent,
  })),
}));
fs.writeFileSync(path.join(studio, 'brand', 'sample-estimate.json'), JSON.stringify(numbers, null, 2));

await browser.close();
for (const f of ['icon-192.png', 'icon-512.png']) {
  fs.copyFileSync(path.resolve(studio, '..', f), path.join(studio, 'brand', f));
}
console.log('Brand captured:', fs.readdirSync(screens).join(', '));
console.log(numbers.tiers);
