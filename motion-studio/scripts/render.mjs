// Renders composition/index.html frame-by-frame and muxes the soundtrack.
// Usage:
//   node scripts/render.mjs                 -> out/film.mp4
//   node scripts/render.mjs --stills 0.5    -> critique/frames/*.jpg every 0.5 s (no video)
//   node scripts/render.mjs --serve         -> preview at http://localhost:4173/composition/
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';

const { chromium } = createRequire(import.meta.url)('playwright');
const studio = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = name => { const i = args.indexOf(name); return i < 0 ? null : (args[i + 1] ?? true); };

const types = { '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.wav': 'audio/wav', '.js': 'text/javascript' };
const server = http.createServer((req, res) => {
  const p = path.join(studio, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!p.startsWith(studio) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    const idx = path.join(p, 'index.html');
    if (fs.existsSync(idx)) { res.writeHead(200, { 'content-type': 'text/html' }); return fs.createReadStream(idx).pipe(res); }
    res.writeHead(404); return res.end();
  }
  res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
});
const port = flag('--serve') ? 4173 : 0;
await new Promise(r => server.listen(port, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

if (flag('--serve')) {
  console.log(`Preview: ${base}/composition/  (Ctrl+C to stop)`);
} else {
  const stillsEvery = flag('--stills') ? parseFloat(flag('--stills')) : null;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', e => { console.error('Composition error:', e.message); process.exit(1); });
  await page.goto(`${base}/composition/index.html?render`);
  const { duration, fps } = await page.evaluate(() => window.ready);
  const stage = page.locator('#stage');

  if (stillsEvery) {
    const dir = path.join(studio, 'critique', 'frames');
    fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
    for (let t = 0, i = 0; t <= duration; t += stillsEvery, i++) {
      await page.evaluate(t => window.seek(t), t);
      await stage.screenshot({ path: path.join(dir, `t${t.toFixed(2).padStart(6, '0')}.jpg`), type: 'jpeg', quality: 85 });
    }
    console.log('Stills written to', dir);
  } else {
    const out = path.join(studio, 'out'); fs.mkdirSync(out, { recursive: true });
    const wav = path.join(out, 'soundtrack.wav');
    if (!fs.existsSync(wav) || flag('--rescore')) {
      const r = spawnSync('python3', [path.join(studio, 'scripts', 'score.py')], { stdio: 'inherit' });
      if (r.status !== 0) process.exit(r.status);
    }
    const frames = Math.round(duration * fps);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error',
      '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
      '-i', wav, '-map', '0:v', '-map', '1:a',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
      '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', '-movflags', '+faststart',
      path.join(out, 'film.mp4')], { stdio: ['pipe', 'inherit', 'inherit'] });
    const t0 = Date.now();
    for (let i = 0; i < frames; i++) {
      await page.evaluate(t => window.seek(t), i / fps);
      const buf = await stage.screenshot({ type: 'jpeg', quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (i % 60 === 0) process.stdout.write(`\rframe ${i}/${frames}`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
    console.log(`\rRendered ${frames} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s -> out/film.mp4`);
  }
  await browser.close();
  server.close();
}
