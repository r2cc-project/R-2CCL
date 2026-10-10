// node render.mjs [fps] [from_s] [to_s] [outdir] : screenshots demo.html frame by frame
import { chromium } from "playwright-core";
import { mkdirSync } from "fs";
import { resolve } from "path";
const fps = +(process.argv[2] || 30), from = +(process.argv[3] || 0), to = process.argv[4], out = process.argv[5] || "frames";
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto("file://" + resolve("demo.html") + "?export=1");
await page.waitForFunction("window.ready === true");
await page.evaluate("document.fonts.ready");
const total = to ? +to : await page.evaluate("TOTAL");
const stage = await page.$("#stage");
let k = 0;
for (let f = Math.round(from * fps); f < Math.round(total * fps); f++, k++) {
  await page.evaluate(`renderAt(${f / fps})`);
  await stage.screenshot({ path: `${out}/${String(k).padStart(5, "0")}.png` });
}
await browser.close();
console.log(`${k} frames -> ${out}`);
