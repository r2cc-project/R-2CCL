import { chromium } from "playwright-core";
import { resolve } from "path";
const times = process.argv.slice(2).map(Number);
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on("pageerror", e => errs.push(String(e))); page.on("console", m => m.type() === "error" && errs.push(m.text()));
await page.goto("file://" + resolve("demo.html") + "?export=1");
await page.waitForFunction("window.ready === true", null, { timeout: 10000 }).catch(() => {});
await page.evaluate("document.fonts.ready");
for (const t of times) { await page.evaluate(`renderAt(${t})`); await (await page.$("#stage")).screenshot({ path: `still_${t}.png` }); }
console.log(errs.length ? "ERRORS:\n" + errs.join("\n") : "no errors");
await browser.close();
