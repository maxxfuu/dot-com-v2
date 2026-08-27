const { chromium } = require("playwright-core");
const fs = require("fs"), path = require("path");
const EXE = "/home/maxxfuu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
(async () => {
  const files = process.argv.slice(2);
  const b = await chromium.launch({ executablePath: EXE });
  const p = await b.newPage();
  const errs = [];
  p.on("pageerror", e => errs.push(String(e)));
  await p.goto("http://127.0.0.1:8787/validate.html");
  await p.waitForFunction("window.__ready2 === true");
  let bad = 0;
  for (const f of files) {
    const scene = JSON.parse(fs.readFileSync(f, "utf8"));
    const r = await p.evaluate(sc => {
      const out = window.__restore(sc, null, null, { repairBindings: true, refreshDimensions: false });
      return { inN: sc.elements.length, outN: out.elements.length,
               types: [...new Set(out.elements.map(e => e.type))] };
    }, scene);
    const ok = r.inN === r.outN;
    if (!ok) bad++;
    console.log(`${ok ? "ok  " : "FAIL"}  ${path.basename(f).padEnd(34)} ${r.inN} -> ${r.outN}  [${r.types.join(",")}]`);
  }
  if (errs.length) console.log("page errors:", errs.slice(0, 5));
  console.log(bad ? `${bad} FILES FAILED` : "all files restore cleanly");
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
