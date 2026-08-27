// Normalise text metrics in .excalidraw files, then render each to PNG.
const { chromium } = require("playwright-core");
const fs = require("fs"), path = require("path");
const EXE = "/home/maxxfuu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";

const OUTDIR = process.env.PREVIEW_DIR || "/tmp/claude-1000/-home-maxxfuu-Applications/9fd78ae5-bc49-456a-b9ef-88643db6638c/scratchpad/preview";

(async () => {
  const files = process.argv.slice(2);
  fs.mkdirSync(OUTDIR, { recursive: true });
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  page.on("console", m => { if (m.type() === "error" && !/ERR_FILE_NOT_FOUND/.test(m.text())) console.error("PAGE:", m.text()); });
  await page.goto(process.env.HARNESS_URL + "/page.html");
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });

  // register + load every subset face, then build a measuring context
  await page.evaluate(async () => {
    const FAM = { 1: "Virgil", 2: "Helvetica", 3: "Cascadia", 5: "Excalifont", 6: "Nunito" };
    await document.fonts.ready;
    await Promise.all([...document.fonts].map(f => f.load().catch(() => {})));
    const cvs = document.createElement("canvas");
    const ctx = cvs.getContext("2d");
    window.__measure = (text, fontSize, fontFamily, lineHeight) => {
      const stack = window.__STACKS[FAM[fontFamily]] || window.__STACKS.Excalifont;
      ctx.font = `${fontSize}px ${stack}`;
      const lines = String(text).split("\n");
      let w = 0;
      for (const l of lines) w = Math.max(w, ctx.measureText(l).width);
      return { width: Math.round(w * 100) / 100,
               height: Math.round(lines.length * fontSize * lineHeight * 100) / 100 };
    };
  });

  for (const f of files) {
    const scene = JSON.parse(fs.readFileSync(f, "utf8"));
    const byId = Object.fromEntries(scene.elements.map(e => [e.id, e]));

    const texts = scene.elements.filter(e => e.type === "text");
    const dims = await page.evaluate(
      ts => ts.map(t => window.__measure(t.text, t.fontSize, t.fontFamily, t.lineHeight)),
      texts.map(t => ({ text: t.text, fontSize: t.fontSize, fontFamily: t.fontFamily, lineHeight: t.lineHeight }))
    );

    texts.forEach((t, i) => {
      t.width = dims[i].width;
      t.height = dims[i].height;
      if (t.containerId) {
        const c = byId[t.containerId];
        if (c) {
          if (t.width > c.width - 10) {
            console.warn(`  ! label "${t.text}" (${t.width.toFixed(0)}) overflows container ${c.width}`);
          }
          t.x = Math.round((c.x + (c.width - t.width) / 2) * 100) / 100;
          t.y = Math.round((c.y + (c.height - t.height) / 2 + (t._boundDy || 0)) * 100) / 100;
        }
        delete t._boundDy;
      } else {
        const ax = t._anchorX !== undefined ? t._anchorX : t.x;
        const ay = t._anchorY !== undefined ? t._anchorY : t.y;
        const a = t._anchor || "left", v = t._valign || "top";
        t.x = Math.round((a === "center" ? ax - t.width / 2 : a === "right" ? ax - t.width : ax) * 100) / 100;
        t.y = Math.round((v === "middle" ? ay - t.height / 2 : v === "bottom" ? ay - t.height : ay) * 100) / 100;
        delete t._anchorX; delete t._anchorY; delete t._anchor; delete t._valign;
      }
    });

    fs.writeFileSync(f, JSON.stringify(scene, null, 2));

    const dataUrl = await page.evaluate(async (scene) => {
      const blob = await window.__exportToBlob({
        elements: scene.elements, files: scene.files || {},
        appState: { ...(scene.appState || {}), exportBackground: true,
                    viewBackgroundColor: "#ffffff", exportPadding: 12 },
        mimeType: "image/png", quality: 1,
        getDimensions: (w, h) => ({ width: w * 2, height: h * 2, scale: 2 }),
      });
      return await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
    }, scene);

    const out = path.join(OUTDIR, path.basename(f).replace(/\.excalidraw$/, "") + ".png");
    fs.writeFileSync(out, Buffer.from(dataUrl.split(",")[1], "base64"));
    console.log("rendered", out);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
