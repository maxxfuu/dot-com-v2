const fs = require("fs"), path = require("path");
const FONTDIR = "node_modules/@excalidraw/excalidraw/dist/prod/fonts";
const FAMS = { Excalifont: "Excalifont", Cascadia: "Cascadia", Nunito: "Nunito", Virgil: "Virgil" };
const faces = [];
const stacks = {};
for (const [dir, fam] of Object.entries(FAMS)) {
  const files = fs.readdirSync(path.join(FONTDIR, dir)).filter(f => f.endsWith(".woff2")).sort();
  const names = [];
  files.forEach((f, i) => {
    const name = `${fam}__${i}`;
    names.push(name);
    faces.push(`@font-face{font-family:"${name}";src:url("/${FONTDIR}/${dir}/${encodeURIComponent(f)}") format("woff2");font-display:block;}`);
  });
  stacks[fam] = names.map(n => `"${n}"`).join(", ");
}
const html = `<!doctype html><html><head><meta charset="utf-8"><style>
${faces.join("\n")}
</style></head><body>
<script>
window.EXCALIDRAW_ASSET_PATH = "/${FONTDIR.replace(/fonts$/, "")}";
window.__STACKS = ${JSON.stringify(stacks)};
</script>
<script src="/bundle.js"></script>
</body></html>`;
fs.writeFileSync("page.html", html);
console.log("page.html written;", faces.length, "faces");
console.log(JSON.stringify(stacks, null, 1).slice(0, 400));
