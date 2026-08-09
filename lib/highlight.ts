import { createHighlighter, type Highlighter } from "shiki";

export const codeThemes = { light: "github-light", dark: "github-dark" } as const;

// Loaded once per process; every fenced block in the site is highlighted from
// this instance, so grammar/theme parsing happens a single time per build.
const loadedLanguages = [
  "cpp",
  "c",
  "python",
  "bash",
  "ts",
  "tsx",
  "js",
  "jsx",
  "json",
  "yaml",
  "md",
  "asm",
  "diff",
] as const;

// Shiki ships no CUDA grammar, so .cu code is highlighted as C++ — it covers
// everything except the __global__/__shared__ style qualifiers, which fall
// through as plain identifiers. The fence's own label is what gets displayed.
const languageAliases: Record<string, string> = {
  cuda: "cpp",
  "cuda-cpp": "cpp",
  cu: "cpp",
  "c++": "cpp",
  h: "cpp",
  hpp: "cpp",
  py: "python",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  console: "bash",
  typescript: "ts",
  javascript: "js",
  yml: "yaml",
  markdown: "md",
  // PTX/SASS dumps read closest to generic assembly.
  ptx: "asm",
  sass: "asm",
};

let highlighterPromise: Promise<Highlighter> | null = null;

export function getHighlighter() {
  highlighterPromise ??= createHighlighter({
    themes: [codeThemes.light, codeThemes.dark],
    langs: [...loadedLanguages],
  });

  return highlighterPromise;
}

// Returns null when the fence has no grammar we can use, letting the caller
// fall back to rendering the code unhighlighted rather than throwing.
export function highlightCode(highlighter: Highlighter, code: string, language: string) {
  const normalized = language.trim().toLowerCase();
  const lang = languageAliases[normalized] ?? normalized;

  if (!highlighter.getLoadedLanguages().includes(lang)) {
    return null;
  }

  return highlighter.codeToHtml(code, {
    lang,
    themes: codeThemes,
    // Emit --shiki-light/--shiki-dark custom properties instead of a baked-in
    // color, so the class-based dark mode can swap themes in CSS.
    defaultColor: false,
  });
}
