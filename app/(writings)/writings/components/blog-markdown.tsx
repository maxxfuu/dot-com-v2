import type { ReactNode } from "react";
import { Fragment } from "react";
import Link from "next/link";
import katex from "katex";
import { CheckUnderstanding } from "@/app/(writings)/writings/components/check-understanding";
import { ZoomableImage } from "@/components/zoomable-image";
import { getHighlighter, highlightCode } from "@/lib/highlight";

const referenceAccentColor = "#036FFF";

// code span | footnote marker | [text](url) | bare url | **bold** | *italic*
// The emphasis markers are fenced by non-word lookaround so arithmetic in prose
// ("2*M*N*K") is left alone; only a * that opens or closes a word counts.
const inlineTokenPattern =
  /(`[^`]+`)|\[\^(\d+)\]|\[([^\]]+)\]\(([^)\s]+)\)|(https?:\/\/[^\s)]+)|(?<![A-Za-z0-9])\*\*([^*\s](?:[^*]*[^*\s])?)\*\*(?![A-Za-z0-9])|(?<![A-Za-z0-9])\*([^*\s](?:[^*]*[^*\s])?)\*(?![A-Za-z0-9])/g;

function renderInlineMarkdown(text: string) {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(inlineTokenPattern)) {
    if (match.index > lastIndex) {
      nodes.push(
        <Fragment key={`text-${lastIndex}`}>
          {text.slice(lastIndex, match.index)}
        </Fragment>
      );
    }

    const [token, codeSpan, refNumber, linkText, linkHref, bareUrl, boldText, italicText] =
      match;

    if (codeSpan) {
      nodes.push(
        <code
          key={`code-${match.index}`}
          // Same hues the github-light/github-dark themes give keywords in the
          // fenced blocks, so an identifier reads the same inline as in code.
          className="font-mono text-[0.9em] text-[#d73a49] dark:text-[#f97583]"
        >
          {codeSpan.slice(1, -1)}
        </code>
      );
    } else if (refNumber) {
      nodes.push(
        <sup key={`ref-${match.index}`}>
          <a
            href={`#ref-${refNumber}`}
            className="font-sans text-[0.72em] font-medium no-underline"
            style={{ color: referenceAccentColor }}
          >
            {refNumber}
          </a>
        </sup>
      );
    } else if (boldText) {
      nodes.push(
        <strong key={`bold-${match.index}`} className="font-semibold">
          {renderInlineMarkdown(boldText)}
        </strong>
      );
    } else if (italicText) {
      nodes.push(
        <em key={`italic-${match.index}`}>{renderInlineMarkdown(italicText)}</em>
      );
    } else {
      const href = linkHref ?? bareUrl;
      const linkClassName =
        "underline decoration-foreground/30 underline-offset-[3px] transition-colors hover:decoration-foreground";

      // Site-relative links stay in the tab and use client-side navigation.
      nodes.push(
        href.startsWith("/") ? (
          <Link key={`link-${match.index}`} href={href} className={linkClassName}>
            {linkText ?? href}
          </Link>
        ) : (
          <a
            key={`link-${match.index}`}
            href={href}
            target="_blank"
            rel="noreferrer"
            className={linkClassName}
          >
            {linkText ?? bareUrl}
          </a>
        )
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(<Fragment key={`text-${lastIndex}`}>{text.slice(lastIndex)}</Fragment>);
  }

  return nodes;
}

const orderedItemPattern = /^(\d+)\.\s+/;

// Figure sizes for the whole blog, named here and chosen per image in markdown:
//   ![caption](/src "small")    square or portrait diagrams
//   ![caption](/src)            medium, the default
//   ![caption](/src "large")
//   ![caption](/src "full")     the full reading column
// Both caps apply, so the tighter one wins and aspect ratio is preserved.
// w-auto everywhere except "full", so an image is never stretched past its own
// pixels unless the author explicitly asks for the whole column.
const figureSizes = {
  small: "w-auto max-h-[16rem] max-w-[50%]",
  medium: "w-auto max-h-[24rem] max-w-[80%]",
  large: "w-auto max-h-[32rem] max-w-[95%]",
  full: "w-full max-h-none max-w-full",
} as const;

// ```latex fences are typeset as display math rather than shown as code.
const mathLanguages = new Set(["latex", "tex", "math"]);

function renderMath(source: string) {
  return katex.renderToString(source.trim(), {
    displayMode: true,
    // Render the error inline instead of failing the whole page build.
    throwOnError: false,
    strict: false,
  });
}

function figureSizeClass(option: string | undefined) {
  const key = option?.trim().toLowerCase() ?? "";

  return key in figureSizes
    ? figureSizes[key as keyof typeof figureSizes]
    : figureSizes.medium;
}

function normalizeHeading(text: string) {
  return text.trim().toLowerCase();
}

// GFM pipe tables. A table is a header row, a delimiter row, then body rows:
//   | load  | naive      | coalesced |
//   |-------|------------|-----------|
//   | A     | 32 sectors | 1 sector  |
// Outer pipes are optional, and a colon in the delimiter row sets the column's
// alignment (`:---` left, `:---:` center, `---:` right).
const tableDelimiterPattern = /^\s*\|?(\s*:?-+:?\s*\|)*\s*:?-+:?\s*\|?\s*$/;

function isTableRow(line: string | undefined) {
  return Boolean(line?.trim().startsWith("|"));
}

// A table needs both lines to commit: a lone `|` line is just a paragraph.
function isTableStart(lines: string[], index: number) {
  return (
    isTableRow(lines[index]) &&
    lines[index + 1] !== undefined &&
    tableDelimiterPattern.test(lines[index + 1]) &&
    lines[index + 1].includes("-")
  );
}

// Split on unescaped pipes, dropping the optional outer pair. `\|` survives as a
// literal so a cell can contain one.
function splitTableRow(line: string) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  const cells: string[] = [];
  let cell = "";

  for (let i = 0; i < trimmed.length; i += 1) {
    if (trimmed[i] === "\\" && trimmed[i + 1] === "|") {
      cell += "|";
      i += 1;
      continue;
    }

    if (trimmed[i] === "|") {
      cells.push(cell.trim());
      cell = "";
      continue;
    }

    cell += trimmed[i];
  }

  cells.push(cell.trim());

  return cells;
}

function tableAlignments(delimiter: string) {
  return splitTableRow(delimiter).map((cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");

    if (left && right) return "text-center";
    if (right) return "text-right";

    return "text-left";
  });
}

interface BlogMarkdownProps {
  body: string;
  title?: string;
}

export async function BlogMarkdown({ body, title }: BlogMarkdownProps) {
  // Awaited once up front so the parse loop below can stay synchronous —
  // codeToHtml itself is sync on an already-created highlighter.
  const highlighter = await getHighlighter();

  // Reference definitions ("[^1]: <source>") can live anywhere in the file;
  // they are pulled out here and rendered as a References section at the end.
  const references = new Map<string, string>();
  const lines = body.split("\n").filter((line) => {
    const referenceMatch = line.match(/^\[\^(\d+)\]:\s*(.+)$/);

    if (referenceMatch) {
      references.set(referenceMatch[1], referenceMatch[2]);
      return false;
    }

    return true;
  });
  const elements: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    // HTML comments are the only way to park a block of draft content in a
    // markdown file. Skip them so they never leak out as paragraph text.
    if (line.trim().startsWith("<!--")) {
      while (index < lines.length && !lines[index].includes("-->")) {
        index += 1;
      }

      index += 1;
      continue;
    }

    if (line.startsWith("```")) {
      // A bare ``` fence renders unlabelled and unhighlighted — the right frame
      // for ASCII diagrams, which are not code in any language.
      const language = line.slice(3).trim();
      const codeLines: string[] = [];
      index += 1;

      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }

      const code = codeLines.join("\n");

      if (mathLanguages.has(language.toLowerCase())) {
        elements.push(
          <div
            key={`math-${index}`}
            className="my-8 overflow-x-auto overflow-y-hidden py-1 text-center"
            dangerouslySetInnerHTML={{ __html: renderMath(code) }}
          />
        );

        index += 1;
        continue;
      }

      const highlighted = highlightCode(highlighter, code, language);

      elements.push(
        <div
          key={`code-${index}`}
          className="my-10 border border-neutral-300 px-5 py-4 dark:border-neutral-700"
        >
          {/* The fence language selects the grammar only; it is never shown. */}
          {highlighted ? (
            <div
              className="code-block overflow-x-auto font-mono text-sm leading-relaxed"
              dangerouslySetInnerHTML={{ __html: highlighted }}
            />
          ) : (
            <pre className="overflow-x-auto font-mono text-sm leading-relaxed text-foreground">
              <code>{code}</code>
            </pre>
          )}
        </div>
      );

      index += 1;
      continue;
    }

    // Check-your-understanding block:
    // :::check
    // <question>
    // ---
    // <answer>
    // :::
    if (line.trim() === ":::check") {
      const questionLines: string[] = [];
      const answerLines: string[] = [];
      let target = questionLines;
      index += 1;

      while (index < lines.length && lines[index].trim() !== ":::") {
        if (lines[index].trim() === "---") {
          target = answerLines;
        } else {
          target.push(lines[index]);
        }
        index += 1;
      }

      index += 1;

      const answerParagraphs = answerLines
        .join("\n")
        .trim()
        .split(/\n\s*\n/)
        .filter(Boolean);

      elements.push(
        <CheckUnderstanding
          key={`check-${index}`}
          question={renderInlineMarkdown(questionLines.join(" ").trim())}
          answer={answerParagraphs.map((paragraph, paragraphIndex) => (
            <p key={paragraphIndex}>
              {renderInlineMarkdown(paragraph.replace(/\n/g, " "))}
            </p>
          ))}
        />
      );

      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);

    if (headingMatch) {
      const [, hashes, headingText] = headingMatch;
      const level = hashes.length;

      if (
        level === 1 &&
        title &&
        normalizeHeading(headingText) === normalizeHeading(title)
      ) {
        index += 1;
        continue;
      }

      if (level === 1) {
        elements.push(
          <h2
            key={`h2-${index}`}
            className="mt-16 mb-6 text-2xl font-medium tracking-tight text-foreground md:text-3xl"
          >
            {renderInlineMarkdown(headingText)}
          </h2>
        );
      } else if (level === 2) {
        elements.push(
          <h3
            key={`h3-${index}`}
            className="mt-12 mb-4 text-xl font-medium tracking-tight text-foreground md:text-2xl"
          >
            {renderInlineMarkdown(headingText)}
          </h3>
        );
      } else if (level === 3) {
        elements.push(
          <h4
            key={`h4-${index}`}
            className="mt-10 mb-3 text-lg font-medium tracking-tight text-foreground"
          >
            {renderInlineMarkdown(headingText)}
          </h4>
        );
      } else {
        elements.push(
          <h5
            key={`h5-${index}`}
            className="mt-8 mb-3 text-base font-medium tracking-tight text-foreground"
          >
            {renderInlineMarkdown(headingText)}
          </h5>
        );
      }

      index += 1;
      continue;
    }

    if (line.startsWith("> ") || line === ">") {
      const calloutLines: string[] = [];

      while (index < lines.length && (lines[index].startsWith("> ") || lines[index] === ">")) {
        calloutLines.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }

      elements.push(
        <aside
          key={`callout-${index}`}
          className="my-10 border-l-2 border-foreground/20 bg-muted/50 px-5 py-4 text-[0.98rem] leading-[1.85] text-muted-foreground"
        >
          {renderInlineMarkdown(calloutLines.join(" "))}
        </aside>
      );

      continue;
    }

    // Trailing markdown title (![alt](/src "title")) is accepted and ignored,
    // so a stray title never silently downgrades a figure to paragraph text.
    // Greedy alt so brackets inside a caption (t[0][0], data[i]) don't end the
    // match early and silently downgrade the figure to paragraph text.
    const mediaMatch = line
      .trim()
      .match(/^!\[(.*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)$/);

    if (mediaMatch) {
      const [, caption, src, sizeOption] = mediaMatch;
      const isVideo = /\.(mp4|webm|mov)$/i.test(src);
      const mediaClassName = `mx-auto block rounded-md ${figureSizeClass(sizeOption)}`;

      elements.push(
        <figure key={`media-${index}`} className="my-10">
          {isVideo ? (
            <video
              src={src}
              autoPlay
              loop
              muted
              playsInline
              className={mediaClassName}
            />
          ) : (
            <ZoomableImage src={src} alt={caption} className={mediaClassName} />
          )}
          {caption ? (
            <figcaption className="mt-3 text-center font-sans text-xs text-muted-foreground">
              {caption}
            </figcaption>
          ) : null}
        </figure>
      );

      index += 1;
      continue;
    }

    if (line.startsWith("- ")) {
      const items: string[] = [];

      while (index < lines.length && lines[index].startsWith("- ")) {
        items.push(lines[index].slice(2));
        index += 1;
      }

      elements.push(
        <ul
          key={`list-${index}`}
          className="my-8 ml-5 list-disc space-y-3 text-base leading-[1.85] text-foreground"
        >
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>
      );

      continue;
    }

    if (orderedItemPattern.test(line)) {
      const items: string[] = [];
      const start = Number(line.match(orderedItemPattern)![1]);

      while (index < lines.length && orderedItemPattern.test(lines[index])) {
        items.push(lines[index].replace(orderedItemPattern, ""));
        index += 1;
      }

      elements.push(
        <ol
          key={`ordered-${index}`}
          start={start}
          className="my-8 ml-5 list-decimal space-y-3 text-base leading-[1.85] text-foreground"
        >
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ol>
      );

      continue;
    }

    if (isTableStart(lines, index)) {
      const header = splitTableRow(lines[index]);
      const alignments = tableAlignments(lines[index + 1]);
      index += 2;

      const rows: string[][] = [];

      while (index < lines.length && isTableRow(lines[index])) {
        rows.push(splitTableRow(lines[index]));
        index += 1;
      }

      elements.push(
        <div key={`table-${index}`} className="my-10 overflow-x-auto">
          <table className="w-full border-collapse text-sm leading-[1.7] text-foreground">
            <thead>
              <tr className="border-b border-neutral-300 dark:border-neutral-700">
                {header.map((cell, cellIndex) => (
                  <th
                    key={`th-${cellIndex}`}
                    className={`px-3 py-2 font-medium ${
                      alignments[cellIndex] ?? "text-left"
                    }`}
                  >
                    {renderInlineMarkdown(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr
                  key={`tr-${rowIndex}`}
                  className="border-b border-neutral-200 last:border-0 dark:border-neutral-800"
                >
                  {/* Pad short rows so the columns stay aligned. */}
                  {header.map((_, cellIndex) => (
                    <td
                      key={`td-${cellIndex}`}
                      className={`px-3 py-2 align-top ${
                        alignments[cellIndex] ?? "text-left"
                      }`}
                    >
                      {renderInlineMarkdown(row[cellIndex] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

      continue;
    }

    const paragraphLines = [line];
    index += 1;

    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].match(/^#{1,6}\s+/) &&
      !lines[index].startsWith("> ") &&
      lines[index] !== ">" &&
      !lines[index].startsWith("- ") &&
      !orderedItemPattern.test(lines[index]) &&
      !lines[index].startsWith("```") &&
      !lines[index].trim().startsWith(":::") &&
      !lines[index].trim().startsWith("![") &&
      !isTableStart(lines, index)
    ) {
      paragraphLines.push(lines[index]);
      index += 1;
    }

    elements.push(
      <p
        key={`p-${index}`}
        className="text-base leading-[1.85] text-foreground md:text-[1.05rem]"
      >
        {renderInlineMarkdown(paragraphLines.join(" "))}
      </p>
    );
  }

  if (references.size > 0) {
    elements.push(
      <section key="references" className="mt-16! border-t border-border pt-8">
        <h2 className="font-sans text-[0.6875rem] font-medium uppercase tracking-[0.2em] text-muted-foreground">
          References
        </h2>
        <ol className="mt-5 space-y-3 text-[0.95rem] leading-relaxed text-muted-foreground">
          {[...references.entries()].map(([number, content]) => (
            <li
              key={number}
              id={`ref-${number}`}
              className="reference-item flex gap-3 scroll-mt-32"
            >
              <span
                className="shrink-0 font-sans text-xs leading-6"
                style={{ color: referenceAccentColor }}
              >
                {number}
              </span>
              <span className="min-w-0 break-words">{renderInlineMarkdown(content)}</span>
            </li>
          ))}
        </ol>
      </section>
    );
  }

  return <div className="space-y-6">{elements}</div>;
}
