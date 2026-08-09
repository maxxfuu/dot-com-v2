import type { ReactNode } from "react";
import { Fragment } from "react";
import Link from "next/link";
import { CheckUnderstanding } from "@/app/(essays)/essays/components/check-understanding";

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
          className="font-mono text-[0.9em] text-muted-foreground"
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

function normalizeHeading(text: string) {
  return text.trim().toLowerCase();
}

interface BlogMarkdownProps {
  body: string;
  title?: string;
}

export function BlogMarkdown({ body, title }: BlogMarkdownProps) {
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

    if (line.startsWith("```")) {
      const language = line.slice(3).trim() || "text";
      const codeLines: string[] = [];
      index += 1;

      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }

      elements.push(
        <pre
          key={`code-${index}`}
          className="my-10 overflow-x-auto border-y border-border py-6 font-mono text-sm leading-relaxed text-foreground"
        >
          <div className="mb-3 font-sans text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            {language}
          </div>
          <code>{codeLines.join("\n")}</code>
        </pre>
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

    const mediaMatch = line.trim().match(/^!\[([^\]]*)\]\(([^)\s]+)\)$/);

    if (mediaMatch) {
      const [, caption, src] = mediaMatch;
      const isVideo = /\.(mp4|webm|mov)$/i.test(src);

      elements.push(
        <figure key={`media-${index}`} className="my-10">
          {isVideo ? (
            <video
              src={src}
              autoPlay
              loop
              muted
              playsInline
              className="w-full rounded-md border border-border"
            />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={src}
              alt={caption}
              className="w-full rounded-md border border-border"
            />
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
      !lines[index].trim().startsWith("![")
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
        <h2 className="font-sans text-[11px] font-medium uppercase tracking-[0.2em] text-muted-foreground">
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
