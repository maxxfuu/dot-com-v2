import fs from "node:fs/promises";
import path from "node:path";

const blogContentDirectory = path.join(process.cwd(), "content", "essays");

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  summary: string;
  body: string;
  href: string;
  // Set when the post is one page of a multi-page essay (a subdirectory of content/essays).
  series: string | null;
  pageNumber: number | null;
}

function normalizeDashes(value: string) {
  return value.replace(/[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]/g, "-");
}

function parseFrontmatter(fileContent: string) {
  const normalizedContent = normalizeDashes(fileContent);
  const frontmatterMatch = normalizedContent.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);

  if (!frontmatterMatch) {
    return {
      metadata: {},
      body: fileContent.trim(),
    };
  }

  const [, frontmatter, body] = frontmatterMatch;
  const metadata = Object.fromEntries(
    frontmatter
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const separatorIndex = line.indexOf(":");

        if (separatorIndex === -1) {
          return [line, ""];
        }

        const key = line.slice(0, separatorIndex).trim();
        const value = normalizeDashes(
          line.slice(separatorIndex + 1).trim().replace(/^"|"$/g, "")
        );

        return [key, value];
      })
  );

  return {
    metadata,
    body: body.trim(),
  };
}

export async function getBlogPosts(): Promise<BlogPost[]> {
  const entries = await fs.readdir(blogContentDirectory, {
    recursive: true,
    withFileTypes: true,
  });

  const markdownFiles = entries.filter(
    (entry) => entry.isFile() && entry.name.endsWith(".md") && entry.name !== "README.md"
  );

  const posts = await Promise.all(
    markdownFiles.map(async (entry) => {
      const relativePath = entry.parentPath
        ? path.relative(blogContentDirectory, path.join(entry.parentPath, entry.name))
        : entry.name;
      const fullPath = path.join(blogContentDirectory, relativePath);
      const fileContent = await fs.readFile(fullPath, "utf8");
      const { metadata, body } = parseFrontmatter(fileContent);
      const segments = relativePath.split(path.sep);
      const slug = segments
        .map((segment, index) =>
          index === segments.length - 1 ? path.basename(segment, ".md") : segment
        )
        .join("/");

      if (!slug) {
        return null;
      }

      const series = segments.length > 1 ? segments[0] : null;
      const pageNumberMatch = series
        ? path.basename(relativePath, ".md").match(/(\d+)$/)
        : null;

      return {
        slug,
        title: String(metadata.title ?? slug),
        date: String(metadata.date ?? "1970-01-01"),
        summary: String(metadata.summary ?? ""),
        body,
        href: `/writings/${slug}`,
        series,
        pageNumber: pageNumberMatch ? Number(pageNumberMatch[1]) : null,
        // Only the first page of a series carries frontmatter; later pages
        // inherit its title/date below so they are not left with slug fallbacks.
        hasFrontmatter: metadata.title !== undefined,
      };
    })
  );

  return inheritSeriesMetadata(
    posts.filter((post): post is BlogPost & { hasFrontmatter: boolean } => post !== null)
  ).sort((a, b) => b.date.localeCompare(a.date));
}

// A series page without its own frontmatter takes the series title, date, and
// summary from the first page, so tab titles and OG images stay meaningful.
function inheritSeriesMetadata(posts: (BlogPost & { hasFrontmatter: boolean })[]): BlogPost[] {
  return posts.map(({ hasFrontmatter, ...post }) => {
    if (hasFrontmatter || !post.series) {
      return post;
    }

    const source = posts
      .filter((other) => other.series === post.series && other.hasFrontmatter)
      .sort(comparePages)[0];

    if (!source) {
      return post;
    }

    return {
      ...post,
      title: source.title,
      date: source.date,
      summary: source.summary,
    };
  });
}

export async function getBlogPost(slug: string) {
  const posts = await getBlogPosts();

  return posts.find((post) => post.slug === slug) ?? null;
}

// Index listing: standalone essays plus only the first page of each series.
export async function getEssayIndex() {
  const posts = await getBlogPosts();

  return posts.filter(
    (post) => !post.series || post === firstPage(posts, post.series)
  );
}

function comparePages(a: BlogPost, b: BlogPost) {
  if (a.pageNumber !== null && b.pageNumber !== null) {
    return a.pageNumber - b.pageNumber;
  }

  return a.slug.localeCompare(b.slug);
}

function firstPage(posts: BlogPost[], series: string) {
  return posts
    .filter((post) => post.series === series)
    .sort(comparePages)[0] ?? null;
}

export async function getSeriesFirstPage(series: string) {
  return firstPage(await getBlogPosts(), series);
}

export interface SeriesNav {
  prev: BlogPost | null;
  next: BlogPost | null;
  index: number;
  total: number;
}

export async function getSeriesNav(post: BlogPost): Promise<SeriesNav | null> {
  if (!post.series) {
    return null;
  }

  const posts = await getBlogPosts();
  const pages = posts
    .filter((other) => other.series === post.series)
    .sort(comparePages);
  const index = pages.findIndex((page) => page.slug === post.slug);

  if (index === -1 || pages.length < 2) {
    return null;
  }

  return {
    prev: pages[index - 1] ?? null,
    next: pages[index + 1] ?? null,
    index,
    total: pages.length,
  };
}

export function formatBlogDate(date: string) {
  const parsedDate = new Date(date);

  if (Number.isNaN(parsedDate.getTime())) {
    return date;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  })
    .format(parsedDate);
}
