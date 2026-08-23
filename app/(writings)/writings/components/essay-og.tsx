import { ImageResponse } from "next/og";
import { formatBlogDate, getBlogPost } from "@/lib/blog";
import { siteName } from "@/lib/metadata";

export const essayOgAlt = "Essay on maxxfuu.com";
export const essayOgSize = { width: 1200, height: 630 };
export const essayOgContentType = "image/png";

async function loadNewsreader(text: string) {
  // Fetch only the glyphs we need so the serif title matches the site.
  const url = `https://fonts.googleapis.com/css2?family=Newsreader:wght@400&text=${encodeURIComponent(text)}`;
  const css = await fetch(url, {
    headers: {
      // A browser UA makes Google return woff2 (which next/og can't use);
      // an older UA returns a TrueType .ttf that ImageResponse supports.
      "User-Agent":
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8) AppleWebKit/534.30",
    },
  }).then((res) => res.text());

  const fontUrl = css.match(/src: url\((.+?)\) format\('(opentype|truetype)'\)/)?.[1];
  if (!fontUrl) return null;

  return fetch(fontUrl).then((res) => res.arrayBuffer());
}

export async function renderEssayOgImage(slug: string) {
  const post = await getBlogPost(slug);

  const title = post?.title ?? siteName;
  const summary = post?.summary ?? "";
  const date = post ? formatBlogDate(post.date) : "";

  const fontData = await loadNewsreader(`${title}${summary}`).catch(() => null);

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          backgroundColor: "#ffffff",
          padding: "80px",
        }}
      >
        <div
          style={{
            display: "flex",
            fontSize: 28,
            color: "#737373",
            fontFamily: "monospace",
          }}
        >
          maxxfuu
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              display: "flex",
              fontSize: 72,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              color: "#0a0a0a",
              fontFamily: fontData ? "Newsreader" : "serif",
            }}
          >
            {title}
          </div>
          {summary ? (
            <div
              style={{
                display: "flex",
                marginTop: 32,
                fontSize: 32,
                lineHeight: 1.4,
                color: "#525252",
                fontFamily: fontData ? "Newsreader" : "serif",
              }}
            >
              {summary}
            </div>
          ) : null}
        </div>

        <div
          style={{
            display: "flex",
            fontSize: 24,
            color: "#a3a3a3",
            fontFamily: "monospace",
          }}
        >
          {date}
        </div>
      </div>
    ),
    {
      ...essayOgSize,
      fonts: fontData
        ? [{ name: "Newsreader", data: fontData, style: "normal", weight: 400 }]
        : [],
    }
  );
}
