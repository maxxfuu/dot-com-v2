import type { Metadata } from "next";

export const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "https://maxxfuu.com";
export const siteName = "Max Fu";
export const defaultDescription =
  "Software engineer, entrepreneur, and student.";

// Previously the social preview image was hardcoded to the favicon, which is
// why sharing the site (e.g. in iMessage) showed the favicon as the card image.
// Open Graph images are now generated dynamically via file-based
// `opengraph-image` routes (app/opengraph-image.tsx for the site default and
// app/(writings)/writings/[slug]/opengraph-image.tsx per essay).
// const defaultOgImage = {
//   url: "/favicon.ico",
//   width: 1200,
//   height: 630,
//   alt: siteName,
// };

export function createPageMetadata({
  title,
  description = defaultDescription,
  path = "/",
  type = "website",
}: {
  title: string;
  description?: string;
  path?: string;
  type?: "website" | "article";
}): Metadata {
  const url = `${siteUrl}${path}`;
  const pageTitle = path === "/" ? title : `${title} · ${siteName}`;

  return {
    title: pageTitle,
    description,
    alternates: {
      canonical: url,
    },
    openGraph: {
      title,
      description,
      url,
      siteName,
      locale: "en_US",
      type,
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
    },
  };
}

export const rootMetadata: Metadata = {
  metadataBase: new URL(siteUrl),
  icons: {
    icon: "/favicon.ico",
  },
  ...createPageMetadata({
    title: siteName,
    description: defaultDescription,
    path: "/",
  }),
};
