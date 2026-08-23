import {
  essayOgAlt,
  essayOgContentType,
  essayOgSize,
  renderEssayOgImage,
} from "@/app/(writings)/writings/components/essay-og";
import { getBlogPosts } from "@/lib/blog";

export const alt = essayOgAlt;
export const size = essayOgSize;
export const contentType = essayOgContentType;

export async function generateStaticParams() {
  const posts = await getBlogPosts();
  return posts
    .filter((post) => post.series)
    .map((post) => {
      const [slug, page] = post.slug.split("/");
      return { slug, page };
    });
}

export default async function Image({
  params,
}: {
  params: Promise<{ slug: string; page: string }>;
}) {
  const { slug, page } = await params;
  return renderEssayOgImage(`${slug}/${page}`);
}
