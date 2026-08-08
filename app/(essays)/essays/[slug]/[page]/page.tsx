import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { EssayArticle } from "@/app/(essays)/essays/components/essay-article";
import { getBlogPost, getBlogPosts } from "@/lib/blog";
import { createPageMetadata } from "@/lib/metadata";

interface SeriesPageProps {
  params: Promise<{
    slug: string;
    page: string;
  }>;
}

export async function generateMetadata({ params }: SeriesPageProps): Promise<Metadata> {
  const { slug, page } = await params;
  const post = await getBlogPost(`${slug}/${page}`);

  if (!post) {
    return {};
  }

  return createPageMetadata({
    title: post.title,
    description: post.summary || post.title,
    path: post.href,
    type: "article",
  });
}

export async function generateStaticParams() {
  const posts = await getBlogPosts();

  return posts
    .filter((post) => post.series)
    .map((post) => {
      const [slug, page] = post.slug.split("/");
      return { slug, page };
    });
}

export default async function SeriesPage({ params }: SeriesPageProps) {
  const { slug, page } = await params;
  const post = await getBlogPost(`${slug}/${page}`);

  if (!post) {
    notFound();
  }

  return <EssayArticle post={post} />;
}
