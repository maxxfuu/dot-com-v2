import type { Metadata } from "next";
import { notFound, redirect } from "next/navigation";
import { EssayArticle } from "@/app/(writings)/writings/components/essay-article";
import { getBlogPost, getBlogPosts, getSeriesFirstPage } from "@/lib/blog";
import { createPageMetadata } from "@/lib/metadata";

interface BlogPostPageProps {
  params: Promise<{
    slug: string;
  }>;
}

export async function generateMetadata({ params }: BlogPostPageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = await getBlogPost(slug);

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
    .filter((post) => !post.series)
    .map((post) => ({
      slug: post.slug,
    }));
}

export default async function BlogPostPage({ params }: BlogPostPageProps) {
  const { slug } = await params;
  const post = await getBlogPost(slug);

  if (!post) {
    // A bare series slug (e.g. /writings/article4) lands on its first page.
    const seriesFirstPage = await getSeriesFirstPage(slug);

    if (seriesFirstPage) {
      redirect(seriesFirstPage.href);
    }

    notFound();
  }

  return <EssayArticle post={post} />;
}
