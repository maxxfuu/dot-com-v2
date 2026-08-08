import Link from "next/link";
import { BlogMarkdown } from "@/app/(essays)/essays/components/blog-markdown";
import { SeriesPager } from "@/app/(essays)/essays/components/series-pager";
import { formatBlogDate, getSeriesNav, type BlogPost } from "@/lib/blog";
import { ArrowLeftIcon } from "lucide-react";

export async function EssayArticle({ post }: { post: BlogPost }) {
  const seriesNav = await getSeriesNav(post);

  return (
    <main className="mx-auto max-w-2xl px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <div className="mb-16 font-sans text-sm">
        <Link href="/essays" className="text-muted-foreground transition-opacity hover:text-foreground flex items-center gap-2">
          <ArrowLeftIcon className="w-4 h-4" /> back
        </Link>
      </div>

      <article>
        <header className="mb-16 text-center">
          <h1 className="text-4xl font-normal leading-[1.15] tracking-tight md:text-5xl">
            {post.title}
          </h1>
          {post.summary ? (
            <p className="mx-auto mt-6 max-w-lg text-lg leading-relaxed text-muted-foreground">
              {post.summary}
            </p>
          ) : null}
          <p className="mt-8 font-sans text-xs text-muted-foreground">
            {formatBlogDate(post.date)}
          </p>
        </header>

        <BlogMarkdown body={post.body} title={post.title} />
      </article>

      {seriesNav ? <SeriesPager nav={seriesNav} /> : null}
    </main>
  );
}
