import Link from "next/link";
import type { Metadata } from "next";
import { formatBlogDate, getEssayIndex } from "@/lib/blog";
import { createPageMetadata } from "@/lib/metadata";

export const metadata: Metadata = createPageMetadata({
  title: "Essays",
  description: "A collection of my literary atrocities on my interests, passions, and experiences...",
  path: "/essays",
});

export default async function Essays() {
  const posts = await getEssayIndex();

  return (
    <main className="mx-auto max-w-2xl px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <header className="mb-20 text-center">
        <h1 className="text-4xl font-normal leading-tight tracking-tight md:text-5xl">
          Essays
        </h1>
        <p className="mx-auto mt-5 max-w-lg font-sans text-sm leading-relaxed text-muted-foreground">
          A collection of my literary atrocities on my interests, passions, and experiences.
        </p>
      </header>

      {posts.length > 0 ? (
        <ol className="divide-y divide-border border-t border-border">
          {posts.map((post) => (
            <li key={post.slug}>
              <Link
                href={post.href}
                prefetch
                className="group block py-10 transition-opacity hover:opacity-70"
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-baseline md:justify-between">
                  <h2 className="text-2xl font-normal leading-snug tracking-tight md:text-[1.75rem]">
                    {post.title}
                  </h2>
                  <p className="shrink-0 font-sans text-xs text-muted-foreground">
                    {formatBlogDate(post.date)}
                  </p>
                </div>
                {post.summary ? (
                  <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
                    {post.summary}
                  </p>
                ) : null}
              </Link>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-center font-sans text-sm text-muted-foreground">
          nothing here yet
        </p>
      )}
    </main>
  );
}
