import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, SquareArrowOutUpRight } from "lucide-react";
import { resources } from "@/lib/resources";
import { createPageMetadata } from "@/lib/metadata";

export const metadata: Metadata = createPageMetadata({
  title: "Resources",
  description: "Textbooks, papers, and courses I've read and would recommend.",
  path: "/resources",
});

export default function ResourcesPage() {
  return (
    <main className="mx-auto max-w-2xl px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <div className="mb-16 font-sans text-sm">
        <Link
          href="/"
          className="flex items-center gap-2 text-muted-foreground transition-opacity hover:text-foreground"
        >
          <ArrowLeftIcon className="h-4 w-4" /> back
        </Link>
      </div>

      <header className="mb-12">
        <h1 className="text-4xl font-normal leading-tight tracking-tight md:text-5xl">
          Resources
        </h1>
        <p className="mt-2 max-w-lg text-base leading-relaxed text-muted-foreground">
          The textbooks, papers, and courses I&apos;ve worked through, and the
          ones I&apos;d hand to anyone walking the same path.
        </p>
      </header>

      {resources.length === 0 ? (
        <p className="text-base leading-relaxed text-muted-foreground">
          Still putting this list together.
        </p>
      ) : (
        <div className="space-y-14">
          {resources.map((group) => (
            <section key={group.category}>
              <h2 className="font-sans text-sm lowercase tracking-wide text-muted-foreground">
                {group.category}
              </h2>
              {group.description ? (
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {group.description}
                </p>
              ) : null}

              <ul className="mt-6 space-y-6">
                {group.items.map((item) => (
                  <li key={`${group.category}-${item.title}`}>
                    <div className="flex items-baseline gap-2">
                      {item.link ? (
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="group inline-flex items-baseline gap-1.5 text-base text-foreground"
                        >
                          <span className="relative after:absolute after:bottom-0 after:left-0 after:h-px after:w-full after:origin-left after:scale-x-0 after:bg-current after:transition-transform after:duration-300 after:ease-out group-hover:after:scale-x-100 motion-reduce:after:scale-x-100 motion-reduce:transition-none">
                            {item.title}
                          </span>
                          <SquareArrowOutUpRight className="h-3 w-3 shrink-0 self-center text-muted-foreground" />
                        </a>
                      ) : (
                        <span className="text-base text-foreground">{item.title}</span>
                      )}
                    </div>
                    {item.author ? (
                      <p className="mt-0.5 font-sans text-sm text-muted-foreground">
                        {item.author}
                      </p>
                    ) : null}
                    {item.note ? (
                      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                        {item.note}
                      </p>
                    ) : null}
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </main>
  );
}
