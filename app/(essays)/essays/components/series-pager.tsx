import Link from "next/link";
import { ArrowLeftIcon, ArrowRightIcon } from "lucide-react";
import type { SeriesNav } from "@/lib/blog";

export function SeriesPager({ nav }: { nav: SeriesNav }) {
  return (
    <nav
      aria-label="Essay pages"
      className="mt-20 flex items-baseline justify-between border-t border-border pt-8"
    >
      {nav.prev ? (
        <Link
          href={nav.prev.href}
          prefetch
          className="flex items-center gap-2 text-lg text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeftIcon className="h-4 w-4" aria-hidden />
          Prev
        </Link>
      ) : (
        <span aria-hidden className="flex items-center gap-2 text-lg text-transparent select-none">
          <span className="h-4 w-4" />
          Prev
        </span>
      )}

      <span className="self-center font-sans text-xs text-muted-foreground">
        {nav.index + 1} / {nav.total}
      </span>

      {nav.next ? (
        <Link
          href={nav.next.href}
          prefetch
          className="flex items-center gap-2 text-lg text-muted-foreground transition-colors hover:text-foreground"
        >
          Next
          <ArrowRightIcon className="h-4 w-4" aria-hidden />
        </Link>
      ) : (
        <span aria-hidden className="flex items-center gap-2 text-lg text-transparent select-none">
          Next
          <span className="h-4 w-4" />
        </span>
      )}
    </nav>
  );
}
