import type { Metadata } from "next";
import { createPageMetadata } from "@/lib/metadata";

export const metadata: Metadata = createPageMetadata({
  title: "Resume",
  description: "Max Fu's resume.",
  path: "/resume",
});

export default function ResumePage() {
  return (
    <main className="mx-auto max-w-4xl px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <header className="mb-10">
        <h1 className="text-4xl font-normal leading-tight tracking-tight md:text-5xl">
          Resume
        </h1>
        <p className="mt-4 font-sans text-sm text-muted-foreground">
          <a
            href="/resume.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-foreground"
          >
            Open in new tab.
          </a>
        </p>
      </header>

      <iframe
        src="/resume.pdf#toolbar=0&navpanes=0&view=FitH"
        title="Max Fu Resume"
        className="aspect-[8.5/11] w-full border border-border bg-background"
      />
    </main>
  );
}
