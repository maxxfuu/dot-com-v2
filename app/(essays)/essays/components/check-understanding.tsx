"use client";

import { useState, type ReactNode } from "react";

interface CheckUnderstandingProps {
  question: ReactNode;
  answer: ReactNode;
}

export function CheckUnderstanding({ question, answer }: CheckUnderstandingProps) {
  const [open, setOpen] = useState(false);

  return (
    <aside className="my-12 rounded-lg border border-border bg-muted/40 px-6 py-5">
      <p className="font-sans text-[11px] font-medium uppercase tracking-[0.2em] text-muted-foreground">
        Check your understanding
      </p>

      <div className="mt-3 text-base leading-[1.85] text-foreground md:text-[1.05rem]">
        {question}
      </div>

      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="mt-3 flex items-center gap-2 text-base text-muted-foreground transition-colors hover:text-foreground"
      >
        {open ? "Hide answer" : "Show answer"}
        <svg
          viewBox="0 0 8 10"
          aria-hidden
          className={`h-2.5 w-2 fill-current transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        >
          <path d="M0 0 L8 5 L0 10 Z" />
        </svg>
      </button>

      {open ? (
        <div className="mt-4 space-y-4 border-t border-border pt-4 text-base leading-[1.85] text-muted-foreground">
          {answer}
        </div>
      ) : null}
    </aside>
  );
}
