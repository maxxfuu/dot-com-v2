"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

const POLL_INTERVAL_MS = 700;

/**
 * Dev-only live reload for markdown under content/.
 *
 * Those files are read with fs at request time, so editing one produces no
 * HMR event and the page sits stale until a manual refresh. This polls a
 * version stamp and calls router.refresh() when it moves, which re-runs the
 * server components and swaps in the new content without a full page load —
 * scroll position and theme survive.
 */
export function ContentRefresh() {
  const router = useRouter();
  const lastVersion = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const response = await fetch("/api/content-version", { cache: "no-store" });

        if (!response.ok) {
          return;
        }

        const { version } = (await response.json()) as { version: string };

        if (cancelled) {
          return;
        }

        if (lastVersion.current === null) {
          lastVersion.current = version;
        } else if (lastVersion.current !== version) {
          lastVersion.current = version;
          router.refresh();
        }
      } catch {
        // Dev server restarting — just try again on the next tick.
      }
    };

    poll();
    const timer = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [router]);

  return null;
}
