"use client";

import { useEffect, useRef } from "react";

const POLL_INTERVAL_MS = 700;

/**
 * Dev-only live reload for markdown under content/.
 *
 * Those files are read with fs at request time, so editing one produces no
 * HMR event and the page sits stale until a manual refresh. This polls a
 * version stamp and reloads when it moves.
 *
 * A full reload rather than router.refresh(): these routes are prerendered
 * via generateStaticParams, and refresh() was being answered from the client
 * router cache, so the page stayed stale while a document request rendered
 * fresh. Browsers restore scroll position across a reload anyway.
 */
export function ContentRefresh() {
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
          window.location.reload();
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
  }, []);

  return null;
}
