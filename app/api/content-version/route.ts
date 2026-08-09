import fs from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const contentDirectory = path.join(process.cwd(), "content");

// Dev-only endpoint backing the live-reload poller. Markdown is read through
// fs at request time, so it is invisible to Turbopack's module graph and a save
// triggers no HMR — this gives the client something to watch instead.
export async function GET() {
  if (process.env.NODE_ENV !== "development") {
    return new NextResponse(null, { status: 404 });
  }

  const entries = await fs.readdir(contentDirectory, {
    recursive: true,
    withFileTypes: true,
  });

  const stats = await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
      .map(async (entry) => {
        const stat = await fs.stat(path.join(entry.parentPath, entry.name));
        return stat.mtimeMs;
      })
  );

  // Count included so deleting a file registers as a change too.
  const version = `${stats.length}:${Math.max(0, ...stats)}`;

  return NextResponse.json({ version }, { headers: { "cache-control": "no-store" } });
}
