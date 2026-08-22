"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function navLabel(active: boolean, label: string) {
  return active ? `• ${label}` : label;
}

export function SiteHeader() {
  const pathname = usePathname();
  const isHome = pathname === "/";
  const isEssays = pathname.startsWith("/essays");
  const isResources = pathname === "/resources";
  const isProjects = pathname === "/projects";

  return (
    <header className="mx-auto flex max-w-5xl items-start justify-between px-6 pt-10 md:px-10 md:pt-14">
      <Link
        href="/"
        className="font-sans text-sm tracking-wide text-foreground lowercase transition-opacity hover:opacity-60"
      >
        maxxfuu
      </Link>

      <nav className="flex flex-col items-end gap-1 font-sans text-sm lowercase text-muted-foreground">
        <Link
          href="/"
          className={isHome ? "text-foreground" : "transition-colors hover:text-foreground"}
        >
          {navLabel(isHome, "home")}
        </Link>
        <Link
          href="/essays"
          className={isEssays ? "text-foreground" : "transition-colors hover:text-foreground"}
        >
          {navLabel(isEssays, "essays")}
        </Link>
        <Link
          href="/projects"
          className={isProjects ? "text-foreground" : "transition-colors hover:text-foreground"}
        >
          {navLabel(isProjects, "projects")}
        </Link>
        <Link
          href="/resources"
          className={isResources ? "text-foreground" : "transition-colors hover:text-foreground"}
        >
          {navLabel(isResources, "resources")}
        </Link>
      </nav>
    </header>
  );
}
