import Link from "next/link";

export function Navbar() {
  return (
    <nav className="flex max-w-xl mx-auto justify-between items-center sticky h-12 top-0 z-50 bg-background/95 opacity-85 backdrop-blur px-8">
      <Link href="/" className="text-sm font-bold">me@maxxfuu.com</Link>
      <ul className="flex space-x-4">
        <li>
          <Link href="/blog" className="text-sm">blog</Link>
        </li>
        <li>
          <Link href="/resume.pdf" target="_blank" className="text-sm">resume</Link>
        </li>
      </ul>
    </nav>
  );
}