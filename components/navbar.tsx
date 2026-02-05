import Link from "next/link";

export function Navbar() {
  return (
    <nav className="max-w-full sticky top-0 z-50 bg-background/95 opacity-85 backdrop-blur">
      <div className="max-w-xl mx-auto flex justify-between items-center h-12">
        <Link href="/" className="text-sm font-bold">me@maxxfuu.com</Link>
        <ul className="flex space-x-4">
          <li>
            <Link href="/blog" className="text-sm">blog</Link>
          </li>
          <li>
            <Link href="/resume.pdf" target="_blank" className="text-sm">resume</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}