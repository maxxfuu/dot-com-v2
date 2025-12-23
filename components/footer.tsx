import Link from "next/link";

export function Footer() {
  return (
    <div className="h-18 bg-secondary/40 border-t border-muted-foreground/10 text-muted-foreground text-xs">
      <footer className="min-h-18 flex flex-col items-center justify-center gap-2.5">
        <p>For more, check out my
          {" "}
          <Link href="https://www.linkedin.com/in/maxxfuu" target="_blank" className="text-primary border-b border-dashed border-muted-foreground cursor-pointer">LinkedIn</Link>,
          {" "}
          <Link href="https://github.com/maxxfuu" target="_blank" className="text-primary border-b border-dashed border-muted-foreground cursor-pointer">Github</Link>,
          {" "}
          or
          {" "}
          <Link href="https://leetcode.com/maxxfuu" target="_blank" className="text-primary border-b border-dashed border-muted-foreground cursor-pointer">LeetCode</Link>
        </p>
        <p className="text-muted-foreground text-xs">© 2025 Max Fu. All rights reserved.</p>
      </footer>
    </div>
  );
}