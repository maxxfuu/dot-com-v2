"use client"

import Link from "next/link";
import { DropdownMenu, DropdownMenuTrigger } from "./ui/dropdown-menu";
import { Button } from "./ui/button";
import { Sun } from "lucide-react";
import { Moon } from "lucide-react";
import { DropdownMenuContent } from "./ui/dropdown-menu";
import { DropdownMenuItem } from "./ui/dropdown-menu";
import { useTheme } from "next-themes";

export function Navbar() {
  const { setTheme } = useTheme();
  return (
    <nav className="max-w-full sticky top-0 z-50 bg-background/95 opacity-85 backdrop-blur px-8">
      <div className="max-w-xl mx-auto flex justify-between items-center h-12">
        <Link href="/" className="text-sm font-bold">me@maxxfuu.com</Link>
        <ul className="flex flex-row items-center space-x-4">
          <li>
            <Link href="/blog" className="text-sm">blog</Link>
          </li>
          <li>
            <Link href="/resume.pdf" target="_blank" className="text-sm">resume</Link>
          </li>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <Sun className="h-[1.2rem] w-[1.2rem] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
                <Moon className="absolute h-[1.2rem] w-[1.2rem] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme("light")}>
                Light
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("dark")}>
                Dark
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("system")}>
                System
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </ul>
      </div>
    </nav>
  );
}