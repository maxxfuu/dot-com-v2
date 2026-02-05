"use client";

import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface NavItem {
  year: number;
  sectionId: string;
}

const navItems: NavItem[] = [
  { year: 2026, sectionId: "section-2026" },
  { year: 2025, sectionId: "section-2025" },
  { year: 2024, sectionId: "section-2024" },
  { year: 2023, sectionId: "section-2023" },
  { year: 2022, sectionId: "section-2022" },
  { year: 2021, sectionId: "section-2021" },
  { year: 2020, sectionId: "section-2020" },
];

export function NavSidebar() {
  const [activeIndex, setActiveIndex] = useState(2); // Default to 2024
  const [isHovered, setIsHovered] = useState(false);
  const [scrollProgress, setScrollProgress] = useState<number[]>(
    navItems.map((_, i) => (i === 2 ? 1 : 0.3))
  );
  const sidebarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      const sections = navItems.map((item) =>
        document.getElementById(item.sectionId)
      );

      const scrollPosition = window.scrollY + window.innerHeight / 2;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollPercent = window.scrollY / docHeight;

      // Calculate which section is most visible
      let newActiveIndex = 0;
      let closestDistance = Infinity;

      sections.forEach((section, index) => {
        if (section) {
          const rect = section.getBoundingClientRect();
          const sectionMiddle = rect.top + rect.height / 2;
          const distance = Math.abs(sectionMiddle - window.innerHeight / 2);

          if (distance < closestDistance) {
            closestDistance = distance;
            newActiveIndex = index;
          }
        }
      });

      // If no sections found, use scroll percentage to determine active
      if (!sections.some((s) => s)) {
        newActiveIndex = Math.min(
          Math.floor(scrollPercent * navItems.length),
          navItems.length - 1
        );
      }

      setActiveIndex(newActiveIndex);

      // Update progress values for each bar
      const newProgress = navItems.map((_, index) => {
        const distanceFromActive = Math.abs(index - newActiveIndex);
        if (distanceFromActive === 0) return 1;
        if (distanceFromActive === 1) return 0.6;
        if (distanceFromActive === 2) return 0.2;
        return 0.25;
      });

      setScrollProgress(newProgress);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // Initial call

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleClick = (index: number) => {
    const section = document.getElementById(navItems[index].sectionId);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav
      ref={sidebarRef}
      className="fixed left-8 top-1/2 -translate-y-1/2 z-50 flex flex-col transition-all duration-300 ease-out"
      style={{ gap: isHovered ? "12px" : "2px" }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {navItems.map((item, index) => {
        const progress = scrollProgress[index];
        const isActive = index === activeIndex;

        // Bar width based on progress (16px to 32px)
        const barWidth = 10 + progress * 12;
        // Opacity based on progress
        const opacity = 0.2 + progress * 0.8;

        return (
          <button
            key={item.year}
            onClick={() => handleClick(index)}
            className="relative flex items-center justify-center h-5 cursor-pointer group"
            style={{ width: "40px" }}
            aria-label={`Navigate to ${item.year}`}
          >
            {/* Bar (default state) */}
            <div
              className={cn(
                "absolute left-0 bg-neutral-800 dark:bg-neutral-200 rounded-full transition-all duration-200 ease-out",
                isHovered && "opacity-0 blur-md scale-0"
              )}
              style={{
                width: `${barWidth}px`,
                height: "2.5px",
                opacity: isHovered ? 0 : opacity,
              }}
            />

            {/* Year label (hover state) - emerges from left where bars are */}
            <span
              className={cn(
                "absolute left-0 text-sm font-medium transition-all duration-200 ease-out whitespace-nowrap",
                isHovered
                  ? "opacity-100 blur-0 scale-100"
                  : "opacity-0 blur-md scale-0",
                isActive
                  ? "text-neutral-900 dark:text-neutral-100 font-semibold"
                  : "text-neutral-400 dark:text-neutral-500"
              )}
              style={{ transformOrigin: "left center" }}
            >
              {item.year}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
