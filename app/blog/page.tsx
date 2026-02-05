import { NavSidebar } from "./components/nav-sidebar";

const years = [2026, 2025, 2024, 2023, 2022, 2021, 2020];

export default function Blog() {
  return (
    <>
      <NavSidebar />
      <div className="min-h-screen flex flex-col justify-center items-center">
        <h1 className="text-sm font-semibold">
          A collection of my literary atrocities on anything tech and life
        </h1>
      </div>

      {/* Year sections for scroll navigation */}
      {years.map((year) => (
        <section
          key={year}
          id={`section-${year}`}
          className="min-h-screen flex flex-col justify-center items-center px-8"
        >
          <h2 className="text-6xl font-bold text-neutral-200 dark:text-neutral-800">
            {year}
          </h2>
          <p className="mt-4 text-neutral-400 dark:text-neutral-500 text-center max-w-md">
            Nothing from {year} yet, check back later!
          </p>
        </section>
      ))}
    </>
  );
}