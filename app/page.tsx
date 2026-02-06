import { Experience } from "@/components/experience";
import { Projects } from "@/components/projects";
import Link from "next/link";

export default function Page() {
  return (
    <div className="max-w-2xl mx-auto tracking-tight my-24 px-8">
      <div className="flex flex-col">
        <h1 className="text-4xl">i'm <span className="font-bold">max</span>{", a "}
          <Link href="https://www.linkedin.com/in/maxxfuu" target="_blank">
            <span className="inline-flex items-center text-primary border-b border-dashed border-muted-foreground cursor-pointer">software engineer*</span>{" "}
          </Link>
          obsessed with design, web developement, and systems programming.
        </h1>
        <p className="font-normal text-gray-500 my-8 max-w-xl dark:text-gray-200">
          * im a {" full-time "}
          <Link href="https://www.linkedin.com/in/maxxfuu" target="_blank">
            <span className="inline-flex items-center text-gray-500 border-b border-dashed border-muted-foreground dark:text-gray-200 cursor-pointer">student</span>
          </Link>, and when i'm not studying or building, i'm experimenting with new recipes, pursuing {" "}
          <Link href="https://www.instagram.com/max_lyfts" target="_blank">
            <span className="inline-flex items-center text-gray-500 border-b border-dashed border-muted-foreground dark:text-gray-200 cursor-pointer">powerlifting</span>
          </Link>,
         and occasionally snowboarding.
        </p>
      </div>

      <br />
      <Experience />
      <Projects />
    </div>
  );
}