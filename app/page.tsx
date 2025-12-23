import { Experience } from "@/components/experience";
import { Projects } from "@/components/projects";
import Link from "next/link";

export default function Page() {
  return (
    <div className="max-w-xl mx-auto tracking-tight my-18 px-8">
      <div className="flex flex-col">
        <h1>hey, im max!</h1>
        <p>im a{" "}
          <Link href="https://www.ucmerced.edu/" target="_blank">
            <span className="inline-flex items-center text-primary border-b border-dashed border-muted-foreground cursor-pointer">student</span>,{" "}
          </Link>
          <Link href="https://www.linkedin.com/in/maxxfuu" target="_blank">
            <span className="inline-flex items-center text-primary border-b border-dashed border-muted-foreground cursor-pointer">software engineer</span>,{" "}
          </Link>
          <span className="inline-flex items-center text-primary border-b border-dashed border-muted-foreground">entrepreneur</span>{", "}
          and{" "}
          <Link href="https://www.instagram.com/max_lyfts" target="_blank">
            <span className="inline-flex items-center text-primary border-b border-dashed border-muted-foreground cursor-pointer">powerlifter</span>
          </Link>
        </p>
      </div>
      <br />
      <Experience />
      <Projects />
    </div>
  );
}