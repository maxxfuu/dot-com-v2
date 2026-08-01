import { Experience } from "@/components/experience";
import { Projects } from "@/components/projects";
import { Calendar } from "@/components/calendar";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"
import { Separator } from "@/components/ui/separator"
import Link from "next/link";
import { ArrowRightIcon } from "lucide-react";

const inlineLink =
  "border-b border-dashed border-muted-foreground text-foreground transition-colors hover:border-foreground";

export default function Page() {
  return (
    <main className="mx-auto max-w-2xl px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <header className="mb-20">
        <h1 className="text-4xl font-normal leading-tight tracking-tight md:text-5xl">
          Hi, I&apos;m Max
        </h1>
        <div className="my-8 space-y-6 text-base leading-[1.85] text-foreground md:text-[1.05rem]">
          <p>
            I&apos;m a student at UC Merced. I&apos;ve always been drawn to
            work that&apos;s both technically challenging and visually
            appealing, and chasing it has left me with a{" "}
            <HoverCard>
              <HoverCardTrigger asChild>
                <span className={`${inlineLink} cursor-pointer`}>collection</span>
              </HoverCardTrigger>
              <HoverCardContent>
                <p className="font-semibold">Achievements and interests:</p>
                <ul>
                  <li>- Kernel Engineering</li>
                  <li>- Cooking</li>
                  <li>- Powerlifting</li>
                  <li>- Cardistry</li>
                  <li>- Animation Design</li>
                  <li>- Reading</li>
                  <li>- Writing</li>
                  <li>- Filming</li>
                  <li>- Golfing</li>
                </ul>
              </HoverCardContent>
            </HoverCard>{" "}
            of interests, and an obsession with mastery across every one of
            them.
          </p>
          <p>
            Lately, that obsession has pulled me deep into CUDA and kernel
            engineering, model performance, inference infrastructure,
            and everything in between. I&apos;m learning as much as I can, as
            fast as I can, and I&apos;d love to connect with anyone further
            along this path. I hope that in sharing my work, you feel some of the same delight I
            find in it.
          </p>
        </div>
        <Link href="/essays" className="text-sm text-foreground transition-opacity hover:text-foreground flex items-center justify-end gap-2 mb-8">
          Essays 
          <ArrowRightIcon className="w-4 h-4" />
        </Link>
        <Separator />
        <div className="mt-16">
          <Calendar />
          <Experience /> 
          <Projects limit={2} showViewAll />
        </div>
      </header>
    </main>
  );
}
