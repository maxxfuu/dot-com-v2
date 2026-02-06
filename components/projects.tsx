"use client";

import { Card  } from "./ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { SquareArrowOutUpRight } from 'lucide-react';


interface Project {
  title: string;
  description: string;
  image: string;
  type?: string;
  github?: string;
  video?: string;
  link?: string;
}

const projects: Record<string, Project> = {
  project1: {
    title: "ClockIn",
    description: "An app that helps you stay accountable by recording timelapses of you working, so you can “clock in” with social proof",
    image: "/clockin.png",
    type: "image",
    link: "https://clockin.now",
  },
  project2: {
    title: "Yagmi",
    description: "Create AI Mentors that lets you chat with an “advisor” tailored to your goals, giving you guidance and next steps as you grow",
    image: "/yagmi.png",
    type: "image",
    link: "https://yagmi.app",
  },
  project3: {
    title: "Obscuri",
    description: "A desktop application that silently awaits for your screenshots and querys an LLM without a trace",
    image: "/obscuri.webp",
    type: "image",
    link: "https://obscuri.app",
  },
  project4: {
    title: "Bcon",
    description: "A C based CLI tool that can convert data between binary value, hexadecimal, and decimal value.",
    image: "/bcon.png",
    video: "/bcon.mp4",
    type: "popup",
    github: "https://github.com/maxxfuu/bcon",
  },
}

import type { SVGProps } from "react";
import { motion } from "framer-motion";

const GitHub = (props: SVGProps<SVGSVGElement>) => (
  <svg {...props} viewBox="0 0 1024 1024" fill="currentColor">
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M8 0C3.58 0 0 3.58 0 8C0 11.54 2.29 14.53 5.47 15.59C5.87 15.66 6.02 15.42 6.02 15.21C6.02 15.02 6.01 14.39 6.01 13.72C4 14.09 3.48 13.23 3.32 12.78C3.23 12.55 2.84 11.84 2.5 11.65C2.22 11.5 1.82 11.13 2.49 11.12C3.12 11.11 3.57 11.7 3.72 11.94C4.44 13.15 5.59 12.81 6.05 12.6C6.12 12.08 6.33 11.73 6.56 11.53C4.78 11.33 2.92 10.64 2.92 7.58C2.92 6.71 3.23 5.99 3.74 5.43C3.66 5.23 3.38 4.41 3.82 3.31C3.82 3.31 4.49 3.1 6.02 4.13C6.66 3.95 7.34 3.86 8.02 3.86C8.7 3.86 9.38 3.95 10.02 4.13C11.55 3.09 12.22 3.31 12.22 3.31C12.66 4.41 12.38 5.23 12.3 5.43C12.81 5.99 13.12 6.7 13.12 7.58C13.12 10.65 11.25 11.33 9.47 11.53C9.76 11.78 10.01 12.26 10.01 13.01C10.01 14.08 10 14.94 10 15.21C10 15.42 10.15 15.67 10.55 15.59C13.71 14.53 16 11.53 16 8C16 3.58 12.42 0 8 0Z"
      transform="scale(64)"
      fill="currentColor"
      color="currentColor"
    />
  </svg>
);

export { GitHub };


export function Projects() {
  const [selectedProject, setSelectedProject] = useState<string | null>(null);

  const handleProjectClick = (key: string, project: any) => {
    if (project.type === "popup") {
      setSelectedProject(key);
    } else {
      window.open(project.link, "_blank");
    }
  };

  const currentProject = selectedProject ? projects[selectedProject as keyof typeof projects] : null;

  return (
    <>
      <div className="my-14 grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(projects).map(([key, value]) => (
          <Card
            key={key}
            role="button"
            tabIndex={0}
            className="group relative aspect-[16/9] w-full overflow-hidden rounded-md border-0 bg-white dark:bg-black p-0 cursor-pointer shadow-[0px_0px_0px_1px_rgba(9,9,11,0.08),0px_1px_2px_-1px_rgba(9,9,11,0.08),0px_2px_4px_0px_rgba(9,9,11,0.04)] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            onClick={() => handleProjectClick(key, value)}
            onKeyDown={(e) => e.key === "Enter" && handleProjectClick(key, value)}
          >
            <Image
              src={value.image}
              alt={value.title}
              width={800}
              height={400}
              className="object-cover transition-all duration-180 ease-in-out scale-100 blur-0 group-hover:scale-102 group-hover:blur-[2px]"
            />
            {/* Description card: inset with margins, slides up on hover/focus */}
            <div
              className="absolute bottom-2.5 left-2.5 right-2.5 top-2.5 rounded-md bg-white/95 dark:bg-black/95 backdrop-blur-sm px-4 py-3.5 text-[13px] shadow-[0px_0px_0px_1px_rgba(0,0,0,0.08),0px_1px_2px_-1px_rgba(0,0,0,0.08),0px_2px_4px_0px_rgba(0,0,0,0.04)] transition-transform duration-300 ease-out translate-y-[calc(100%+1.5rem)] group-hover:translate-y-0 group-focus-visible:translate-y-0 pointer-events-none"
            >
              <div className="flex flex-row justify-between items-center">
                <h3 className="font-semibold text-gray-900 dark:text-white">{value.title}</h3>
                <Link href={value.link || "#"} target="_blank">
                  <SquareArrowOutUpRight className="w-4 h-4" />
                </Link>
              </div>
              <p className="mt-1 leading-snug text-gray-600 dark:text-gray-200">{value.description}</p>
            </div>
          </Card>
        ))}
      </div>
      <Dialog open={!!selectedProject} onOpenChange={(open) => !open && setSelectedProject(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{currentProject?.title}</DialogTitle>
            <DialogDescription>{currentProject?.description}</DialogDescription>
          </DialogHeader>
          <div className="aspect-video w-full bg-black overflow-hidden relative">
            <video
              autoPlay
              controls
              playsInline
            >
              <source src={currentProject?.video} type="video/mp4" />
            </video>
          </div>
          <DialogFooter className="sm:justify-start">
          <div className="flex w-full gap-2">
            <motion.div
              className="flex-[2]"  
              whileTap={{ scale: 0.98 }}
              transition={{ duration: 0.2 }}
            >
              <Button variant="default" className="w-full cursor-pointer" asChild>
                <Link href={currentProject?.github || "#"} target="_blank">
                  <GitHub />
                  GitHub
                </Link>
              </Button>
            </motion.div>

            <motion.div
              className="flex-1"
              whileTap={{ scale: 0.98 }}
              transition={{ duration: 0.2 }}
            >
              <Button
                variant="secondary"
                className="w-full cursor-pointer hover:bg-gray-200"
                onClick={() => setSelectedProject(null)}
              >
                Cancel
              </Button>
            </motion.div>
          </div>

          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}