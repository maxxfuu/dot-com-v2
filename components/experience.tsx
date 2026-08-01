"use client";

import { Timeline, TimelineConnector, TimelineDot, TimelineContent, TimelineDescription, TimelineItem, TimelineTime, TimelineTitle } from "./ui/timeline";
import Link from "next/link";
import { motion } from "framer-motion";
import type { CSSProperties, ReactNode } from "react";

const MotionTimelineItem = motion.create(TimelineItem);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

function getDotStyle(index: number, total: number): CSSProperties {
  const opacityPercent = total <= 1 ? 100 : 100 - (index / (total - 1)) * 90;

  return {
    backgroundColor: `color-mix(in oklab, var(--foreground) ${opacityPercent}%, transparent)`,
  };
}

type ExperienceEntry = {
  time: string;
  title: ReactNode;
  description?: string;
  titleClassName?: string;
};

const experiences: ExperienceEntry[] = [
  {
    time: "May 2026 - Present",
    title: <>Kernel Engineering at <span className="text-foreground">7labs</span></>,
    description: "Working on GPU kernels and runtime inference for large language models at 7labs.",
  },
  {
    time: "May 2025 - Aug 2025",
    title: (
      <>
        <span className="text-foreground">Software Engineer Intern</span> at{" "}
        <Link href="https://machyna.com/" target="_blank">
          <span className="border-b border-dashed border-foreground text-foreground transition-colors hover:border-foreground hover:text-foreground">Machyna</span>
        </Link>
      </>
    ),
    description: "Built a serverless file management system using AWS Lambda, AppSync, and S3.",
  },
  {
    time: "Aug 2024 - Dec 2024",
    title: "Deep Learning Researcher",
    titleClassName: "text-foreground",
    description: "Researched mechanistic interpretability of LLMs to mitigate sycophantic behavior.",
  },
  {
    time: "Aug 2024 - Dec 2024",
    title: "Undergraduate Teaching Assistant",
    titleClassName: "text-foreground",
    description: "Facilitated lectures for 150+ students in electrical engineering programming.",
  },
  {
    time: "Oct 2023 - Oct 2024",
    title: "Undergraduate Research Assistant",
    titleClassName: "text-foreground",
    description: "Developed an LLM-based autograder, improving scoring accuracy and efficiency.",
  },
  {
    time: "Dec 2023 - Aug 2024",
    title: "SIG AI Lead @ ACM, UC Merced",
    titleClassName: "text-foreground",
    description: "Led AI workshops on deep learning and LLMs for 50+ active ACM members.",
  },
  {
    time: "Jun 2022 - Jun 2023",
    title: "Powerlifting Coaching Business",
    titleClassName: "text-foreground",
    description: "Coached 6 collegiate athletes to nationals across CTPA, HKPF, and BPU.",
  },
  {
    time: "Dec 2020 - Dec 2020",
    title: "All Young International Deadlift Open",
    titleClassName: "text-foreground",
    description: "Placed in the top 25% in the deadlift, pulling 210 kg in the Open Class as a 74 kg sub-junior.",
  },
  {
    time: "Dec 2019 - Dec 2019",
    title: "HKPF Regionals",
    titleClassName: "text-foreground",
    description: "Placed 3rd overall at HKPF Regionals in the Open Class as a sub-junior.",
  },
];

export function Experience() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="">
        <Timeline>
          {experiences.map((entry, index) => (
              <MotionTimelineItem key={`${entry.time}-${index}`} variants={itemVariants}>
                <TimelineConnector>
                  <TimelineDot
                    className="bg-transparent"
                    style={getDotStyle(index, experiences.length)}
                    active={index === 0}
                  />
                </TimelineConnector>
                <TimelineContent>
                  <TimelineTime>{entry.time}</TimelineTime>
                  <TimelineTitle className={entry.titleClassName}>{entry.title}</TimelineTitle>
                  {entry.description && (
                    <TimelineDescription>{entry.description}</TimelineDescription>
                  )}
                </TimelineContent>
              </MotionTimelineItem>
          ))}
        </Timeline>
      </div>
    </motion.div>
  );
}
