"use client";

import { Timeline, TimelineConnector, TimelineDot, TimelineContent, TimelineDescription, TimelineItem, TimelineTime, TimelineTitle } from "./ui/timeline";
import Link from "next/link";
import { motion } from "framer-motion";

const MotionTimelineItem = motion(TimelineItem);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15, // increased stagger for more distinct 1-by-1 feel
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export function Experience() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="max-w-xl mx-auto tracking-tight">
        <Timeline>
          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black" active />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>Sep 2025 - Present</TimelineTime>
              <TimelineTitle>
                Co-founder to a <span className="font-semibold">Productivity Platform</span>
              </TimelineTitle>
              <TimelineDescription>
                Bootstrapped a SaaS company and scaling it to 10k MRR
              </TimelineDescription>
            </TimelineContent>
          </MotionTimelineItem>

          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black/80" />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>May 2025 - Aug 2025</TimelineTime>
              <TimelineTitle>
                <span className="font-semibold">Software Engineer Intern</span> at{" "}
                <Link href="https://machyna.com/" target="_blank">
                  <span className="text-muted-foreground border-dashed border-b border-muted-foreground">Machyna</span>
                </Link>
              </TimelineTitle>
              <TimelineDescription>
                Built a serverless file management system using AWS Lambda, AppSync, and S3
              </TimelineDescription>
            </TimelineContent>
          </MotionTimelineItem>

          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black/60" />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>Aug 2024 - Dec 2024</TimelineTime>
              <TimelineTitle className="font-semibold">
                Deep Learning Researcher
              </TimelineTitle>
              <TimelineDescription>
                Research on mechanistic interpretability of LLM, mitigating sycophantic behaviors
              </TimelineDescription>
            </TimelineContent>
          </MotionTimelineItem>

          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black/40" />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>Aug 2024 - Dec 2024</TimelineTime>
              <TimelineTitle className="font-semibold">
                Undergraduate Teaching Assistant
              </TimelineTitle>
              <TimelineDescription>
                Facilitated lectures for 150+ students in Electrical Engineering Programming
              </TimelineDescription>
            </TimelineContent>
          </MotionTimelineItem>

          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black/20" />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>Oct 2023 - Oct 2024</TimelineTime>
              <TimelineTitle className="font-semibold">
                Undergraduate Research Assistant
              </TimelineTitle>
              <TimelineDescription>
                Developed an LLM-based autograder, improving scoring accuracy and efficiency.
              </TimelineDescription>
            </TimelineContent>
          </MotionTimelineItem>

          <MotionTimelineItem variants={itemVariants}>
            <TimelineConnector>
              <TimelineDot className="bg-black/10" />
            </TimelineConnector>
            <TimelineContent>
              <TimelineTime>June 2023 - Aug 2023</TimelineTime>
              <TimelineTitle className="font-semibold">
                Moved from Taichung to California
              </TimelineTitle>
            </TimelineContent>
          </MotionTimelineItem>
        </Timeline>
      </div>
    </motion.div>
  )
}