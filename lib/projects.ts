export interface Project {
  title: string;
  description: string;
  image: string;
  type?: "image" | "popup";
  github?: string;
  video?: string;
  link?: string;
}

export const projects: Project[] = [
  {
    title: "VisualGPU",
    description:
      "An interactive, memory-first curriculum for GPU programming and inference performance, from bytes and caches to kernels and tokens per second",
    image: "/visualgpu.png",
    type: "image",
    link: "https://visualgpu.com",
  },
  {
    title: "VisualKernels",
    description:
      "Turns your C/CUDA kernels into a live, steppable 3D view of your buffers — watch every thread, loop, and buffer update as it happens",
    image: "/visualkernels.png",
    type: "image",
    link: "https://visualkernels.com",
  },
  {
    title: "Bookie",
    description:
      "A local-first ledger that tracks every eval fee, reset, refund, and payout across prop trading firms — cost per funded account, recovery ratio, and your exact breakeven point",
    image: "/bookie.png",
    type: "image",
    link: "https://trybookie.app",
  },
  {
    title: "Yagmi",
    description:
      "Create AI Mentors that lets you chat with an “advisor” tailored to your goals, giving you guidance and next steps as you grow",
    image: "/yagmi.png",
    type: "image",
    link: "https://yagmi.app",
  },
  {
    title: "Obscuri",
    description:
      "A desktop application that silently awaits for your screenshots and querys an LLM without a trace",
    image: "/obscuri.webp",
    type: "image",
    link: "https://obscuri.app",
  },
  {
    title: "ClockIn",
    description:
      "An app that helps you stay accountable by recording timelapses of you working, so you can “clock in” with social proof",
    image: "/clockin.png",
    type: "image",
    link: "https://clockin.now",
  },
  {
    title: "Bcon",
    description:
      "A C based CLI tool that can convert data between binary value, hexadecimal, and decimal value.",
    image: "/bcon.png",
    video: "/bcon.mp4",
    type: "popup",
    github: "https://github.com/maxxfuu/bcon",
  },
];
