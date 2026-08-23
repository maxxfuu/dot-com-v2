export interface ResourceItem {
  title: string;
  author?: string;
  note?: string;
  link?: string;
}

export interface ResourceGroup {
  category: string;
  description?: string;
  items: ResourceItem[];
}

export const resources: ResourceGroup[] = [
  // {
  //   category: "gpu programming",
  //   description: "Where I'd start if I were learning this again.",
  //   items: [
  //     {
  //       title: "Programming Massively Parallel Processors",
  //       author: "Hwu, Kirk, El Hajj",
  //       note: "The book that made the memory hierarchy click for me.",
  //       link: "https://example.com",
  //     },
  //   ],
  // },
];
