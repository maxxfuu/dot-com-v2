export interface Resource {
  title: string;
  author?: string;
  note?: string;
  link?: string;
}

export interface ResourceGroup {
  category: string;
  description?: string;
  items: Resource[];
}

// Books, papers, and courses worth reading. Add entries here — the page at
// /resources renders every group in order.
export const resources: ResourceGroup[] = [];
