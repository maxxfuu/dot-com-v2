import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/essays/:year(\\d{4})/:slug",
        destination: "/essays/:slug",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
