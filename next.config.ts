import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/essays/:year(\\d{4})/:slug",
        destination: "/writings/:slug",
        permanent: true,
      },
      {
        source: "/essays",
        destination: "/writings",
        permanent: true,
      },
      {
        source: "/essays/:path*",
        destination: "/writings/:path*",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
