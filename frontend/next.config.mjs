/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Suppress Socket.IO 404 errors in development
  async rewrites() {
    return [
      {
        source: '/socket.io/:path*',
        destination: '/404',
      },
    ]
  },
}

export default nextConfig
