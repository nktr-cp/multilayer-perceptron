/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    externalDir: true,
  },
  output: 'export',
  trailingSlash: true,
  basePath: process.env.NODE_ENV === 'production' ? '/multilayer-perceptron' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/multilayer-perceptron/' : '',
}

export default nextConfig
