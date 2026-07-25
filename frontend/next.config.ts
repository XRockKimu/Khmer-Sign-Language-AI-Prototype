import type { NextConfig } from "next";

const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";
// backend_keras3 (Milestone 7): a separate service on its own port, reusing
// the same same-origin-rewrite trick as BACKEND_ORIGIN above so no CORS
// configuration is needed there either.
const LSTM_BACKEND_ORIGIN = process.env.LSTM_BACKEND_ORIGIN ?? "http://127.0.0.1:8001";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${BACKEND_ORIGIN}/:path*`,
      },
      {
        source: "/api/backend-lstm/:path*",
        destination: `${LSTM_BACKEND_ORIGIN}/:path*`,
      },
    ];
  },
};

export default nextConfig;
