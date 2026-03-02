/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        scan: {
          "0%, 100%": { top: "0%" },
          "50%": { top: "100%" },
        },
      },
      animation: {
        scan: "scan 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
