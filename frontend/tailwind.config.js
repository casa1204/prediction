/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#1a1a2e',
        secondary: '#16213e',
        accent: '#0f3460',
        highlight: '#e94560',
        success: '#00c853',
        warning: '#ffd600',
        danger: '#ff1744',
      },
    },
  },
  plugins: [],
};
