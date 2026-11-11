/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sage: {
          50: '#f4f7f4',
          100: '#e5ede6',
          200: '#ccdccd',
          300: '#a6c3a8',
          400: '#7ca480',
          500: '#5c8861',
          600: '#486d4c',
          700: '#3a573e',
          800: '#2f4632',
          900: '#273a2a',
        },
        surface: {
          50: '#fbfcf9',
          100: '#f5f7f2',
          200: '#ebefe6',
          300: '#dde4d5',
          border: '#e1e7dc',
        },
      },
    },
  },
  plugins: [],
}
