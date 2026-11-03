/** Tailwind CSS v3 build for the dashboard. `make css` writes app/static/css/app.css;
 *  run it after changing classes in the templates (`make check` fails if app.css is stale). */
module.exports = {
  content: ['./app/templates/**/*.html'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        darkBg: '#090d16',
        darkCard: '#111726',
        darkBorder: '#1e293b',
        bidGreen: '#10b981',
        bidBg: '#064e3b',
        askRed: '#f43f5e',
        askBg: '#881337',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
