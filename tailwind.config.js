/** Tailwind CSS v3 build for the dashboard. `make css` writes app/static/css/themes.css (the color
 *  themes, from scripts/build_themes.py) and app/static/css/app.css; run it after changing classes in
 *  the templates (`make check` fails if either is stale).
 *
 *  Every shade of the color families the templates use, and the dashboard's own surface and buy/sell
 *  colors, is a CSS variable (--c-<family>-<shade>, an "R G B" triplet so opacity modifiers work), so
 *  a theme is one attribute on <html>: data-theme="midnight|daylight|terminal", and
 *  data-palette="colorblind" for blue/orange buy/sell colors. */
const FAMILIES = ['slate', 'emerald', 'rose', 'cyan', 'amber', 'violet', 'sky', 'lime'];
const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const variable = (name) => `rgb(var(--c-${name}) / <alpha-value>)`;
const themed = Object.fromEntries(FAMILIES.map((family) =>
  [family, Object.fromEntries(SHADES.map((shade) => [shade, variable(`${family}-${shade}`)]))]));

module.exports = {
  content: ['./app/templates/**/*.html', './app/static/js/lob-*.js'],  // not the vendored htmx and charts
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ...themed,
        darkBg: variable('darkBg'),
        darkCard: variable('darkCard'),
        darkBorder: variable('darkBorder'),
        bidGreen: variable('bidGreen'),
        bidBg: variable('bidBg'),
        askRed: variable('askRed'),
        askBg: variable('askBg'),
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['var(--font-sans)'],
      },
    },
  },
  plugins: [],
};
