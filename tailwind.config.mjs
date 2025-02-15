import defaultTheme from "tailwindcss/defaultTheme";

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
        serif: ["Lora", ...defaultTheme.fontFamily.serif],
      },
      typography: {
        DEFAULT: {
          css: {
            'ol > li': {
              '&::marker': {
                content: 'counter(list-item) "."',
              },
            },
            'ol > li > ol': {
              listStyleType: 'lower-alpha',
              '& > li::marker': {
                content: 'counter(list-item, lower-alpha) "."',
              },
            },
            'ol > li > ol > li > ol': {
              listStyleType: 'lower-roman',
              '& > li::marker': {
                content: 'counter(list-item, lower-roman) "."',
              },
            },
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
