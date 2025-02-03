import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: "https://YOUR_GITHUB_USERNAME.github.io",
  base: "/portfolio-website",
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  }
});
