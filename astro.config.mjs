// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
	site: 'https://violet-quartz.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		// $inline$ and $$block$$ math, rendered to HTML at build time by KaTeX.
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
		shikiConfig: {
			// Light so code blocks sit inside the page instead of punching holes in it.
			theme: 'github-light',
			wrap: false,
		},
	},
	// Downloaded and self-hosted at build time — no runtime request to Google.
	// CJK families sit before the generic fallback so Chinese glyphs land on a
	// real sans instead of whatever `sans-serif` happens to be.
	fonts: [
		{
			provider: fontProviders.google(),
			name: 'Inter',
			cssVariable: '--font-sans',
			weights: [400, 500, 600, 700],
			subsets: ['latin', 'latin-ext'],
			fallbacks: ['PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'system-ui', 'sans-serif'],
		},
		{
			provider: fontProviders.google(),
			name: 'JetBrains Mono',
			cssVariable: '--font-mono',
			weights: [400],
			subsets: ['latin'],
			fallbacks: ['ui-monospace', 'SFMono-Regular', 'monospace'],
		},
	],
});
