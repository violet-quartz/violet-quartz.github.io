import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
	// Load Markdown and MDX files in the `src/content/blog/` directory.
	loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
	// Type-check frontmatter using a schema
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			description: z.string(),
			// Transform string to Date object
			pubDate: z.coerce.date(),
			updatedDate: z.coerce.date().optional(),
			heroImage: z.optional(image()),
			// Free-form topic tags, e.g. ['cuda', 'inference'].
			tags: z.array(z.string()).default([]),
			// Pin to the top of the post list.
			featured: z.boolean().default(false),
			// Hide from listings and RSS without deleting the file.
			draft: z.boolean().default(false),
		}),
});

export const collections = { blog };
