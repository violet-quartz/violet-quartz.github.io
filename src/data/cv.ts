// Résumé content. Edit this file — cv.astro renders it and never needs changing.
// Anything marked TODO is placeholder text.

export const profile = {
	name: 'Flora Ma',
	// One line under your name. Say what you do, not what you want.
	tagline: 'LLM inference & GPU systems',
	// Public page: keep to channels you don't mind being scraped.
	// Phone/address belong in the PDF you send directly, not here.
	links: [
		{ label: 'violet-quartz.github.io', href: 'https://violet-quartz.github.io' },
		{ label: 'github.com/violet-quartz', href: 'https://github.com/violet-quartz' },
		{ label: 'mczggy@gmail.com', href: 'mailto:mczggy@gmail.com' },
	],
};

// Listed before experience on purpose: for infra roles, work someone can read
// beats a job title. Link each one to its repo and to your write-up.
export const projects = [
	{
		name: 'nano-vllm',
		blurb: 'TODO — one sentence on what it is and what was hard about it.',
		// Quantify where you can: throughput, latency, memory, model size.
		highlights: [
			'TODO — e.g. "Cut prefill latency 2.3x by batching prompts across requests."',
		],
		href: 'https://github.com/violet-quartz/nano-vllm',
		// Optional: link the matching blog post to show depth.
		writeup: null as string | null,
	},
	{
		name: 'mini-sglang',
		blurb: 'TODO',
		highlights: ['TODO'],
		href: 'https://github.com/violet-quartz/mini-sglang',
		writeup: null as string | null,
	},
	{
		name: 'llm-inference-energy-benchmark',
		blurb: 'TODO',
		highlights: ['TODO'],
		href: 'https://github.com/violet-quartz/llm-inference-energy-benchmark',
		writeup: null as string | null,
	},
];

export const experience = [
	{
		role: 'TODO — Job Title',
		org: 'TODO — Company',
		location: 'TODO',
		start: 'TODO',
		end: 'Present',
		// Each bullet: what you did -> what changed as a result.
		highlights: [
			'TODO — lead with the verb and end with the number.',
			'TODO',
		],
	},
];

export const skills = [
	{ group: 'Systems', items: ['TODO — CUDA, C++, ...'] },
	{ group: 'ML', items: ['TODO — PyTorch, ...'] },
	{ group: 'Tooling', items: ['TODO — Docker, ...'] },
];

export const education = [
	{
		degree: 'TODO — Degree',
		org: 'TODO — Institution',
		start: 'TODO',
		end: 'TODO',
		note: null as string | null,
	},
];
