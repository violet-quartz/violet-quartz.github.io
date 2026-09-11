// How the post list groups articles. A post lands in the first category whose
// `keywords` overlap its frontmatter `tags` (exact, case-insensitive match), so
// order matters: put the more specific category first. Posts matching nothing
// are shown together under `uncategorized`.

export type Category = {
	name: string;
	keywords: string[];
};

export const categories: Category[] = [
	{ name: 'vLLM', keywords: ['vllm'] },
	{ name: 'GPU 编程', keywords: ['cuda', 'triton'] },
	{ name: '大语言模型', keywords: ['llm'] },
	{ name: '并发编程', keywords: ['concurrency'] },
	{ name: 'MLOps', keywords: ['mlops'] },
];

export const uncategorized = '其他';

export function categoryOf(tags: string[]): string {
	const lower = tags.map((t) => t.toLowerCase());
	return categories.find((c) => c.keywords.some((k) => lower.includes(k.toLowerCase())))?.name ?? uncategorized;
}
