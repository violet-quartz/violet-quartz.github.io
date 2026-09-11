// Résumé content for the public /cv page. Edit this file — cv.astro renders it
// and never needs changing.
//
// This is the *public* version: no phone number, no full name, and only the
// headline facts. The complete CV lives in cv/cv_private.md (gitignored) and
// is turned into a PDF with `npm run cv:pdf`.

export const profile = {
	name: 'Flora',
	// One line under your name. Say what you do, not what you want.
	tagline: 'AI Infra 工程师 · LLM 推理服务与平台',
	// Public page: keep to channels you don't mind being scraped.
	// Phone/address belong in the PDF you send directly, not here.
	// One per line: `name` is the plain-text prefix, `label` is the clickable part.
	links: [
		{ name: '个人主页', label: 'violet-quartz.github.io', href: 'https://violet-quartz.github.io' },
		{ name: 'GitHub', label: 'github.com/violet-quartz', href: 'https://github.com/violet-quartz' },
		{ name: '邮箱', label: 'myf.py@163.com', href: 'mailto:myf.py@163.com' },
	],
};

// Listed before experience on purpose: for infra roles, work someone can read
// beats a job title. Link each one to its repo and to your write-up.
// The section is hidden while this is empty.
type Project = {
	name: string;
	blurb: string;
	// Quantify where you can: throughput, latency, memory, model size.
	highlights: string[];
	href: string;
	// Optional: link the matching blog post to show depth.
	writeup: string | null;
};
export const projects: Project[] = [];

export const experience = [
	{
		role: '高级软件工程师',
		org: '阿里云',
		location: '北京',
		start: '2024.01',
		end: '2026.07',
		// Each bullet: what you did -> what changed as a result.
		highlights: [] as string[],
	},
	{
		role: '软件工程师',
		org: '微软',
		location: '中国',
		start: '2015.07',
		end: '2023.03',
		highlights: [] as string[],
	},
];

export const skills = [
	{ group: '语言', items: ['Python', 'Go', 'C++', 'C#'] },
	{ group: '推理与部署', items: ['vLLM', 'SGLang', 'XGrammar'] },
	{ group: '训练平台', items: ['ms-swift', 'PyTorch'] },
	{ group: 'GPU', items: ['Triton', 'CUDA', '显存建模'] },
	{ group: '性能分析', items: ['py-spy', 'memray', 'gperftools', '火焰图', '全链路分段打点'] },
	{ group: '服务与存储', items: ['gRPC', 'FastAPI', 'Cosmos DB'] },
];

export const education = [
	{
		degree: '硕士 · 管理信息系统',
		org: '清华大学',
		start: '2012',
		end: '2015',
		note: '2014 年暑期于 Google Tokyo 任软件工程师实习生' as string | null,
	},
	{
		degree: '学士 · 计算机科学与技术',
		org: '北京大学',
		start: '2008',
		end: '2012',
		note: '经济学双学位' as string | null,
	},
];
