#!/usr/bin/env node
// Turn a CV written in Markdown into a PDF: markdown → HTML (styled by
// scripts/cv-pdf.css) → the locally installed Chrome, printing headless.
//
//   npm run cv:pdf                          # cv/cv_private.md → cv/cv_private.pdf
//   npm run cv:pdf -- cv/cv_public.md       # any file; PDF lands next to it
//   npm run cv:pdf -- in.md out.pdf --html  # also keep out.html to tweak the CSS in a browser
//
// The Markdown follows a few conventions the plugin below turns into layout:
//   # Name                       → masthead; the lines right under it become the
//                                  tagline and the contact line
//   ### Company · Role　2024.01 – 2026.07
//   **School** · Degree　2012 – 2015
//                                → a trailing date range is pulled out and
//                                  right-aligned
//   **Team name**                → a bold-only line is a sub-heading
// Hard-wrapped lines are joined; a wrap between two CJK characters leaves no
// stray space behind.

import { spawn } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createMarkdownProcessor } from '@astrojs/markdown-remark';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

// ---------- CLI ----------

const args = process.argv.slice(2);
const keepHtml = args.includes('--html');
const positional = args.filter((a) => !a.startsWith('--'));
const input = path.resolve(root, positional[0] ?? 'cv/cv_private.md');
const output = path.resolve(root, positional[1] ?? input.replace(/\.md$/i, '') + '.pdf');

if (!existsSync(input)) {
	console.error(`cv-pdf: input not found: ${path.relative(root, input)}`);
	process.exit(1);
}

// ---------- Markdown → HTML ----------

const CJK = '[\\u3000-\\u303f\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff\\uff00-\\uffef]';
const CJK_WRAP = new RegExp(`(${CJK})\\n(${CJK})`, 'g');
// "2024.01 – 2026.07", "2012 – 2015", "2021 – 至今", with any dash flavour.
const DATE_RANGE = /(?:^|\s|　)((?:\d{4}(?:\.\d{1,2})?)\s*[–—-]\s*(?:\d{4}(?:\.\d{1,2})?|至今|Present|present))\s*$/;

const isText = (node) => node?.type === 'text';

// Wraps children in an element that mdast-util-to-hast will emit as <span class>.
const span = (className, children) => ({
	type: 'cvSpan',
	data: { hName: 'span', hProperties: { className: [className] } },
	children,
});

const addClass = (node, className) => {
	node.data ??= {};
	node.data.hProperties ??= {};
	node.data.hProperties.className = [...(node.data.hProperties.className ?? []), className];
};

// Split a paragraph's inline children into lines at "\n" inside text nodes,
// but only where `shouldSplit(textBefore, nextSibling)` agrees. Returns an
// array of child arrays, one per resulting paragraph.
function splitLines(children, shouldSplit) {
	const lines = [[]];
	children.forEach((child, i) => {
		if (!isText(child) || !child.value.includes('\n')) {
			lines.at(-1).push(child);
			return;
		}
		const parts = child.value.split('\n');
		parts.forEach((part, j) => {
			if (part) lines.at(-1).push({ type: 'text', value: part });
			if (j === parts.length - 1) return;
			// What follows this break decides whether it splits: the next text
			// part, or — when the text ends right at the break — the next sibling.
			const rest = parts[j + 1];
			const next = rest === '' && j + 1 === parts.length - 1 ? children[i + 1] : { type: 'text', value: rest };
			if (shouldSplit(part, next)) lines.push([]);
			else lines.at(-1).push({ type: 'text', value: '\n' });
		});
	});
	return lines.filter((line) => line.length > 0);
}

// Merge adjacent text nodes so later regexes see whole lines.
function mergeText(children) {
	return children.reduce((acc, child) => {
		const prev = acc.at(-1);
		if (isText(child) && isText(prev)) prev.value += child.value;
		else acc.push(child);
		return acc;
	}, []);
}

function pullDates(node) {
	const last = node.children.at(-1);
	if (!isText(last)) return;
	const m = last.value.match(DATE_RANGE);
	if (!m) return;
	last.value = last.value.slice(0, m.index).replace(/[\s　]+$/, '');
	const title = node.children.filter((c) => !(isText(c) && c.value === ''));
	node.children = [span('title', title), span('dates', [{ type: 'text', value: m[1] }])];
	addClass(node, 'dated');
}

function remarkCv() {
	return (tree) => {
		const out = [];
		let afterName = false;

		for (const node of tree.children) {
			if (node.type === 'heading' && node.depth === 1) {
				afterName = true;
				out.push(node);
				continue;
			}

			if (node.type === 'paragraph') {
				node.children = mergeText(node.children);
				// Masthead: every line under the name stands on its own.
				// Elsewhere: a new line that opens with **bold** starts a new entry
				// (education rows, team names); anything else is a hard wrap.
				const lines = afterName
					? splitLines(node.children, () => true)
					: splitLines(node.children, (_, next) => next?.type === 'strong');
				lines.forEach((children, i) => {
					const p = { type: 'paragraph', children: mergeText(children) };
					if (afterName) addClass(p, i === 0 ? 'tagline' : 'contact');
					out.push(p);
				});
				afterName = false;
				continue;
			}

			afterName = false;
			out.push(node);
		}
		tree.children = out;

		// Second pass over everything, including list items: join wrapped
		// lines and right-align trailing date ranges.
		const walk = (node) => {
			if (isText(node)) node.value = node.value.replace(CJK_WRAP, '$1$2');
			if (node.type === 'heading' && node.depth === 3) pullDates(node);
			if (node.type === 'paragraph' && !node.data?.hProperties?.className) pullDates(node);
			node.children?.forEach(walk);
		};
		walk(tree);
	};
}

const processor = await createMarkdownProcessor({
	remarkPlugins: [remarkCv],
	syntaxHighlight: false,
	smartypants: false,
});
const markdown = await readFile(input, 'utf8');
const { code: body } = await processor.render(markdown);
const css = await readFile(path.join(here, 'cv-pdf.css'), 'utf8');
const title = markdown.match(/^#\s+(.+)$/m)?.[1] ?? path.basename(input, '.md');

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>
${css}
</style>
</head>
<body>
<main class="cv">
${body}
</main>
</body>
</html>
`;

// ---------- HTML → PDF ----------

function findChrome() {
	const candidates = [
		process.env.CHROME_PATH,
		'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		'/Applications/Chromium.app/Contents/MacOS/Chromium',
		'/usr/bin/google-chrome',
		'/usr/bin/chromium',
		'/usr/bin/chromium-browser',
	];
	const found = candidates.find((c) => c && existsSync(c));
	if (!found) {
		console.error('cv-pdf: Chrome not found. Install Google Chrome or set CHROME_PATH.');
		process.exit(1);
	}
	return found;
}

// Chrome's new headless mode (152 at the time of writing) writes the PDF and
// then just sits there instead of exiting, so rather than waiting on the
// process we wait on the file: once it exists and stops growing, we're done.
function printToPdf(chrome, htmlPath, pdfPath, profileDir) {
	return new Promise((resolve, reject) => {
		const proc = spawn(
			chrome,
			[
				'--headless=new',
				'--disable-gpu',
				'--no-first-run',
				'--no-default-browser-check',
				'--disable-background-networking',
				'--disable-component-update',
				'--disable-sync',
				'--disable-extensions',
				'--no-pdf-header-footer',
				// A throwaway profile so this never touches the running Chrome.
				`--user-data-dir=${profileDir}`,
				`--print-to-pdf=${pdfPath}`,
				pathToFileURL(htmlPath).href,
			],
			{ stdio: 'ignore' },
		);

		const started = Date.now();
		let lastSize = -1;
		const poll = setInterval(() => {
			const size = existsSync(pdfPath) ? statSync(pdfPath).size : -1;
			if (size > 0 && size === lastSize) return finish();
			lastSize = size;
			if (Date.now() - started > 60_000) return finish(new Error('cv-pdf: Chrome did not produce a PDF within 60s'));
		}, 250);

		let done = false;
		let exited = false;
		function finish(err) {
			if (done) return;
			done = true;
			clearInterval(poll);
			const settle = () => (err ? reject(err) : resolve());
			if (exited) return settle();
			// Let Chrome shut down (it's still flushing its profile) before the
			// temp dir goes away; escalate if it drags its feet.
			proc.once('exit', settle);
			proc.kill();
			setTimeout(() => proc.kill('SIGKILL'), 5_000).unref();
		}
		proc.on('error', finish);
		proc.on('exit', () => {
			exited = true;
			finish(existsSync(pdfPath) ? undefined : new Error('cv-pdf: Chrome exited without writing a PDF'));
		});
	});
}

const tmp = await mkdtemp(path.join(os.tmpdir(), 'cv-pdf-'));
try {
	const htmlPath = path.join(tmp, 'cv.html');
	await writeFile(htmlPath, html);
	if (keepHtml) await writeFile(output.replace(/\.pdf$/i, '') + '.html', html);
	await rm(output, { force: true });
	await printToPdf(findChrome(), htmlPath, output, path.join(tmp, 'profile'));
	console.log(`cv-pdf: wrote ${path.relative(root, output)}`);
} finally {
	await rm(tmp, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
