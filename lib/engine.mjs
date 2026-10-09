// Cloud QMD engine — makes `qmd` use cloud models (SiliconFlow / any OpenAI-compatible
// embeddings + rerank + chat endpoint) instead of local GGUF models via node-llama-cpp.
//
// Invoked by the `qmd` shim at ~/.pi/agent/bin/qmd, which the pi extension
// `extensions/cloud-qmd/index.ts` prepends to PATH.
//
//   vsearch <query>   -> cloud embedding + cosine over the local vector index
//   query   <query>   -> query expansion + cloud vector + BM25 (RRF) + cloud rerank
//   embed             -> build/refresh the cloud vector index
//   update            -> delegate to real qmd (FTS), then refresh the vector index
//   anything else     -> delegated verbatim to the real qmd binary
//
// Embeddings and rerank always go to the cloud, so no embedding/rerank GGUF is
// ever loaded. Query expansion defaults to qmd's own local query-expansion GGUF,
// loaded through qmd's own LlamaCpp (see `qmd __cloud pull`); it is never
// downloaded implicitly. Set search.expansion.provider to `cloud` or `off` to change that.

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

/* ------------------------------------------------------------------ config */

const AGENT_DIR = process.env.PI_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
export const CONFIG_PATH = process.env.CLOUD_QMD_CONFIG || path.join(AGENT_DIR, "cloud-qmd.json");

const DEFAULTS = {
	enabled: true,
	realQmdPath: "/opt/homebrew/lib/node_modules/@tobilu/qmd/bin/qmd",
	qmdConfigPath: "~/.config/qmd/index.yml",
	stateDir: path.join(AGENT_DIR, "cloud-qmd"),
	collections: {},
	provider: {
		baseUrl: "https://api.siliconflow.cn/v1",
		apiKey: "env:SILICONFLOW_API_KEY",
		embedModel: "BAAI/bge-m3",
		rerankModel: "BAAI/bge-reranker-v2-m3",
		chatModel: "Qwen/Qwen2.5-7B-Instruct",
		timeoutMs: 60_000,
	},
	search: {
		chunkTokens: 400,
		chunkOverlapTokens: 80,
		embedBatchSize: 24,
		maxChunksPerDocument: 400,
		candidateLimit: 40,
		inlineRefresh: true,
		inlineRefreshDeadlineMs: 10_000,
		hybrid: { rrfK: 60, bm25Weight: 1.0, vectorWeight: 1.0 },
		expansion: {
			enabled: true,
			provider: "local", // "local" (qmd's GGUF) | "cloud" (provider.chatModel) | "off"
			variants: 3,
			localModel: "hf:tobil/qmd-query-expansion-0.6B-gguf/qmd-query-expansion-0.6B-q4_k_m.gguf", // null = qmd index.yml `models.generate`
			localTimeoutMs: 25_000,
			cloudFallback: false, // use the cloud chat model if the local model fails
			hfEndpoint: "https://hf-mirror.com", // used when downloading the local model
		},
		rerank: { enabled: true },
	},
};

export function expandHome(p) {
	if (!p) return p;
	if (p === "~") return os.homedir();
	if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
	return p;
}

export function loadConfig() {
	let fileCfg = {};
	if (fs.existsSync(CONFIG_PATH)) {
		try {
			fileCfg = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
		} catch (err) {
			process.stderr.write(`cloud-qmd: cannot parse ${CONFIG_PATH}: ${err.message}\n`);
		}
	}
	const cfg = {
		...DEFAULTS,
		...fileCfg,
		provider: { ...DEFAULTS.provider, ...(fileCfg.provider ?? {}) },
		search: {
			...DEFAULTS.search,
			...(fileCfg.search ?? {}),
			hybrid: { ...DEFAULTS.search.hybrid, ...(fileCfg.search?.hybrid ?? {}) },
			expansion: { ...DEFAULTS.search.expansion, ...(fileCfg.search?.expansion ?? {}) },
			rerank: { ...DEFAULTS.search.rerank, ...(fileCfg.search?.rerank ?? {}) },
		},
		collections: { ...DEFAULTS.collections, ...(fileCfg.collections ?? {}) },
	};
	cfg.stateDir = expandHome(cfg.stateDir);
	cfg.apiKey = resolveApiKey(cfg.provider.apiKey);
	cfg.baseUrl = String(cfg.provider.baseUrl ?? "").replace(/\/+$/, "");
	return cfg;
}

/** Resolve an API key from `env:NAME`, `$NAME`, `file:PATH`, or a literal. */
export function resolveApiKey(spec) {
	if (!spec || typeof spec !== "string") return "";
	if (spec.startsWith("env:")) return (process.env[spec.slice(4)] ?? "").trim();
	if (spec.startsWith("file:")) {
		try {
			return fs.readFileSync(expandHome(spec.slice(5)), "utf8").trim();
		} catch {
			return "";
		}
	}
	if (spec.startsWith("$$")) return spec.slice(1);
	if (spec.startsWith("$")) return (process.env[spec.slice(1)] ?? "").trim();
	return spec.trim();
}

/* -------------------------------------------------------------- collections */

/**
 * Minimal reader for qmd's own `index.yml` so the shim works for any collection
 * qmd knows about, without a YAML dependency. Only the `collections:` block is read.
 */
function scanQmdYaml(cfg) {
	const result = { collections: {}, models: {} };
	const p = expandHome(cfg.qmdConfigPath);
	if (!fs.existsSync(p)) return result;
	let section = null;
	let current = null;
	for (const raw of fs.readFileSync(p, "utf8").split(/\r?\n/)) {
		if (/^\S/.test(raw)) {
			section = /^collections:/.test(raw) ? "collections" : /^models:/.test(raw) ? "models" : null;
			current = null;
			continue;
		}
		if (!section) continue;
		if (section === "models") {
			const kv = raw.match(/^\s+(embed|embedding|generate|query|rerank):\s*(.+?)\s*$/);
			if (!kv) continue;
			const value = kv[2].replace(/^["']|["']$/g, "");
			const key = kv[1] === "embedding" ? "embed" : kv[1] === "query" ? "generate" : kv[1];
			result.models[key] = value;
			continue;
		}
		const name = raw.match(/^\s{1,4}([A-Za-z0-9_.-]+):\s*$/);
		if (name) {
			current = name[1];
			result.collections[current] = result.collections[current] ?? {};
			continue;
		}
		const kv = raw.match(/^\s+(path|pattern|mask):\s*(.+?)\s*$/);
		if (kv && current) {
			const value = kv[2].replace(/^["']|["']$/g, "");
			if (kv[1] === "path") result.collections[current].root = expandHome(value);
			else result.collections[current].pattern = value;
		}
	}
	return result;
}

export function readQmdCollections(cfg) {
	return scanQmdYaml(cfg).collections;
}

/** Model URIs from qmd's own config (`models: {embed, generate, rerank}`). */
export function readQmdModels(cfg) {
	return scanQmdYaml(cfg).models;
}

/** Resolve a collection name to `{root, pattern}`, explicit config winning. */
export function resolveCollection(cfg, name) {
	const explicit = cfg.collections?.[name];
	if (explicit?.root) {
		return { root: expandHome(explicit.root), pattern: explicit.pattern ?? "**/*.md" };
	}
	const fromQmd = readQmdCollections(cfg)[name];
	if (fromQmd?.root) return { root: fromQmd.root, pattern: fromQmd.pattern ?? "**/*.md" };
	if (name === "pi-memory") {
		const root = expandHome(process.env.PI_MEMORY_DIR || path.join(AGENT_DIR, "memory"));
		return { root, pattern: "**/*.md" };
	}
	return null;
}

/* ----------------------------------------------------------------- chunking */

const CJK = /[\u1100-\u11ff\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/;

/** Rough token count: CJK codepoints ≈ 1 token each, other text ≈ 4 chars/token. */
export function estimateTokens(text) {
	let cjk = 0;
	let other = 0;
	for (const ch of text) {
		if (CJK.test(ch)) cjk++;
		else other++;
	}
	return Math.ceil(cjk + other / 4);
}

/**
 * Markdown-aware chunking: split on blank lines and heading boundaries, then pack
 * blocks up to a token budget with a trailing overlap for context continuity.
 */
export function chunkMarkdown(text, options = {}) {
	const budget = options.chunkTokens ?? 400;
	const overlap = Math.min(options.chunkOverlapTokens ?? 80, Math.floor(budget / 2));
	const lines = text.split(/\r?\n/);
	const blocks = [];
	let buf = [];
	let startLine = 1;
	let sawContent = false;

	const flush = () => {
		if (buf.length === 0) return;
		const joined = buf.join("\n").trim();
		if (joined) blocks.push({ text: joined, line: startLine });
		buf = [];
	};

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const isHeading = /^#{1,6}\s/.test(line);
		// A heading is a preferred break point, but only once the buffer is already
		// half full — otherwise every heading would flush a tiny block.
		if (isHeading && sawContent && estimateTokens(buf.join("\n")) >= budget * 0.5) {
			flush();
			startLine = i + 1;
		}
		if (!sawContent && line.trim()) {
			startLine = i + 1;
			sawContent = true;
		}
		buf.push(line);
		if (line.trim() === "") {
			// End of a paragraph: flush if the buffer is already near budget.
			if (estimateTokens(buf.join("\n")) >= budget) flush();
		}
		if (estimateTokens(buf.join("\n")) >= budget) flush();
	}
	flush();

	// Pack small blocks together, then split anything still oversized.
	const packed = [];
	for (const block of blocks) {
		const last = packed[packed.length - 1];
		if (last && estimateTokens(`${last.text}\n\n${block.text}`) <= budget) {
			last.text = `${last.text}\n\n${block.text}`;
		} else {
			packed.push({ ...block });
		}
	}

	const chunks = [];
	for (const block of packed) {
		if (estimateTokens(block.text) <= budget) {
			chunks.push(block);
			continue;
		}
		const sentences = block.text.split(/(?<=[。！？!?.;\n])/);
		let cur = "";
		let curLine = block.line;
		const pushCur = () => {
			const t = cur.trim();
			if (t) chunks.push({ text: t, line: curLine });
			const tail = t.slice(-Math.max(0, overlap * 2));
			cur = overlap > 0 ? tail : "";
		};
		for (const s of sentences) {
			if (estimateTokens(cur + s) > budget && cur) {
				pushCur();
				curLine = block.line;
			}
			cur += s;
		}
		if (cur.trim()) chunks.push({ text: cur.trim(), line: curLine });
	}

	return chunks.filter((c) => c.text.trim().length > 0);
}

/* ------------------------------------------------------------- http helpers */

class CloudError extends Error {}

async function postJson(cfg, urlPath, body, { retries = 3, timeoutMs } = {}) {
	if (!cfg.apiKey) {
		throw new CloudError(
			`no API key configured (set provider.apiKey in ${CONFIG_PATH}, or export SILICONFLOW_API_KEY)`,
		);
	}
	const url = `${cfg.baseUrl}${urlPath}`;
	const budget = timeoutMs ?? cfg.provider.timeoutMs;
	let lastErr;
	for (let attempt = 1; attempt <= retries; attempt++) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), budget);
		try {
			const res = await fetch(url, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${cfg.apiKey}`,
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
				signal: controller.signal,
			});
			if (!res.ok) {
				const detail = (await res.text().catch(() => "")).slice(0, 300);
				const retryable = res.status === 429 || res.status >= 500;
				lastErr = new CloudError(`${res.status} ${res.statusText} from ${urlPath}: ${detail}`);
				if (!retryable || attempt === retries) throw lastErr;
				await sleep(400 * attempt);
				continue;
			}
			return await res.json();
		} catch (err) {
			lastErr = err;
			const retryable = err.name === "AbortError" || err instanceof TypeError;
			if (!retryable || attempt === retries) {
				throw new CloudError(
					err.name === "AbortError"
						? `request to ${urlPath} timed out after ${budget}ms`
						: `${urlPath} failed: ${err.message}`,
				);
			}
			await sleep(400 * attempt);
		} finally {
			clearTimeout(timer);
		}
	}
	throw lastErr;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------ cloud clients */

/** Embed an array of texts. Returns one float array per input, in input order. */
export async function embedTexts(cfg, texts, { log } = {}) {
	if (texts.length === 0) return [];
	const batchSize = Math.max(1, cfg.search.embedBatchSize);
	const out = new Array(texts.length);
	for (let i = 0; i < texts.length; i += batchSize) {
		const batch = texts.slice(i, i + batchSize);
		const payload = { model: cfg.provider.embedModel, input: batch, encoding_format: "float" };
		const json = await postJson(cfg, "/embeddings", payload);
		const data = json?.data;
		if (!Array.isArray(data) || data.length !== batch.length) {
			throw new CloudError(`/embeddings returned ${data?.length ?? 0} vectors for ${batch.length} inputs`);
		}
		for (const item of data) {
			const idx = typeof item.index === "number" ? item.index : out.findIndex((v) => v === undefined);
			const vec = item.embedding;
			if (!Array.isArray(vec) || vec.length === 0) throw new CloudError("/embeddings returned an empty vector");
			out[i + idx] = vec;
		}
		log?.(`embedded ${Math.min(i + batchSize, texts.length)}/${texts.length}`);
	}
	return out;
}

/** Rerank documents for a query via an OpenAI-compatible /rerank endpoint. */
export async function rerankDocs(cfg, query, documents, topN) {
	if (documents.length === 0) return [];
	const json = await postJson(cfg, "/rerank", {
		model: cfg.provider.rerankModel,
		query,
		documents,
		top_n: Math.min(topN, documents.length),
		return_documents: false,
	});
	const results = json?.results;
	if (!Array.isArray(results)) return [];
	return results
		.filter((r) => typeof r?.index === "number")
		.map((r) => ({ index: r.index, score: Number(r.relevance_score ?? r.score ?? 0) }));
}

/** Generate a few paraphrased/expanded queries with a cloud chat model. */
export async function expandQueries(cfg, query, variants) {
	const json = await postJson(
		cfg,
		"/chat/completions",
		{
			model: cfg.provider.chatModel,
			temperature: 0.3,
			max_tokens: 400,
			messages: [
				{
					role: "system",
					content:
						"你是检索查询扩展器。给定用户查询，输出 JSON 数组，包含若干条改写/近义/补充上下文的检索查询。" +
						`最多 ${variants} 条，每条为独立可检索的短语或句子，保留原意、专有名词、日期与标识符。` +
						"只输出 JSON 数组本身，不要解释、不要代码块。",
				},
				{ role: "user", content: query },
			],
		},
		{ timeoutMs: Math.min(cfg.provider.timeoutMs, 30_000) },
	);
	const content = json?.choices?.[0]?.message?.content ?? "";
	const parsed = extractJsonArray(content);
	return parsed
		.filter((s) => typeof s === "string" && s.trim())
		.map((s) => s.trim())
		.filter((s) => s !== query.trim())
		.slice(0, variants);
}

function extractJsonArray(text) {
	const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	const candidates = [fenced?.[1], text];
	for (const candidate of candidates) {
		if (!candidate) continue;
		const start = candidate.indexOf("[");
		const end = candidate.lastIndexOf("]");
		if (start === -1 || end <= start) continue;
		try {
			const parsed = JSON.parse(candidate.slice(start, end + 1));
			if (Array.isArray(parsed)) return parsed;
		} catch {
			/* try next candidate */
		}
	}
	return [];
}

/* ------------------------------------------------------------------ the index */

/* --------------------------------------------------- local query expansion */

/**
 * Locate qmd's own `dist/llm.js` next to the configured real qmd binary, so the
 * extension can reuse qmd's LlamaCpp (model cache, HF download, node-llama-cpp
 * wiring) instead of reimplementing it. Keeps working across qmd upgrades.
 */
function qmdLlmModulePath(cfg) {
	let bin = expandHome(cfg.realQmdPath);
	try {
		bin = fs.realpathSync(bin);
	} catch {
		/* keep the configured path */
	}
	let dir = path.dirname(bin);
	for (let i = 0; i < 6; i++) {
		const candidate = path.join(dir, "dist", "llm.js");
		if (fs.existsSync(candidate)) return candidate;
		const parent = path.dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return null;
}

let llmModulePromise = null;
let llmModuleError = null;

async function loadQmdLlm(cfg) {
	if (llmModuleError) throw llmModuleError;
	if (llmModulePromise) return llmModulePromise;
	const llmPath = qmdLlmModulePath(cfg);
	if (!llmPath) {
		llmModuleError = new CloudError(`cannot find qmd's dist/llm.js next to ${cfg.realQmdPath}`);
		throw llmModuleError;
	}
	// qmd's own launcher sets this before the native addon loads (bin/qmd:51);
	// without it Metal can abort on Apple Silicon at process exit (issue #368).
	if (process.platform === "darwin") process.env.GGML_METAL_NO_RESIDENCY ??= "1";
	const endpoint = cfg.search.expansion.hfEndpoint;
	if (endpoint && !process.env.HF_ENDPOINT) process.env.HF_ENDPOINT = endpoint;
	llmModulePromise = import(pathToFileURL(llmPath).href);
	return llmModulePromise;
}

/** Model URI for local expansion: explicit config, else index.yml, else qmd's default. */
export function localExpansionModel(cfg) {
	const explicit = cfg.search.expansion.localModel;
	if (explicit) return expandHome(explicit);
	return readQmdModels(cfg).generate || "hf:tobil/qmd-query-expansion-0.6B-gguf/qmd-query-expansion-0.6B-q4_k_m.gguf";
}

function localMarkerPath(cfg) {
	return path.join(cfg.stateDir, "local-expansion.json");
}

/**
 * Fallback lookup in qmd's own model cache (~/.cache/qmd/models), so a wiped
 * stateDir (or a second config sharing the cache) does not force a re-download.
 */
function findModelInQmdCache(modelUri) {
	if (!modelUri.startsWith("hf:")) return null;
	const file = modelUri.split("/").pop();
	const stem = file.replace(/\.gguf$/i, "");
	const dir = path.join(os.homedir(), ".cache", "qmd", "models");
	try {
		const hit = fs
			.readdirSync(dir)
			.find((name) => name.endsWith(".gguf") && name.includes(stem));
		return hit ? path.join(dir, hit) : null;
	} catch {
		return null;
	}
}

/** Path of the local expansion model iff it is already in qmd's model cache. */
export function cachedLocalExpansionModel(cfg) {
	const model = localExpansionModel(cfg);
	try {
		const marker = JSON.parse(fs.readFileSync(localMarkerPath(cfg), "utf8"));
		if (marker?.model === model && marker.path && fs.existsSync(marker.path)) return marker.path;
	} catch {
		/* fall through to the model cache */
	}
	return findModelInQmdCache(model);
}

/**
 * Download (or locate) the local expansion model. Downloads are explicit:
 * a search never starts a multi-hundred-MB transfer behind pi-memory's back.
 */
export async function ensureLocalExpansionModel(cfg, { download = false, log } = {}) {
	const cached = cachedLocalExpansionModel(cfg);
	if (cached) return cached;
	if (!download) return null;
	const model = localExpansionModel(cfg);
	const { LlamaCpp } = await loadQmdLlm(cfg);
	const llm = new LlamaCpp({ generateModel: model });
	try {
		log?.(`fetching local query-expansion model: ${model}`);
		const modelPath = await llm.resolveModel(model);
		fs.mkdirSync(cfg.stateDir, { recursive: true });
		fs.writeFileSync(
			localMarkerPath(cfg),
			`${JSON.stringify({ model, path: modelPath, fetchedAt: new Date().toISOString() }, null, 2)}\n`,
		);
		return modelPath;
	} finally {
		await llm.dispose().catch(() => {});
	}
}

function withTimeout(promise, ms, message) {
	if (!ms || ms <= 0) return promise;
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => reject(new CloudError(message)), ms);
		promise.then(
			(value) => {
				clearTimeout(timer);
				resolve(value);
			},
			(err) => {
				clearTimeout(timer);
				reject(err);
			},
		);
	});
}

/**
 * Expand a query with qmd's local query-expansion GGUF (`lex`/`vec`/`hyde`
 * variants). Returns [] when the model was never pulled, so `query` still works.
 */
export async function expandQueriesLocal(cfg, query, variants) {
	const modelPath = cachedLocalExpansionModel(cfg);
	if (!modelPath) {
		warn("local query expansion model not downloaded yet; run: qmd __cloud pull");
		return [];
	}
	const { LlamaCpp } = await loadQmdLlm(cfg);
	const llm = new LlamaCpp({ generateModel: modelPath });
	let settled = false;
	const expansion = llm.expandQuery(query).then(
		(value) => {
			settled = true;
			return value;
		},
		(err) => {
			settled = true;
			throw err;
		},
	);
	const timeoutMs = cfg.search.expansion.localTimeoutMs;
	try {
		const items = await withTimeout(
			expansion,
			timeoutMs,
			`local query expansion did not finish within ${timeoutMs}ms`,
		);
		const out = [];
		const seen = new Set();
		for (const item of items ?? []) {
			const text = String(item?.text ?? "").trim();
			if (!text || text === query.trim() || seen.has(text)) continue;
			seen.add(text);
			out.push(text);
			if (out.length >= variants) break;
		}
		return out;
	} finally {
		// Only dispose once the generation is done: disposing mid-flight would free
		// Metal resources the session is still using. Otherwise leave it to process exit.
		if (settled) await llm.dispose().catch(() => {});
	}
}

/** Route to the configured expansion backend. Returns [] when it cannot run. */
async function expandQueriesFor(cfg, query) {
	const exp = cfg.search.expansion;
	if (!exp.enabled || exp.provider === "off") return [];
	// Cloud search cannot run without a key, so do not burn a local model load on it.
	if (!cfg.apiKey) return [];
	if (exp.provider === "cloud") return expandQueries(cfg, query, exp.variants);
	try {
		return await expandQueriesLocal(cfg, query, exp.variants);
	} catch (err) {
		warn(`local query expansion failed: ${err.message}`);
		if (!exp.cloudFallback) return [];
		return expandQueries(cfg, query, exp.variants);
	}
}

function expansionSummary(cfg) {
	const exp = cfg.search.expansion;
	if (!exp.enabled || exp.provider === "off") return "off";
	if (exp.provider === "cloud") return `cloud ${cfg.provider.chatModel}`;
	const model = localExpansionModel(cfg);
	const cached = cachedLocalExpansionModel(cfg);
	return `local ${model}${cached ? " (cached)" : " (not downloaded — run: qmd __cloud pull)"}`;
}

function indexPath(cfg) {
	return path.join(cfg.stateDir, "index.json");
}

export function loadIndex(cfg) {
	const p = indexPath(cfg);
	if (!fs.existsSync(p)) return { version: 1, model: cfg.provider.embedModel, collections: {} };
	try {
		const parsed = JSON.parse(fs.readFileSync(p, "utf8"));
		parsed.collections = parsed.collections ?? {};
		return parsed;
	} catch (err) {
		process.stderr.write(`cloud-qmd: index unreadable (${err.message}); rebuilding\n`);
		return { version: 1, model: cfg.provider.embedModel, collections: {} };
	}
}

export function saveIndex(cfg, index) {
	fs.mkdirSync(cfg.stateDir, { recursive: true });
	const tmp = `${indexPath(cfg)}.tmp`;
	const serialize = (v) => {
		if (Array.isArray(v)) return v.map((x) => (typeof x === "number" ? Math.round(x * 1e6) / 1e6 : x));
		return v;
	};
	fs.writeFileSync(tmp, JSON.stringify(index, (k, v) => (k === "vec" ? serialize(v) : v)));
	fs.renameSync(tmp, indexPath(cfg));
}

function walkMarkdown(root, pattern) {
	const out = [];
	const re = globToRegExp(pattern);
	const walk = (dir) => {
		let entries;
		try {
			entries = fs.readdirSync(dir, { withFileTypes: true });
		} catch {
			return;
		}
		for (const entry of entries) {
			if (entry.name.startsWith(".")) continue;
			if (entry.name === "node_modules") continue;
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) walk(full);
			else if (entry.isFile()) {
				const rel = path.relative(root, full).split(path.sep).join("/");
				if (re.test(rel)) out.push({ abs: full, rel });
			}
		}
	};
	walk(root);
	return out.sort((a, b) => a.rel.localeCompare(b.rel));
}

function globToRegExp(pattern) {
	const source = String(pattern ?? "**/*.md")
		.split(",")
		.map((p) => p.trim())
		.filter(Boolean)
		.map((p) => {
			let rx = "";
			for (let i = 0; i < p.length; i++) {
				const ch = p[i];
				if (ch === "*" && p[i + 1] === "*") {
					rx += ".*";
					i++;
					if (p[i + 1] === "/") i++;
				} else if (ch === "*") rx += "[^/]*";
				else if (ch === "?") rx += "[^/]";
				else rx += ch.replace(/[.+^${}()|[\]\\]/g, "\\$&");
			}
			return `(?:${rx})`;
		})
		.join("|");
	return new RegExp(`^(?:${source})$`);
}

function fileFingerprint(abs) {
	const st = fs.statSync(abs);
	return `${st.size}:${Math.floor(st.mtimeMs)}`;
}

/**
 * Refresh the vector index for a collection: walk markdown files, chunk changed
 * ones, embed only unseen chunks. Returns a summary.
 */
export async function refreshIndex(cfg, collectionName, { force = false, log, deadline } = {}) {
	const resolved = resolveCollection(cfg, collectionName);
	if (!resolved) throw new CloudError(`unknown collection "${collectionName}"`);
	if (resolved.root === cfg.stateDir || !fs.existsSync(resolved.root)) {
		return { added: 0, removed: 0, chunks: 0, files: 0, skipped: true, reason: "collection root missing" };
	}
	const index = loadIndex(cfg);
	const coll = index.collections[collectionName] ?? { root: resolved.root, files: {} };
	coll.root = resolved.root;
	index.collections[collectionName] = coll;

	// Embedding vectors are not comparable across models, so a model change or a
	// change in vector length forces a full rebuild of this collection.
	const modelChanged = Boolean(index.model && index.model !== cfg.provider.embedModel);
	if (modelChanged) log?.(`embedding model changed (${index.model} → ${cfg.provider.embedModel}); re-embedding everything`);

	const diskFiles = walkMarkdown(resolved.root, resolved.pattern);
	const seen = new Set();
	const pending = [];
	let removed = 0;

	for (const file of diskFiles) {
		seen.add(file.rel);
		const fp = fileFingerprint(file.abs);
		const prev = coll.files[file.rel];
		const dimMatches = !index.dim || (prev?.chunks?.[0]?.vec?.length ?? index.dim) === index.dim;
		if (!force && !modelChanged && dimMatches && prev && prev.fp === fp && prev.chunks?.length) continue;
		if (prev) removed += prev.chunks?.length ?? 0;
		const text = fs.readFileSync(file.abs, "utf8");
		let chunks = chunkMarkdown(text, cfg.search);
		if (chunks.length > cfg.search.maxChunksPerDocument) {
			chunks = chunks.slice(0, cfg.search.maxChunksPerDocument);
		}
		coll.files[file.rel] = { fp, chunks: [], pending: true };
		pending.push({ rel: file.rel, abs: file.abs, chunks });
	}
	for (const rel of Object.keys(coll.files)) {
		if (!seen.has(rel)) {
			removed += coll.files[rel].chunks?.length ?? 0;
			delete coll.files[rel];
		}
	}

	// Embed everything still missing vectors.
	const todo = [];
	for (const item of pending) {
		const record = coll.files[item.rel];
		const missing = item.chunks
			.map((c, i) => ({ i, text: c.text }))
			.filter(({ i }) => !record.chunks[i]?.vec);
		if (missing.length) todo.push({ rel: item.rel, missing });
	}

	let added = 0;
	if (todo.length > 0) {
		if (deadline && Date.now() > deadline) {
			// Keep whatever we already have; the rest is embedded on the next pass.
			for (const rel of Object.keys(coll.files)) {
				if (coll.files[rel].pending && !coll.files[rel].chunks.length) delete coll.files[rel].pending;
			}
			saveIndex(cfg, index);
			return { added: 0, removed, chunks: countChunks(coll), files: diskFiles.length, truncated: true };
		}
		const flat = todo.flatMap((t) => t.missing.map((m) => ({ rel: t.rel, i: m.i, text: m.text })));
		log?.(`embedding ${flat.length} chunk(s) via ${cfg.provider.embedModel}`);
		const vectors = await embedTexts(
			cfg,
			flat.map((f) => f.text),
			{ log },
		);
		const dim = vectors[0]?.length ?? 0;
		for (let k = 0; k < flat.length; k++) {
			const item = flat[k];
			const record = coll.files[item.rel];
			const chunk = chunkMarkdown(
				fs.readFileSync(path.join(resolved.root, item.rel), "utf8"),
				cfg.search,
			)[item.i];
			record.chunks[item.i] = {
				text: chunk?.text ?? item.text,
				line: chunk?.line ?? 1,
				vec: vectors[k],
			};
			added++;
		}
		if (dim) {
			index.dim = dim;
			index.model = cfg.provider.embedModel;
		}
	}

	for (const rel of Object.keys(coll.files)) delete coll.files[rel].pending;
	saveIndex(cfg, index);
	return { added, removed, chunks: countChunks(coll), files: diskFiles.length };
}

function countChunks(coll) {
	let n = 0;
	for (const file of Object.values(coll.files ?? {})) n += (file.chunks ?? []).filter(Boolean).length;
	return n;
}

export function indexStats(cfg, collectionName) {
	const index = loadIndex(cfg);
	const collections = {};
	for (const [name, coll] of Object.entries(index.collections ?? {})) {
		collections[name] = { root: coll.root, documents: Object.keys(coll.files ?? {}).length, chunks: countChunks(coll) };
	}
	return {
		stateDir: cfg.stateDir,
		dim: index.dim ?? null,
		model: index.model ?? null,
		collections,
		requested: collectionName ? (collections[collectionName] ?? null) : null,
	};
}

/* ------------------------------------------------------------------- search */

export function cosine(a, b) {
	if (!a || !b || a.length !== b.length) return -1;
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (na === 0 || nb === 0) return -1;
	return dot / Math.sqrt(na * nb);
}

function makeDocId(text) {
	let h = 2166136261;
	for (let i = 0; i < text.length; i++) {
		h ^= text.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return `#${(h >>> 0).toString(16).padStart(8, "0").slice(0, 6)}`;
}

function virtualPath(collectionName, rel) {
	return `qmd://${collectionName}/${rel}`;
}

function titleOf(text) {
	const m = text.match(/^#{1,6}\s+(.+)$/m);
	if (m) return m[1].trim().slice(0, 120);
	const first = text.split(/\r?\n/).find((l) => l.trim());
	return (first ?? "").trim().slice(0, 120);
}

function snippetOf(text, query) {
	const clean = text.replace(/\s+/g, " ").trim();
	const terms = query
		.toLowerCase()
		.split(/[\s,，。、;；:：/|]+/)
		.filter((t) => t.length > 1);
	const lower = clean.toLowerCase();
	let at = -1;
	for (const t of terms) {
		const i = lower.indexOf(t);
		if (i !== -1 && (at === -1 || i < at)) at = i;
	}
	const window = 600;
	if (at === -1) return clean.slice(0, window);
	const start = Math.max(0, at - 150);
	return `${start > 0 ? "…" : ""}${clean.slice(start, start + window)}${clean.length > start + window ? "…" : ""}`;
}

/** Vector search across a collection. Returns ranked hits. */
export async function vectorSearch(cfg, collectionName, queries, limit) {
	const index = loadIndex(cfg);
	const coll = index.collections?.[collectionName];
	if (!coll) return [];

	// Refreshing the collection root lets us report OS paths in the results.
	const resolved = resolveCollection(cfg, collectionName);
	const root = resolved?.root ?? coll.root;

	const vectors = await embedTexts(cfg, queries);
	const best = new Map();
	for (const queryVec of vectors) {
		for (const [rel, file] of Object.entries(coll.files ?? {})) {
			for (const chunk of file.chunks ?? []) {
				if (!chunk?.vec) continue;
				const score = cosine(queryVec, chunk.vec);
				if (score < 0) continue;
				const key = `${rel}\u0000${chunk.line}`;
				const prev = best.get(key);
				if (!prev || score > prev.score) {
					best.set(key, { rel, chunk, score });
				}
			}
		}
	}

	return [...best.values()]
		.sort((a, b) => b.score - a.score)
		.slice(0, limit)
		.map((hit) => ({
			id: `${hit.rel}\u0000${hit.chunk.line}`,
			score: hit.score,
			file: virtualPath(collectionName, hit.rel),
			path: path.join(root, hit.rel),
			line: hit.chunk.line,
			title: titleOf(hit.chunk.text),
			snippet: snippetOf(hit.chunk.text, queries[0] ?? ""),
			text: hit.chunk.text,
		}));
}

/** Reciprocal Rank Fusion over several ranked lists of `{id}` items. */
export function rrfMerge(lists, k = 60) {
	const scores = new Map();
	const items = new Map();
	lists.forEach((list, listIndex) => {
		const weight = list.weight ?? 1;
		list.items.forEach((item, rank) => {
			const prev = scores.get(item.id) ?? 0;
			scores.set(item.id, prev + weight / (k + rank + 1));
			if (!items.has(item.id)) items.set(item.id, item);
		});
	});
	return [...scores.entries()]
		.sort((a, b) => b[1] - a[1])
		.map(([id, score]) => ({ ...items.get(id), score }));
}

/** Ask the real qmd for BM25 hits (no models involved). */
export function bm25Search(cfg, collectionName, query, limit) {
	const args = ["search", "--json", "-c", collectionName, "-n", String(limit), query];
	const res = spawnSync(cfg.realQmdPath, args, { encoding: "utf8", timeout: 20_000 });
	if (res.status !== 0 || !res.stdout) return [];
	try {
		const cleaned = String(res.stdout).replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "");
		const lines = cleaned.split(/\r?\n/);
		const start = lines.findIndex((l) => l.trimStart().startsWith("[") || l.trimStart().startsWith("{"));
		if (start === -1) return [];
		const parsed = JSON.parse(lines.slice(start).join("\n"));
		const rows = Array.isArray(parsed) ? parsed : (parsed.results ?? parsed.hits ?? []);
		return rows.map((row) => ({
			id: `${row.file}\u0000${row.line ?? 0}`,
			score: Number(row.score ?? 0),
			file: row.file,
			line: row.line,
			title: row.title,
			snippet: row.snippet,
			text: row.snippet ?? row.content ?? "",
		}));
	} catch {
		return [];
	}
}

/* --------------------------------------------------------------- CLI plumbing */

function parseSearchArgs(argv) {
	const opts = { collection: "pi-memory", collectionExplicit: false, limit: 5, json: false, queryParts: [] };
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "-c" || arg === "--collection") {
			opts.collection = argv[++i];
			opts.collectionExplicit = true;
		} else if (arg === "-n" || arg === "--limit") opts.limit = Number(argv[++i]) || 5;
		else if (arg === "-j" || arg === "--json") opts.json = true;
		else if (arg === "--full" || arg === "--explain" || arg === "--line-numbers") continue;
		else opts.queryParts.push(arg);
	}
	opts.limit = Math.max(1, Math.min(50, Math.floor(opts.limit)));
	opts.query = opts.queryParts.join(" ").trim();
	return opts;
}

function printResults(results) {
	const payload = results.map((r) => ({
		docid: makeDocId(`${r.file}:${r.line ?? 0}`),
		score: Math.round(r.score * 1000) / 1000,
		file: r.file,
		...(r.path ? { path: r.path } : {}),
		line: r.line ?? 1,
		...(r.title ? { title: r.title } : {}),
		...(r.snippet ? { snippet: r.snippet } : {}),
	}));
	if (payload.length === 0) {
		process.stdout.write("No results found.\n");
		return;
	}
	process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

function warn(message) {
	process.stderr.write(`cloud-qmd: ${message}\n`);
}

function delegate(cfg, argv, { inherit = true } = {}) {
	const res = spawnSync(cfg.realQmdPath, argv, { stdio: inherit ? "inherit" : ["ignore", "pipe", "inherit"] });
	if (res.error) {
		warn(`could not run real qmd at ${cfg.realQmdPath}: ${res.error.message}`);
		return 127;
	}
	if (!inherit && res.stdout) process.stdout.write(res.stdout);
	return res.status ?? 0;
}

async function cmdVsearch(cfg, argv) {
	const opts = parseSearchArgs(argv);
	if (!opts.query) {
		warn("vsearch requires a query");
		return 2;
	}
	const resolved = resolveCollection(cfg, opts.collection);
	if (!resolved) {
		warn(`unknown collection "${opts.collection}"; delegating to real qmd`);
		return delegate(cfg, ["vsearch", ...argv]);
	}
	try {
		if (cfg.search.inlineRefresh) {
			const deadline = Date.now() + cfg.search.inlineRefreshDeadlineMs;
			await refreshIndex(cfg, opts.collection, { deadline });
		}
		const hits = await vectorSearch(cfg, opts.collection, [opts.query], opts.limit);
		const stats = indexStats(cfg, opts.collection);
		const total = stats.requested?.chunks ?? 0;
		if (total === 0) {
			const files = stats.requested?.documents ?? 0;
			if (files > 0) warn("need embeddings for 1 or more documents; run: qmd embed");
		}
		printResults(hits);
		return 0;
	} catch (err) {
		warn(`${err.message}; falling back to keyword search`);
		return delegate(cfg, ["search", ...argv]);
	}
}

async function cmdQuery(cfg, argv) {
	const opts = parseSearchArgs(argv);
	if (!opts.query) {
		warn("query requires a query");
		return 2;
	}
	const resolved = resolveCollection(cfg, opts.collection);
	if (!resolved) {
		warn(`unknown collection "${opts.collection}"; delegating to real qmd`);
		return delegate(cfg, ["query", ...argv]);
	}
	const candidates = Math.max(opts.limit * 4, cfg.search.candidateLimit);
	try {
		// Expansion loads a local GGUF, so start it first and let the index refresh
		// overlap with it instead of adding to the total latency.
		const expansion = expandQueriesFor(cfg, opts.query).catch((err) => {
			warn(`query expansion skipped: ${err.message}`);
			return [];
		});

		if (cfg.search.inlineRefresh) {
			const deadline = Date.now() + cfg.search.inlineRefreshDeadlineMs;
			await refreshIndex(cfg, opts.collection, { deadline });
		}

		const variants = await expansion;
		const queries = variants.length ? [opts.query, ...variants] : [opts.query];

		const vectorHits = await vectorSearch(cfg, opts.collection, queries, candidates);
		const bm25Hits = bm25Search(cfg, opts.collection, opts.query, candidates);

		const fused = rrfMerge(
			[
				{ weight: cfg.search.hybrid.vectorWeight, items: vectorHits },
				{ weight: cfg.search.hybrid.bm25Weight, items: bm25Hits },
			],
			cfg.search.hybrid.rrfK,
		);
		if (fused.length === 0) {
			warn("no candidates; falling back to keyword search");
			return delegate(cfg, ["search", ...argv]);
		}

		let ranked = fused;
		if (cfg.search.rerank.enabled && fused.length > 1) {
			try {
				const top = fused.slice(0, candidates);
				const reranked = await rerankDocs(
					cfg,
					opts.query,
					top.map((r) => r.text || r.snippet || ""),
					opts.limit,
				);
				if (reranked.length) {
					ranked = reranked.map((r) => ({ ...top[r.index], score: r.score }));
				}
			} catch (err) {
				warn(`rerank skipped: ${err.message}`);
			}
		}

		printResults(ranked.slice(0, opts.limit));
		return 0;
	} catch (err) {
		warn(`${err.message}; falling back to keyword search`);
		return delegate(cfg, ["search", ...argv]);
	}
}

async function cmdEmbed(cfg, argv) {
	const opts = parseSearchArgs(argv);
	// `qmd embed` (no -c) means "refresh everything", like the real qmd.
	const known = Object.keys(cfg.collections).length ? Object.keys(cfg.collections) : Object.keys(readQmdCollections(cfg));
	const names = opts.collectionExplicit
		? [opts.collection]
		: known.length
			? known
			: ["pi-memory"];
	try {
		for (const name of names) {
			if (!resolveCollection(cfg, name)) continue;
			const stats = await refreshIndex(cfg, name, { log: (m) => warn(m) });
			warn(`indexed ${name}: +${stats.added} chunk(s), ${stats.chunks} total, ${stats.files} file(s)`);
		}
		return 0;
	} catch (err) {
		warn(`embed failed: ${err.message}`);
		return 1;
	}
}

async function cmdPull(cfg) {
	const model = localExpansionModel(cfg);
	const cached = cachedLocalExpansionModel(cfg);
	process.stdout.write(
		[
			"cloud-qmd: local query-expansion model",
			`  model:       ${model}`,
			`  hf endpoint: ${process.env.HF_ENDPOINT || cfg.search.expansion.hfEndpoint || "https://huggingface.co"}`,
			`  cache:       ${cached ?? "(not downloaded)"}`,
		].join("\n") + "\n",
	);
	try {
		const modelPath = await ensureLocalExpansionModel(cfg, { download: true, log: (m) => warn(m) });
		process.stdout.write(`  ready:       ${modelPath}\n`);
		const started = Date.now();
		const variants = await expandQueriesLocal(cfg, "记忆索引 检索", cfg.search.expansion.variants);
		process.stdout.write(`  warm-up:     ${variants.length} variant(s) in ${Date.now() - started}ms\n`);
		for (const v of variants) process.stdout.write(`    - ${v}\n`);
		return 0;
	} catch (err) {
		warn(`pull failed: ${err.message}`);
		return 1;
	}
}

async function cmdUpdate(cfg, argv) {
	// Keep qmd's own FTS index fresh (that is what keyword search uses), then
	// refresh the cloud vectors so semantic/deep search sees the same documents.
	const code = delegate(cfg, ["update", ...argv]);
	try {
		await cmdEmbed(cfg, []);
	} catch (err) {
		warn(`vector refresh failed: ${err.message}`);
	}
	return code;
}

async function cmdStatus(cfg) {
	const stats = indexStats(cfg, "pi-memory");
	const lines = [
		"cloud-qmd status",
		`  config:      ${CONFIG_PATH}${fs.existsSync(CONFIG_PATH) ? "" : " (missing, using defaults)"}`,
		`  enabled:     ${cfg.enabled}`,
		`  endpoint:    ${cfg.baseUrl}`,
		`  api key:     ${cfg.apiKey ? `set (${cfg.apiKey.slice(0, 6)}…, ${cfg.apiKey.length} chars)` : "MISSING"}`,
		`  embed model: ${cfg.provider.embedModel}`,
		`  rerank:      ${cfg.provider.rerankModel} (${cfg.search.rerank.enabled ? "on" : "off"})`,
		`  expansion:   ${expansionSummary(cfg)}`,
		`  real qmd:    ${cfg.realQmdPath}${fs.existsSync(cfg.realQmdPath) ? "" : " (NOT FOUND)"}`,
		`  state dir:   ${cfg.stateDir}`,
		`  index dim:   ${stats.dim ?? "n/a"} (model ${stats.model ?? "n/a"})`,
	];
	for (const [name, coll] of Object.entries(stats.collections)) {
		lines.push(`  collection:  ${name} → ${coll.documents} doc(s), ${coll.chunks} chunk(s)  [${coll.root}]`);
	}
	if (Object.keys(stats.collections).length === 0) lines.push("  collection:  (index empty — run: qmd embed)");
	process.stdout.write(`${lines.join("\n")}\n`);
	return 0;
}

async function cmdTest(cfg) {
	process.stdout.write("cloud-qmd self-test\n");
	let failed = 0;
	const step = async (name, fn) => {
		try {
			const detail = await fn();
			process.stdout.write(`  ✓ ${name}${detail ? ` — ${detail}` : ""}\n`);
		} catch (err) {
			failed++;
			process.stdout.write(`  ✗ ${name} — ${err.message}\n`);
		}
	};

	await step("config", async () => (fs.existsSync(CONFIG_PATH) ? CONFIG_PATH : "using defaults (no config file)"));
	await step("real qmd", async () => {
		const res = spawnSync(cfg.realQmdPath, ["--version"], { encoding: "utf8", timeout: 20_000 });
		if (res.status !== 0) throw new Error(`exit ${res.status}`);
		return String(res.stdout).trim().split("\n")[0];
	});
	await step("api key", async () => {
		if (!cfg.apiKey) throw new Error(`no key (${cfg.provider.apiKey})`);
		return `${cfg.apiKey.slice(0, 6)}… (${cfg.apiKey.length} chars)`;
	});
	await step("embeddings", async () => {
		const [vec] = await embedTexts(cfg, ["cloud-qmd connectivity probe"]);
		if (!vec?.length) throw new Error("empty vector");
		return `${vec.length} dims via ${cfg.provider.embedModel}`;
	});
	await step("rerank", async () => {
		if (!cfg.search.rerank.enabled) return "disabled";
		const res = await rerankDocs(cfg, "memory", ["alpha memory note", "unrelated weather report"], 1);
		if (!res.length) throw new Error("no results");
		return `${cfg.provider.rerankModel} → top index ${res[0].index}`;
	});
	await step("query expansion", async () => {
		const exp = cfg.search.expansion;
		if (!exp.enabled || exp.provider === "off") return "disabled";
		const variants = await expandQueriesFor(cfg, "记忆索引");
		if (exp.provider === "local" && !cachedLocalExpansionModel(cfg)) {
			throw new Error(`local model not downloaded (run: qmd __cloud pull) — ${localExpansionModel(cfg)}`);
		}
		return variants.length ? `${variants.length} variant(s) via ${exp.provider}` : "none returned (non-fatal)";
	});
	await step("index refresh", async () => {
		const stats = await refreshIndex(cfg, "pi-memory", { force: false });
		return `+${stats.added} chunk(s), ${stats.chunks} total, ${stats.files} file(s)`;
	});
	process.stdout.write(failed === 0 ? "\nall checks passed\n" : `\n${failed} check(s) failed\n`);
	return failed === 0 ? 0 : 1;
}

/* --------------------------------------------------------------------- entry */

export async function run(argv) {
	const cfg = loadConfig();
	const [command, ...rest] = argv;

	if (command === "__cloud" || command === "cloud" || command?.startsWith("__cloud-")) {
		const sub = command.startsWith("__cloud-") ? command.slice(8) : rest.shift();
		switch (sub) {
			case "status":
				return cmdStatus(cfg);
			case "test":
				return cmdTest(cfg);
			case "reindex":
				return cmdEmbed(cfg, rest);
			case "pull":
				return cmdPull(cfg);
			case "selftest":
			case "smoke": {
				const script = path.join(AGENT_DIR, "extensions", "cloud-qmd", "test", "smoke.mjs");
				if (!fs.existsSync(script)) {
					warn(`selftest script missing at ${script}`);
					return 1;
				}
				const res = spawnSync(process.execPath, [script], { stdio: "inherit", timeout: 300_000 });
				return res.status ?? 1;
			}
			case "config":
				return cmdStatus(cfg);
			case "path":
				process.stdout.write(`${CONFIG_PATH}\n`);
				return 0;
			default:
				process.stderr.write(
					"usage: qmd __cloud <status|test|reindex|pull|selftest|config|path>\n",
				);
				return 2;
		}
	}

	if (!cfg.enabled) return delegate(cfg, argv);

	switch (command) {
		case "vsearch":
			return cmdVsearch(cfg, rest);
		case "query":
			return cmdQuery(cfg, rest);
		case "embed":
			return cmdEmbed(cfg, rest);
		case "update":
			return cmdUpdate(cfg, rest);
		default:
			return delegate(cfg, argv);
	}
}

export {
	CloudError,
	virtualPath,
	snippetOf,
	walkMarkdown,
	globToRegExp,
	fileFingerprint,
};
