#!/usr/bin/env node
/**
 * Offline smoke test for cloud-qmd.
 *
 * Starts a local mock of an OpenAI-compatible provider (embeddings + rerank +
 * chat), points a throwaway config at it, then exercises the real shim the way
 * pi-memory does. No API key or network access required.
 *
 *   node ~/.pi/agent/extensions/cloud-qmd/test/smoke.mjs
 *
 * Exits non-zero on the first failed assertion.
 */

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXT_DIR = path.dirname(HERE);
const ENGINE = path.join(EXT_DIR, "lib", "engine.mjs");
// Where this checkout would be installed (used only to report the installed shim).
const INSTALLED_AGENT_DIR = path.dirname(path.dirname(EXT_DIR));
const INSTALLED_SHIM = path.join(INSTALLED_AGENT_DIR, "bin", "qmd");
const PORT = 20000 + Math.floor(Math.random() * 20000);
const WORK = fs.mkdtempSync(path.join(os.tmpdir(), "cloud-qmd-smoke-"));
const MEMORY = path.join(WORK, "memory");
const STATE = path.join(WORK, "state");
const CONFIG = path.join(WORK, "cloud-qmd.json");
// Test the engine in this checkout through a freshly generated shim, so the suite
// works both from ~/.pi/agent/extensions/cloud-qmd and from a plain git clone.
// Set CLOUD_QMD_SHIM to test an installed shim instead.
const SHIM = process.env.CLOUD_QMD_SHIM || path.join(WORK, "bin", "qmd");
// Passthrough commands are routed to the real qmd binary. If qmd is not installed
// (e.g. CI), a stub stands in so the routing itself can still be asserted.
const MAC_QMD = "/opt/homebrew/lib/node_modules/@tobilu/qmd/bin/qmd";
const REAL_QMD =
	process.env.CLOUD_QMD_REAL_QMD === "stub"
		? null
		: process.env.CLOUD_QMD_REAL_QMD || (fs.existsSync(MAC_QMD) ? MAC_QMD : null);
const REAL_QMD_STUB = path.join(WORK, "bin", "real-qmd-stub");
const REAL_QMD_PATH = REAL_QMD ?? REAL_QMD_STUB;

function writeShim(dest, enginePath) {
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(
		dest,
		`#!/usr/bin/env node
const { pathToFileURL } = require("node:url");
import(pathToFileURL(${JSON.stringify(enginePath)}).href)
	.then(async (engine) => process.exit((await engine.run(process.argv.slice(2))) ?? 0))
	.catch((err) => {
		process.stderr.write(\`cloud-qmd: \${err && err.message ? err.message : String(err)}\n\`);
		process.exit(1);
	});
`,
	);
	fs.chmodSync(dest, 0o755);
}

let failures = 0;
const check = (name, ok, detail = "") => {
	console.log(`  ${ok ? "✓" : "✗"} ${name}${detail ? ` — ${detail}` : ""}`);
	if (!ok) failures++;
};

function run(args, configPath = CONFIG) {
	const env = { ...process.env, CLOUD_QMD_CONFIG: configPath };
	delete env.SILICONFLOW_API_KEY;
	const res = spawnSync(process.execPath, [SHIM, ...args], { encoding: "utf8", env, timeout: 120_000 });
	return { code: res.status ?? -1, out: res.stdout ?? "", err: res.stderr ?? "" };
}

const stats = async () => (await fetch(`http://127.0.0.1:${PORT}/stats`)).json();

function writeRealQmdStub() {
	fs.mkdirSync(path.dirname(REAL_QMD_STUB), { recursive: true });
	fs.writeFileSync(REAL_QMD_STUB, '#!/bin/sh\necho "REAL-QMD $*"\n');
	fs.chmodSync(REAL_QMD_STUB, 0o755);
}

async function main() {
	console.log(`cloud-qmd smoke test\n  workdir ${WORK}\n`);
	// --- fixtures -----------------------------------------------------------
	fs.mkdirSync(MEMORY, { recursive: true });
	fs.writeFileSync(
		path.join(MEMORY, "alpha.md"),
		"# 项目决策记录\n\n## 数据库选型\n我们选择了 PostgreSQL 而不是 MySQL，因为需要 JSONB 与部分索引。\n\n## 缓存策略\nRedis 只做热点缓存，TTL 默认 300 秒。\n",
	);
	fs.writeFileSync(
		path.join(MEMORY, "beta.md"),
		"# Deployment notes\n\nThe staging cluster runs on Kubernetes and deploys with ArgoCD.\nRollback uses the previous image tag.\n",
	);
	fs.writeFileSync(
		path.join(MEMORY, "gamma.md"),
		"# 用户偏好\n\n用户喜欢简洁的中文回答，不要 emoji。时区是 Asia/Shanghai。\n",
	);

	fs.writeFileSync(
		CONFIG,
		JSON.stringify(
			{
				stateDir: STATE,
				realQmdPath: REAL_QMD_PATH,
				collections: { "pi-memory": { root: MEMORY, pattern: "**/*.md" } },
				provider: {
					baseUrl: `http://127.0.0.1:${PORT}/v1`,
					apiKey: "smoke-test-key",
					embedModel: "mock-embed",
					rerankModel: "mock-rerank",
					chatModel: "mock-chat",
					timeoutMs: 20000,
				},
				// The mock provider has no local GGUF, so exercise the cloud path here and
				// test local expansion separately below.
				search: { expansion: { provider: "cloud" } },
			},
			null,
			"\t",
		),
	);

	// --- mock provider ------------------------------------------------------
	const provider = spawn(process.execPath, [path.join(HERE, "mock-provider.mjs")], {
		stdio: ["ignore", "ignore", "inherit"],
		env: { ...process.env, PORT: String(PORT) },
	});
	await new Promise((r) => setTimeout(r, 700));

	try {
		// --- wiring ---------------------------------------------------------
		if (!process.env.CLOUD_QMD_SHIM) writeShim(SHIM, ENGINE);
		if (!REAL_QMD) writeRealQmdStub();
		check("engine loads", fs.existsSync(ENGINE), ENGINE);
		check("shim present and executable", (fs.statSync(SHIM).mode & 0o111) !== 0, SHIM);
		if (process.env.CLOUD_QMD_SHIM) console.log(`  (testing an explicit shim: ${SHIM})`);
		else if (fs.existsSync(INSTALLED_SHIM))
			console.log(`  (installed shim exists at ${INSTALLED_SHIM} — run with CLOUD_QMD_SHIM=… to test it)`);

		const status = run(["__cloud", "status"]);
		check("`qmd __cloud status` runs", status.code === 0 && /cloud-qmd status/.test(status.out));
		check("status sees the api key", /api key:\s+set/.test(status.out));

		const version = run(["--version"]);
		check(
			"unknown commands are delegated to the real qmd binary",
			REAL_QMD ? /^qmd \d/.test(version.out.trim()) : version.out.trim() === "REAL-QMD --version",
			version.out.trim() + (REAL_QMD ? "" : " (qmd not installed → routing checked with a stub)"),
		);

		// --- indexing -------------------------------------------------------
		const embed = run(["embed"]);
		check("`qmd embed` builds the index", /\+3 chunk\(s\)/.test(embed.err), embed.err.trim().split("\n").pop());
		check("index file written", fs.existsSync(path.join(STATE, "index.json")));

		const s1 = await stats();
		const embedAgain = run(["embed"]);
		check("unchanged docs are not re-embedded", (await stats()).embed === s1.embed, `embed requests stayed ${s1.embed}`);
		check("re-index reports +0", /\+0 chunk\(s\)/.test(embedAgain.err));

		// --- semantic search ------------------------------------------------
		const vsearch = run(["vsearch", "--json", "-c", "pi-memory", "-n", "3", "为什么选择 PostgreSQL"]);
		let rows = [];
		try {
			rows = JSON.parse(vsearch.out);
		} catch {
			/* handled by the assertion below */
		}
		check("vsearch returns parseable JSON", Array.isArray(rows) && rows.length === 3, `${rows.length} rows`);
		check("vsearch result shape matches qmd", Boolean(rows[0]?.docid && rows[0]?.file && rows[0]?.path && rows[0]?.score));
		check("best hit is the PostgreSQL note", String(rows[0]?.path ?? "").endsWith("alpha.md"), String(rows[0]?.path));
		const oneEmbed = (await stats()).embed;
		run(["vsearch", "Redis", "-n", "1"]);
		check("one query costs one embedding request", (await stats()).embed === oneEmbed + 1);

		// --- deep search ----------------------------------------------------
		const before = await stats();
		const query = run(["query", "--json", "-c", "pi-memory", "-n", "2", "部署用的是什么工具"]);
		let qrows = [];
		try {
			qrows = JSON.parse(query.out);
		} catch {
			/* handled below */
		}
		const after = await stats();
		check("query returns results", Array.isArray(qrows) && qrows.length > 0, `${qrows.length} rows`);
		check("query called the chat model (expansion)", after.chat === before.chat + 1);
		check("query called the reranker", after.rerank === before.rerank + 1);

		// --- engine API: the cloud expansion parser --------------------------
		// The mock chat endpoint answers with the object shape `[{"query": …}]` that
		// Qwen3-8B actually produces, so a regression here would silently disable
		// query expansion instead of failing the search.
		process.env.CLOUD_QMD_CONFIG = CONFIG;
		const engineApi = await import(pathToFileURL(ENGINE).href);
		const variantsOut = await engineApi.expandQueries(engineApi.loadConfig(), "部署用的是什么工具", 3);
		check(
			"cloud expansion accepts object-shaped JSON arrays",
			variantsOut.length === 3 && variantsOut.every((v) => typeof v === "string" && v.includes("部署用的是什么工具")),
			JSON.stringify(variantsOut),
		);

		// --- incremental updates -------------------------------------------
		fs.writeFileSync(path.join(MEMORY, "delta.md"), "# 新笔记\n\n增量索引应当只嵌入变化的部分。\n");
		const inc = run(["embed"]);
		check("new file is embedded incrementally", /\+1 chunk\(s\)/.test(inc.err), inc.err.trim().split("\n").pop());
		fs.rmSync(path.join(MEMORY, "delta.md"));
		const dec = run(["embed"]);
		check("deleted file is dropped from the index", /\+0 chunk\(s\), 3 total/.test(dec.err), dec.err.trim().split("\n").pop());

		// --- failure handling ----------------------------------------------
		const deadConfig = path.join(WORK, "dead.json");
		fs.writeFileSync(
			deadConfig,
			JSON.stringify({
				stateDir: STATE,
				collections: { "pi-memory": { root: MEMORY, pattern: "**/*.md" } },
				provider: {
					baseUrl: "http://127.0.0.1:1/v1",
					apiKey: "x",
					embedModel: "m",
					rerankModel: "r",
					chatModel: "c",
					timeoutMs: 2000,
				},
			}),
		);
		const dead = run(["vsearch", "PostgreSQL", "-n", "1"], deadConfig);
		check("dead endpoint degrades to keyword search", /falling back to keyword search/.test(dead.err) && dead.code === 0);
		check("dead endpoint leaks no exception", !/TypeError|at Object\.|Unhandled/.test(dead.err));

		const offConfig = path.join(WORK, "off.json");
		fs.writeFileSync(offConfig, JSON.stringify({ enabled: false, realQmdPath: REAL_QMD_PATH }));
		const off = run(["vsearch", "PostgreSQL"], offConfig);
		// With the real qmd installed we assert the passthrough happened at all (its own
		// output varies by index); with the stub we assert the exact argv it received.
		const offOk = REAL_QMD
			? off.code === 0 && !/cloud-qmd/.test(off.err + off.out)
			: off.code === 0 && /REAL-QMD vsearch PostgreSQL/.test(off.out);
		check("enabled:false disables cloud entirely", offOk, (off.out + off.err).trim().split("\n")[0]);

		// --- local query expansion -----------------------------------------
		const localConfig = path.join(WORK, "local.json");
		const missingModel = path.join(WORK, "missing-model.gguf");
		fs.writeFileSync(
			localConfig,
			JSON.stringify({
				stateDir: STATE,
				collections: { "pi-memory": { root: MEMORY, pattern: "**/*.md" } },
				provider: {
					baseUrl: `http://127.0.0.1:${PORT}/v1`,
					apiKey: "smoke-test-key",
					embedModel: "mock-embed",
					rerankModel: "mock-rerank",
					chatModel: "mock-chat",
					timeoutMs: 20000,
				},
				search: { expansion: { provider: "local", localModel: missingModel, localTimeoutMs: 20000 } },
			}),
		);

		const localStatus = run(["__cloud", "status"], localConfig);
		check(
			"status reports the local expansion model",
			/expansion:\s+local /.test(localStatus.out) && /not downloaded/.test(localStatus.out),
			localStatus.out.split("\n").find((l) => /expansion:/.test(l))?.trim(),
		);

		const beforeLocal = await stats();
		const noModel = run(["query", "--json", "-c", "pi-memory", "-n", "2", "部署用的是什么工具"], localConfig);
		let noModelRows = [];
		try {
			noModelRows = JSON.parse(noModel.out);
		} catch {
			/* handled below */
		}
		check(
			"query works without the local model (warns, no chat fallback)",
			noModel.code === 0 &&
				/not downloaded/.test(noModel.err) &&
				noModelRows.length > 0 &&
				(await stats()).chat === beforeLocal.chat,
		);

		// Claim the model is cached, but point the marker at a non-GGUF file: the
		// real qmd LlamaCpp module gets imported and has to fail softly.
		const bogus = path.join(WORK, "not-a-model.bin");
		fs.writeFileSync(bogus, "definitely not a gguf file\n");
		fs.writeFileSync(
			path.join(STATE, "local-expansion.json"),
			JSON.stringify({ model: missingModel, path: bogus }),
		);
		const brokenLocal = run(["query", "--json", "-c", "pi-memory", "-n", "2", "部署用的是什么工具"], localConfig);
		let brokenRows = [];
		try {
			brokenRows = JSON.parse(brokenLocal.out);
		} catch {
			/* handled below */
		}
		check(
			"a broken local model degrades instead of failing the search",
			brokenLocal.code === 0 && /local query expansion failed/.test(brokenLocal.err) && brokenRows.length > 0,
			brokenLocal.err.trim().split("\n")[0],
		);
		fs.rmSync(path.join(STATE, "local-expansion.json"), { force: true });

		const pull = run(["__cloud", "pull"], localConfig);
		check(
			"`qmd __cloud pull` fails cleanly when the model cannot be fetched",
			pull.code !== 0 && /(pull failed|fetching local query-expansion model)/.test(pull.stdout + pull.err) && !/Unhandled/.test(pull.err),
			(pull.stdout + pull.err).trim().split("\n").slice(-1)[0],
		);
	} finally {
		provider.kill();
		fs.rmSync(WORK, { recursive: true, force: true });
	}

	console.log(failures === 0 ? "\nall checks passed\n" : `\n${failures} check(s) FAILED\n`);
	process.exit(failures === 0 ? 0 : 1);
}

await main();
