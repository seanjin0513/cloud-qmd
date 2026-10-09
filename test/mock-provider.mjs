// Offline stand-in for an OpenAI-compatible embeddings/rerank/chat endpoint.
// Deterministic hashed bag-of-words vectors, so plumbing can be tested with no key.
import http from "node:http";

const DIM = 64;
const seen = { embed: 0, rerank: 0, chat: 0 };

function vector(text) {
	const vec = new Array(DIM).fill(0);
	const tokens = String(text)
		.toLowerCase()
		.split(/[^\p{L}\p{N}]+/u)
		.filter(Boolean);
	for (const token of tokens) {
		let h = 2166136261;
		for (const ch of token) {
			h ^= ch.codePointAt(0);
			h = Math.imul(h, 16777619);
		}
		vec[(h >>> 0) % DIM] += 1;
		// also a bigram-ish signal so CJK phrases overlap
		for (const ch of token) {
			let g = ch.codePointAt(0) * 2654435761;
			vec[(g >>> 0) % DIM] += 0.5;
		}
	}
	const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
	return vec.map((v) => v / norm);
}

const readBody = (req) =>
	new Promise((resolve) => {
		let raw = "";
		req.on("data", (c) => {
			raw += c;
		});
		req.on("end", () => {
			try {
				resolve(JSON.parse(raw));
			} catch {
				resolve({});
			}
		});
	});

http
	.createServer(async (req, res) => {
		const body = await readBody(req);
		const send = (obj, status = 200) => {
			res.writeHead(status, { "content-type": "application/json" });
			res.end(JSON.stringify(obj));
		};
		if (req.url.endsWith("/embeddings")) {
			seen.embed++;
			const input = Array.isArray(body.input) ? body.input : [body.input];
			return send({ data: input.map((text, index) => ({ index, embedding: vector(text) })) });
		}
		if (req.url.endsWith("/rerank")) {
			seen.rerank++;
			const docs = body.documents ?? [];
			const qv = vector(body.query ?? "");
			const scored = docs
				.map((d, index) => {
					const dv = vector(d);
					const dot = qv.reduce((s, v, i) => s + v * dv[i], 0);
					return { index, relevance_score: dot };
				})
				.sort((a, b) => b.relevance_score - a.relevance_score);
			return send({ results: scored.slice(0, body.top_n ?? scored.length) });
		}
		if (req.url.endsWith("/chat/completions")) {
			seen.chat++;
			const q = body.messages?.at(-1)?.content ?? "";
			// Deliberately the object shape `[{"query": …}]` that Qwen3-8B returns, so the
			// engine's tolerant parser is exercised.
			return send({
				choices: [
					{ message: { content: JSON.stringify([{ query: `${q} 相关记录` }, { query: `${q} 决策` }, { query: `${q} 偏好` }]) } },
				],
			});
		}
		if (req.url.endsWith("/stats")) return send(seen);
		return send({ error: { message: `no route ${req.url}` } }, 404);
	})
	.listen(Number(process.env.PORT || 8799), "127.0.0.1", () => console.log(`mock provider on :${process.env.PORT || 8799}`));
