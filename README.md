# cloud-qmd

Route **`qmd`'s vector search and reranking through cloud models** (SiliconFlow, or any
OpenAI-compatible endpoint), so `pi-memory`'s semantic search no longer needs the ~640 MB of
local GGUF embedding and reranker models. Query expansion for `query` (deep search) reuses
qmd's own small local GGUF by default (0.6B q4, 378 MB, optional).

[中文说明 →](README.zh-CN.md)

## How it works

`pi-memory` calls qmd via `execFile("qmd", …)` (i.e. through `PATH`), so this extension ships a
thin shim that pi puts at the front of `PATH`:

```
pi-memory ──execFile("qmd")──▶  <agentDir>/bin/qmd          (shim, injected first in PATH)
                                      │
                                      ├─ vsearch / query / embed / update ─▶ lib/engine.mjs (cloud)
                                      └─ everything else ─────────────────▶ real qmd binary (absolute path)
```

- No changes to `pi-memory`, no monkey-patching, no private test hooks.
- Embeddings and reranking always go to the cloud — qmd's local embedding/rerank models are never
  loaded, and upgrading qmd does not break the extension.
- Query expansion uses qmd's own `LlamaCpp` to load a local query-expansion GGUF (0.6B only, and it
  is **never** downloaded implicitly — run `qmd __cloud pull` once), and can be switched to a cloud
  chat model or turned off.
- Semantic commands go to the cloud; `search` (BM25/FTS), `collection`, `doctor`, `mcp`, … are
  passed through to the real qmd untouched.
- If the cloud is unreachable (no key / network failure), semantic commands silently degrade to
  real qmd's keyword search instead of failing.

## Install

```bash
git clone https://github.com/seanjin0513/cloud-qmd.git
cd cloud-qmd
bash install.sh              # copies into ${PI_AGENT_DIR:-~/.pi/agent}/extensions/cloud-qmd
```

Restart pi (or run `/reload`) — pi discovers extensions in `<agentDir>/extensions/`, and the
extension writes `<agentDir>/bin/qmd` plus a config template on first load. Then:

```bash
# put your key in <agentDir>/cloud-qmd.json
/cloud-qmd test              # or: qmd __cloud test
```

`install.sh` refuses to overwrite an existing install; use `--force` to replace it (the old copy
is backed up next to it as `cloud-qmd.bak.<timestamp>`).

The extension writes `bin/qmd` and the config into `$PI_AGENT_DIR` when that variable is set, so if
you install into a custom agent dir (or load it with `pi -e <dir>`), export `PI_AGENT_DIR` to match —
otherwise the paths point at `~/.pi/agent`.

Requirements: Node.js ≥ 20, [qmd](https://github.com/tobilu/qmd) (`npm i -g @tobilu/qmd`),
and a `pi` install that supports extensions. No npm dependencies.

## Files

| Path | Purpose |
| --- | --- |
| `index.ts` | pi extension entry: injects `PATH`, `/cloud-qmd` command, footer status |
| `lib/engine.mjs` | Engine: cloud clients, chunking, vector index, hybrid retrieval |
| `test/smoke.mjs` | Offline end-to-end test (bundled mock provider, no API key needed) |
| `test/mock-provider.mjs` | The mock OpenAI-compatible endpoint used by the test |
| `install.sh` | Installs the extension into an agent dir |

Once installed, these files live at:

| Path | Purpose |
| --- | --- |
| `<agentDir>/extensions/cloud-qmd/index.ts` | extension entry |
| `<agentDir>/extensions/cloud-qmd/lib/engine.mjs` | engine |
| `<agentDir>/extensions/cloud-qmd/test/smoke.mjs` | offline self-test |
| `<agentDir>/bin/qmd` | the shim — **rewritten on every pi start, do not edit by hand** |
| `<agentDir>/cloud-qmd.json` | config (this is where the key goes) |
| `<agentDir>/cloud-qmd/index.json` | vector index (safe to delete, it is rebuilt) |

`<agentDir>` defaults to `~/.pi/agent` and can be overridden with `PI_AGENT_DIR`.

## Configuration

Edit `<agentDir>/cloud-qmd.json` and put your key in it (the default reads the
`SILICONFLOW_API_KEY` environment variable):

```json
{
  "provider": {
    "baseUrl": "https://api.siliconflow.cn/v1",
    "apiKey": "sk-xxxxxxxxxxxxxxxx",
    "embedModel": "BAAI/bge-m3",
    "rerankModel": "BAAI/bge-reranker-v2-m3",
    "chatModel": "Qwen/Qwen2.5-7B-Instruct"
  },
  "search": {
    "expansion": {
      "provider": "local",
      "localModel": "hf:tobil/qmd-query-expansion-0.6B-gguf/qmd-query-expansion-0.6B-q4_k_m.gguf"
    }
  }
}
```

`apiKey` accepts four forms: `env:NAME` (environment variable), `$NAME`, `file:/path/to/key`, or a
literal value.

Any OpenAI-compatible endpoint works (self-hosted vLLM/Ollama, another gateway, …) as long as it
exposes `/v1/embeddings`, `/v1/rerank` and `/v1/chat/completions`. If there is no `/rerank`,
set `search.rerank.enabled` to `false`.

### Query expansion (`query` / deep search)

`search.expansion.provider` takes one of three values:

| Value | Behaviour | Cost |
| --- | --- | --- |
| `local` (default) | qmd's own `LlamaCpp` loads the local query-expansion GGUF | Free, stays on your machine; each deep query costs ~3-6 s extra (model load + generation) and ~0.5-1.5 GB of transient RAM |
| `cloud` | Variants are generated by `provider.chatModel` | One extra chat call per deep query |
| `off` | No expansion, the original query only | Fastest, slightly lower recall |

`localModel` accepts `hf:user/repo/file.gguf` or a local file path; set it to `null` to use
`models.generate` from qmd's own `~/.config/qmd/index.yml`. The local model must be downloaded
once, explicitly:

```bash
qmd __cloud pull        # or /cloud-qmd pull; 0.6B q4 ≈ 378 MB, 1.7B q4 ≈ 1223 MB
```

> Direct access to huggingface.co is usually blocked from mainland China. The engine sets
> `HF_ENDPOINT=https://hf-mirror.com` automatically when downloading (override it in
> `search.expansion.hfEndpoint`, or export `HF_ENDPOINT` yourself).
> It works without the model too: `query` skips expansion, continues with vector + BM25 + rerank,
> and prints a notice.

> ⚠️ **Changing the embedding model requires rebuilding the index** (vectors from different models
> are not comparable). The engine detects an `embedModel` change and re-embeds everything, or run
> `/cloud-qmd reindex`.

## Commands

```
/cloud-qmd              # status (key, models, index size)
/cloud-qmd test         # online self-check: embedding / rerank / query expansion
/cloud-qmd pull         # download the local query-expansion model (one-off, 0.6B ≈ 378 MB)
/cloud-qmd selftest     # offline end-to-end self-test (local mock provider, 24 assertions)
/cloud-qmd reindex      # rebuild the vector index
/cloud-qmd enable       # enable cloud models
/cloud-qmd disable      # hand everything back to stock qmd
/cloud-qmd path         # print config / index paths
```

Equivalent CLI form (usable from pi's bash tool):

```bash
qmd __cloud status | test | selftest | reindex | pull | config | path
qmd embed                      # build/update the cloud vector index
qmd vsearch "query" -n 5       # semantic search (pi-memory's semantic mode)
qmd query   "query" -n 5       # deep search: expansion + vectors + BM25 (RRF) + rerank
qmd search  "query" -n 5       # stock qmd BM25 keyword search (passthrough)
```

`qmd vsearch|query --json` deliberately mirrors qmd's JSON shape so `pi-memory`'s `parseQmdJson`
can read it, and additionally emits `path` (the absolute path) so `memory_search` shows the real
file instead of a `qmd://` virtual path.

## Cost and latency

| Operation | Cloud requests | Typical latency |
| --- | --- | --- |
| First `embed` (N chunks) | ⌈N/24⌉ × `/embeddings` | proportional to N |
| `embed` with no doc changes | 0 | ~10 ms |
| `vsearch` (one query) | 1 × `/embeddings` | ~0.3–1 s |
| `query` | `local`: 0 (local GGUF); `cloud`: 1 × `/chat/completions` + 1 × `/embeddings` + 1 × `/rerank` | `local` ~3–8 s (incl. model load), `cloud` ~2–6 s |
| `update` after writing a memory | real qmd `update` + incremental embedding of changed files | in the background |

The index is **incremental**: files are fingerprinted by `size:mtime`, and only chunks with missing
vectors trigger a request. Each `vsearch`/`query` first performs a cheap staleness check (up to 10 s
by default; disable with `search.inlineRefresh: false`), and the `update` triggered after
`pi-memory` writes a memory refreshes vectors as well.

## Self-test

```bash
node ~/.pi/agent/extensions/cloud-qmd/test/smoke.mjs   # installed copy
node test/smoke.mjs                                    # from a clone
# or
qmd __cloud selftest
```

It starts a local mock OpenAI-compatible endpoint (random port, temp dir removed afterwards) and
covers: index construction, no duplicate requests on re-index, one embedding per query, the
RRF + rerank path, index sync on added/removed files, degradation when the endpoint dies,
complete passthrough with `enabled:false`, and the three local-expansion paths (skipped when not
downloaded / degraded when the model is corrupt / clean failure from `pull`).

Useful knobs: `CLOUD_QMD_SHIM=/path/to/qmd node test/smoke.mjs` tests a specific installed shim,
and `CLOUD_QMD_REAL_QMD=stub node test/smoke.mjs` forces a stub instead of the real qmd binary
(that is what CI does).

## Disable / rollback

- Temporarily: `/cloud-qmd disable` (or `"enabled": false`) → every command goes to stock qmd.
- Completely: delete `~/.pi/agent/bin/qmd` **and** this extension directory (the directory is what
  injects `bin/` into `PATH`), then remove `<agentDir>/cloud-qmd/` and `<agentDir>/cloud-qmd.json`.
  `pi-memory` notices nothing.
- To go back to local-model semantic search: delete `bin/qmd`, then run `qmd pull && qmd embed`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Footer shows `cloud-qmd: no key` | Put the key in `<agentDir>/cloud-qmd.json`, or `export SILICONFLOW_API_KEY=…` |
| `/cloud-qmd test` reports `✗ embeddings` | Invalid key, no balance, or the model is not enabled; a wrong `provider.baseUrl` looks the same |
| Recent memories are missing from results | `qmd __cloud reindex`; make sure `qmd embed` reports `+N chunk(s)` with N > 0 |
| Poor Chinese results | Use `BAAI/bge-m3` (already the default, best for Chinese/multilingual) or `Qwen/Qwen3-Embedding-0.6B` (requires a reindex) |
| Every search is slow | `query` reloads the local expansion model each time; set `search.expansion.provider: "off"` to skip it entirely (`semantic`/`vsearch` are unaffected) |
| `qmd __cloud pull` hangs or times out | Change `search.expansion.hfEndpoint` (e.g. to `https://huggingface.co` if you have a proxy), or download the gguf manually and point `localModel` at its absolute path |
| Want to confirm the real qmd is untouched | `qmd --version`, `qmd collection list`, `qmd doctor` all hit the real binary |

## Why not something else

- **Patch `pi-memory`'s source**: breaks on every upgrade and means maintaining a fork.
- **Use private hooks such as `_setExecFileForTest`**: depends on underscore APIs and requires the
  extension and `pi-memory` to share a module instance.
- **Give qmd a cloud provider**: qmd's `llm.js` only understands local GGUFs (`hf:` URIs or file
  paths) — the whole package contains no apiKey / baseURL / provider code and cannot talk to a
  cloud endpoint natively. That is exactly why this extension exists.

## License

MIT
