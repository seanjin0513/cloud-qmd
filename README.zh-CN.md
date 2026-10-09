# cloud-qmd

让 `qmd` 的**向量检索与重排序走云端模型**（SiliconFlow / 任意 OpenAI 兼容端点），
从而让 `pi-memory` 的 semantic 检索不再需要下载 ~640MB 的本地 embedding 模型和 reranker。
`query`（deep）模式的查询扩展默认复用 qmd 自带的轻量本地 GGUF（0.6B q4，378MB，可选）。

[English README →](README.md)

## 安装

```bash
git clone https://github.com/seanjin0513/cloud-qmd.git
cd cloud-qmd
bash install.sh              # 拷到 ${PI_AGENT_DIR:-~/.pi/agent}/extensions/cloud-qmd
```

重启 pi（或 `/reload`）—— pi 会加载 `<agentDir>/extensions/` 下的扩展，扩展首次加载时
自己写出 `<agentDir>/bin/qmd` 和配置模板。然后把 key 填进 `<agentDir>/cloud-qmd.json`，
运行 `/cloud-qmd test`（或 `qmd __cloud test`）。

`install.sh` 默认拒绝覆盖已有安装；加 `--force` 才会替换（旧的那份会备份成
`cloud-qmd.bak.<时间戳>`）。

扩展把 shim 和配置写到 `$PI_AGENT_DIR`（设了的话），所以装到非默认 agentDir 时（或用
`pi -e <dir>` 加载时）记得 export `PI_AGENT_DIR`，否则路径会指向 `~/.pi/agent`。

依赖：Node.js ≥ 20、[qmd](https://github.com/tobilu/qmd)（`npm i -g @tobilu/qmd`）、
一个支持扩展的 pi。本扩展自身零 npm 依赖。

## 原理

`pi-memory` 是通过 `execFile("qmd", …)`（走 PATH）调用 qmd 的，所以本扩展：

```
pi-memory ──execFile("qmd")──▶  <agentDir>/bin/qmd          （shim，本扩展注入 PATH 首位）
                                      │
                                      ├─ vsearch / query / embed / update ─▶ lib/engine.mjs（云端）
                                      └─ 其它所有命令 ────────────────────▶ 真 qmd 二进制（绝对路径）
```

- 不修改 `pi-memory` 任何代码、不 monkey-patch、不注册私有测试钩子。
- embedding / rerank 永远走云端，不加载本地 embedding/rerank 模型；qmd 自身升级不影响本扩展。
- 查询扩展默认用 qmd 自带的 `LlamaCpp` 加载本地 query-expansion GGUF（只占 0.6B，且**不会**在搜索时隐式下载，要显式 `qmd __cloud pull`），可改成云端 chat 或关闭。
- 语义命令走云端；`search`（BM25/FTS）、`collection`、`doctor`、`mcp` 等**全部原样透传**给真 qmd。
- 云端不可用时（无 key / 网络故障），语义命令自动降级为真 qmd 的关键词检索，不会报错中断。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `index.ts` | pi 扩展入口：注入 PATH、`/cloud-qmd` 命令、footer 状态 |
| `lib/engine.mjs` | 引擎：云端客户端、分块、向量索引、混合检索 |
| `test/smoke.mjs` | 离线端到端自测（自带 mock provider，不需要 key） |
| `test/mock-provider.mjs` | 自测用的 mock OpenAI 兼容端点 |
| `install.sh` | 安装到某个 agentDir |

## 安装后的文件

| 路径 | 说明 |
| --- | --- |
| `<agentDir>/extensions/cloud-qmd/index.ts` | pi 扩展入口：注入 PATH、`/cloud-qmd` 命令、footer 状态 |
| `<agentDir>/extensions/cloud-qmd/lib/engine.mjs` | 引擎：云端客户端、分块、向量索引、混合检索 |
| `<agentDir>/extensions/cloud-qmd/test/smoke.mjs` | 离线端到端自测（自带 mock provider，不需要 key） |
| `<agentDir>/bin/qmd` | shim，**每次 pi 启动自动重写**，不要手改 |
| `<agentDir>/cloud-qmd.json` | 配置（key 放这里） |
| `<agentDir>/cloud-qmd/index.json` | 向量索引（可随时删掉重建） |

`<agentDir>` 默认为 `~/.pi/agent`，可用 `PI_AGENT_DIR` 覆盖。

## 配置

编辑 `<agentDir>/cloud-qmd.json`，把 key 填进去（默认读环境变量 `SILICONFLOW_API_KEY`）：

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

`apiKey` 支持四种写法：`env:NAME`（环境变量）、`$NAME`、`file:/path/to/key`、字面量。

改成任何 OpenAI 兼容端点都可以（自建 vLLM/Ollama、别的中转），只要它提供
`/v1/embeddings`、`/v1/rerank`、`/v1/chat/completions`。缺 `/rerank` 时把
`search.rerank.enabled` 设为 `false`。

### 查询扩展（`query` / deep 模式）

`search.expansion.provider` 三选一：

| 值 | 行为 | 成本 / 代价 |
| --- | --- | --- |
| `local`（默认） | 用 qmd 自带的 `LlamaCpp` 加载 qmd 的 query-expansion GGUF | 免费、不出本机；每次 deep 查询多 ~3-6s（加载模型 + 生成），瞬时内存 ~0.5-1.5GB |
| `cloud` | 用 `provider.chatModel` 生成变体 | 每次 deep 查询多一次 chat 调用 |
| `off` | 不做扩展，只用原查询 | 最快，召回略低 |

`localModel` 支持 `hf:user/repo/file.gguf` 或本地文件路径；设为 `null` 时用 qmd 自己
`~/.config/qmd/index.yml` 里的 `models.generate`。本地模型需要显式下载一次：

```bash
qmd __cloud pull        # 或 /cloud-qmd pull；0.6B q4 ≈ 378MB，1.7B q4 ≈ 1223MB
```

> 国内直连 huggingface.co 通常不通。引擎下载时会自动设置 `HF_ENDPOINT=https://hf-mirror.com`
> （可在 `search.expansion.hfEndpoint` 改，或自行 export `HF_ENDPOINT`）。
> 没下载也能用：`query` 会跳过扩展、继续向量 + BM25 + rerank 并打印一条提示。

> ⚠️ **换 embedding 模型必须重建索引**（不同模型的向量不可比）。引擎靠 `embedModel` 字段
> 自动识别模型变化并全量重嵌，也可以手动 `/cloud-qmd reindex`。

## 命令

```
/cloud-qmd              # 状态（key、模型、索引规模）
/cloud-qmd test         # 联网自检：embedding / rerank / 查询扩展
/cloud-qmd pull         # 下载本地查询扩展模型（一次性，0.6B ≈ 378MB）
/cloud-qmd selftest     # 离线端到端自检（本地 mock provider，24 项断言）
/cloud-qmd reindex      # 重建向量索引
/cloud-qmd enable       # 启用云端
/cloud-qmd disable      # 全部交给原版 qmd
/cloud-qmd path         # 打印配置/索引路径
```

命令行等价形式（在 pi 的 bash 里可直接用）：

```bash
qmd __cloud status | test | selftest | reindex | pull | config | path
qmd embed                      # 建/更新云端向量索引
qmd vsearch "查询" -n 5         # 语义检索（pi-memory 的 semantic 模式）
qmd query   "查询" -n 5         # 深度检索：查询扩展 + 向量 + BM25(RRF) + rerank
qmd search  "查询" -n 5         # 真 qmd 的 BM25 关键词检索（透传）
```

`qmd vsearch|query --json` 的输出刻意对齐 qmd 的 JSON 结构，`pi-memory` 的
`parseQmdJson` 可直接解析；同时额外输出 `path`（绝对路径），使 `memory_search`
显示真实文件路径而不是 `qmd://` 虚拟路径。

## 成本与延迟

| 操作 | 云端请求数 | 典型延迟 |
| --- | --- | --- |
| 首次 `embed`（N 个 chunk） | ⌈N/24⌉ 次 `/embeddings` | 与 N 成正比 |
| 文档未变时的 `embed` | 0 | ~10ms |
| `vsearch`（1 条查询） | 1 次 `/embeddings` | ~0.3–1s |
| `query` | `local`：0 次（本地 GGUF）；`cloud`：1 次 `/chat/completions` + 1 次 `/embeddings` + 1 次 `/rerank` | `local` ~3–8s（含加载模型）、`cloud` ~2–6s |
| 写入记忆后的 `update` | 真 qmd `update` + 增量嵌入变化文件 | 后台执行 |

索引是**增量**的：按 `size:mtime` 指纹判断文件是否变化，只有缺失向量的 chunk 才会发起请求。
每次 `vsearch`/`query` 会先做一次廉价的过期检查（默认最多等 10s，可设
`search.inlineRefresh: false` 关掉）；`pi-memory` 写完记忆后触发的 `update` 也会顺带刷新向量。

## 自测

```bash
node ~/.pi/agent/extensions/cloud-qmd/test/smoke.mjs   # 已安装的那份
node test/smoke.mjs                                    # 仓库里直接跑
# 或
qmd __cloud selftest
```

会在本地起一个 mock 的 OpenAI 兼容端点（随机端口，用完即删临时目录），
覆盖：索引构建、增量不重复请求、查询只发 1 次 embedding、RRF+rerank 链路、
新增/删除文件的索引同步、端点挂掉时的降级、`enabled:false` 完全透传，
以及本地查询扩展的三条路径（未下载时跳过 / 模型损坏时降级 / `pull` 失败时干净报错）。

两个开关：`CLOUD_QMD_SHIM=/path/to/qmd node test/smoke.mjs` 测某个已安装的 shim；
`CLOUD_QMD_REAL_QMD=stub node test/smoke.mjs` 用桩替代真 qmd 二进制（CI 就是这么跑的）。

## 关闭 / 回滚

- 临时关闭：`/cloud-qmd disable`（或配置 `"enabled": false`）→ 所有命令交给原版 qmd。
- 完全移除：删掉 `~/.pi/agent/bin/qmd`（记得同时删掉 `bin` 目录在 PATH 里的注入 —— 即删掉本扩展目录），
  再把 `<agentDir>/cloud-qmd/`、`<agentDir>/cloud-qmd.json` 删掉即可。`pi-memory` 完全无感。
- 想恢复本地模型语义检索：删掉 `bin/qmd` 后执行 `qmd pull && qmd embed`。

## 故障排查

| 现象 | 处理 |
| --- | --- |
| footer 显示 `cloud-qmd: no key` | 把 key 写进 `<agentDir>/cloud-qmd.json` 或 `export SILICONFLOW_API_KEY=…` |
| `/cloud-qmd test` 报 `✗ embeddings` | key 无效、余额不足或该模型未开通；`provider.baseUrl` 写错也会这样 |
| 检索结果里没有最近写的记忆 | `qmd __cloud reindex`；确认 `qmd embed` 的 `+N chunk(s)` 确实大于 0 |
| 中文结果差 | 换 `BAAI/bge-m3`（默认已是最适合中文多语的）或 `Qwen/Qwen3-Embedding-0.6B`（需重建索引） |
| 每次搜索都慢 | `query` 每次要重新加载本地扩展模型；设 `search.expansion.provider: "off"` 可完全跳过（semantic/vsearch 不受影响） |
| `qmd __cloud pull` 卡住或超时 | 换 `search.expansion.hfEndpoint`（如 `https://huggingface.co` 有代理时）或手动把 gguf 放到本地后用绝对路径填 `localModel` |
| 想确认真 qmd 没被劫持 | `qmd --version`、`qmd collection list`、`qmd doctor` 都走真二进制 |

## 为什么不是别的方案

- **改 `pi-memory` 源码**：升级即失效，且要维护 fork。
- **用 `_setExecFileForTest` 等私有钩子**：依赖下划线 API，且要求扩展与 `pi-memory` 拿到同一模块实例。
- **给 qmd 换 cloud provider**：qmd 的 `llm.js` 只认本地 GGUF（`hf:` 或文件路径），全仓库没有任何
  apiKey / baseURL / provider 代码，原生接不了云端 —— 这正是本扩展存在的原因。
