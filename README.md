# BugAgent 🐛

AI agent that monitors **Telegram groups**, detects bug reports using **LangChain + LangGraph + Qdrant**, and automatically creates tasks in **Taiga** via MCP.

```
Telegram long-polling (MCP)
        │
        ▼
  Embedding (Ollama / bge)
        │
        ▼
  Qdrant ANN + Reranker
        │  score > threshold?
        ▼
  LLM Classification (LangGraph node)
        │  is_bug = true?
        ▼
  LLM Draft Builder
        │
        ▼
  Taiga MCP → create_issue()
```

## Services (Docker Compose)

| Service | Image | Port |
|---|---|---|
| `qdrant` | `qdrant/qdrant` | 6333 (REST), 6334 (gRPC) |
| `ollama` | `ollama/ollama` | 11434 |
| `telegram-mcp` | `guangxiangdebizi/telegram-mcp` | 3000 |
| `taiga-mcp` | built from `./pytaiga-mcp` | 8001 |
| `agent` | built from `./agent` | — |

## Quick start

```bash
# 1. Copy and fill env vars
cp .env.example .env
$EDITOR .env

# 2. Build images
make build

# 3. Start everything
make up

# 4. Pull Ollama models (first run only)
make pull-models

# 5. Follow logs
make logs-agent
```

## Manual trigger

Reply to any message in a monitored chat and add `@bug-agent` — the agent
will create a task immediately, bypassing the score threshold.

## LangGraph dashboard (web) 🧭

Теперь есть отдельный веб-дашборд на базе **LangGraph CLI/Studio** для быстрого
тестирования и перенастройки логики без правок файлов.


> Если Studio открывается с ошибкой `TypeError: NetworkError when attempting to fetch resource`,
> не используйте `baseUrl=http://0.0.0.0:8123`.
> Рабочий вариант: вручную замените baseUrl на `http://localhost:8123`.
> Пример: `https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2Flocalhost%3A8123&mode=graph&render=interact`
- `runtime_overrides` для live-настроек

Пример `runtime_overrides`:

- `mcp.taiga_url` — переопределить Taiga MCP URL
- `mcp.sources.telegram` — переопределить MCP URL источника
- `detection.rerank_score_threshold` / `detection.bug_classify_threshold`
- `task_tracker.*` (project, placement, work_item_type, uniqueness_check_enabled)
- `llm.*` (provider, cloud_model, ollama_model, vllm_base_url, llamacpp_base_url)

Это позволяет из веба быстро:

1. Подключать другие MCP серверы,
2. Крутить пороги/политику дедупликации,
3. Менять placement/тип задачи,
4. Проверять поведение графа на тестовых сообщениях.

## Universal connectors

- **Messengers** are connected via `sources` in `config.yaml`.
      - Use `transport: rabbitmq` for push delivery from a bot adapter queue.
      - Use `transport: mcp` for polling adapters implementing the same MCP contract.
- **Task trackers** are connected via `task_tracker` in `config.yaml` and
      `TASK_TRACKER_PROVIDER` in `.env`.
      - Current plugin: `taiga`.
      - To add a new tracker, implement one adapter in `agent/src/trackers/`
            (factory + adapter class), no graph rewrite required.

## Tags

Edit `tags.md` at any time. The agent reloads it on every LLM call — no
restart needed.

## Environment variables

See `.env.example` for full documentation.

Key variables:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `ollama` \| `vllm` \| `llamacpp` \| `openai` \| `openrouter` \| `anthropic` |
| `TELEGRAM_CHAT_IDS` | Comma-separated chat IDs to monitor |
| `TAIGA_PROJECT_SLUG` | Target Taiga project |
| `TASK_TRACKER_PROVIDER` | Task tracker adapter (`taiga`) |
| `TASK_TRACKER_PROJECT` | Generic tracker project key/slug |
| `RERANK_SCORE_THRESHOLD` | Min reranker score to send to LLM (default 0.65) |
| `BUG_CLASSIFY_THRESHOLD` | Min LLM confidence to create task (default 0.75) |
| `RERANK_OPERATOR` | Rerank backend: `qdrant`, `vllm`, `llamacpp`, `ollama` or `ann_only` |
| `RERANK_BASE_URL` | Rerank operator address (defaults to `QDRANT_URL`) |
| `RERANK_MODEL` | Reranker model name used by operator |

## Architecture

```
TgTasksBot/
├── docker-compose.yml
├── .env.example
├── config.yaml            # static config (mounted into agent)
├── tags.md                # dynamic tags (hot-reloaded)
├── Makefile
├── agent/                 # Python AI agent
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       ├── main.py        # entrypoint + polling loop
│       ├── config.py      # settings
│       ├── dedup.py       # seen-message deduplication
│       ├── agent/
│       │   ├── graph.py   # LangGraph compiled graph
│       │   ├── nodes.py   # node implementations
│       │   └── state.py   # AgentState dataclass
│       ├── qdrant_store/
│       │   └── client.py  # embeddings + rerank search
│       ├── mcp/
│       │   ├── taiga_mcp.py     # Taiga MCP client
│       │   └── telegram_mcp.py  # Telegram MCP client (file download etc.)
│       └── telegram/
│           └── polling.py # long-polling loop
└── pytaiga-mcp/           # Taiga MCP server (submodule)
```
