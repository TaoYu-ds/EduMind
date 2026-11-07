# EduMind

EduMind 是面向 **在线教育场景** 的多 Agent 学习辅导应用，提供知识点讲解、作业辅导、学习计划编排、考试咨询和教师/人工升级等处理路径。它把意图识别、Agent 路由、结构化知识库检索、外部工具、学习画像和评测串在同一条对话链路中，适合学习 Agent 工程、验证教学业务流程，也适合在面试中讲清楚"多 Agent 为什么这样拆"。

本仓库是 **Python 后端**，使用 FastAPI 提供 HTTP API，并提供标准 MCP 服务端。配套 Web 前端和 Java 后端位于独立项目中；只运行本仓库即可通过 Swagger 或 HTTP 客户端使用 Python 服务。Java 版仍停留在电商客服场景，Python 版是教育培训场景的主实现。

当前三个教学角色分别负责讲题、排计划和考试事务：讲题只给思路与关键步骤，不代写作业；计划不承诺提分；考试只给流程与官方入口，不编造报名日期和分数线。真实的成绩查询、报名缴费、排课改期需要接入教育平台业务系统。

## 目录

- [项目能力与工作流程](#overview)
- [快速启动](#quickstart)
- [配置说明](#configuration)
- [在线 Embedding 配置](#intent-embedding-setup)
- [对话与知识库使用](#usage)
- [MCP 外部数据接入](#mcp)
- [Skills 业务规则](#skills)
- [评测与监控](#evaluation)
- [本地开发与测试](#development)
- [部署与数据管理](#deployment)
- [常见问题](#faq)
- [项目结构与详细文档](#documentation)

<a id="overview"></a>

## 项目能力与工作流程

| 能力 | 当前实现 |
|---|---|
| 三路意图识别 | LLM、向量相似度、Pattern 规则共同识别意图，覆盖知识点提问、作业辅导、学习计划、考试信息、课程推荐五类教学诉求，输出学科/章节/年级等实体。 |
| 主辅 Agent 路由 | 在 General（学习服务台）、Tutor（讲题）、Plan（计划）、Exam（考试）、Escalation（转人工/老师）之间选择处理路径；复合问题并行执行主辅 Agent 再合成回复。 |
| 结构化 RAG | 知识库按学科/章节/难度/文档类型打标签，检索时先元数据过滤再向量召回；`search_with_rewrite` 把"这个不会"改写成可检索的教材问法。 |
| 学习画像 | Redis 保存近期消息和摘要；ChromaDB 保存情景记忆及学习画像（年级、学科、薄弱章节、目标、可投入时间），讲解和排计划时作为输入。 |
| 标准 MCP | 通过官方 Python SDK 接入外部 MCP 服务，也将本地知识库暴露给其他 MCP 客户端。 |
| 动态 Skills | 从 Markdown 文件加载业务规范，根据 Agent 和关键词匹配后注入提示词，支持运行时重载。 |
| 故障处理与追踪 | 工具超时、缓存、熔断；专业 Agent 故障回退；通过请求 ID 查询调用轨迹。 |
| 评测与监控 | 异步执行意图及对话质量评测；提供在线统计、Prometheus 指标和路由降权恢复机制。 |

### 一条消息如何处理

以下使用纯文本流程图，普通 Markdown 预览器也能显示：

```text
用户 / Web 前端 / HTTP 客户端
             |
             v
POST /chat：鉴权、会话锁、重复请求回执检查
             |
             v
读取近期消息、摘要、相关情景记忆和用户画像
             |
             v
三路意图识别：LLM + Embedding + Pattern
             |
             v
路由决策
  |-- 需要澄清 ---------------------------> 澄清追问（知识点/作业/计划/考试）
  |-- 单 Agent --------------------------> 讲题 / 计划 / 考试回复，失败回退
  `-- 主辅 Agent 并行 -------------------> 合成回复
             |
             |  Agent 执行期间可按需调用：
             |  共享 RAG 工具 / 角色工具 / 获准的外部 MCP 工具
             v
记录路由结果和工具轨迹
             |
             v
Redis 事务写入用户消息、助手回复和响应回执
             |
             +----------------------------> 返回 ChatResponse
             `----------------------------> 后台记忆维护、画像更新
```

**意图识别负责路由和提供上下文，RAG 由 Agent 通过工具自主调用。** 讲题和作业辅导共享同一套教材知识库，靠元数据过滤区分学科与章节，不需要按意图各维护一套文档。

意图融合默认权重为 LLM 0.7、Embedding 0.2、Pattern 0.1。正常投票先累计同领域的粗、细意图分数，再选择领域内的意图；Pattern 细化要求领域一致。默认低置信度阈值为 0.5，另有识别失败和细化分支。权重与阈值定义在 [意图识别器](core/intent_recognizer.py)，不是 `.env` 配置项；应根据业务标注集验证后调整，不能将融合分数直接视为准确率。

<a id="quickstart"></a>

## 快速启动

### 1. 准备环境

- 安装并启动 Docker，以及 Docker Compose v2（命令为 `docker compose`）。
- 准备支持 **Anthropic Messages 协议及工具调用**的模型服务、模型名称和 API Key。
- 首次构建需要联网拉取镜像、安装依赖，并下载 ChromaDB 使用的本地向量模型。

以下命令在本仓库根目录执行。Windows 用户可用 Docker Desktop，接口示例可直接在 Swagger 中填写，避免终端引号和换行差异。

### 2. 创建 `.env`

首次配置时复制 [.env.example](.env.example) 为 `.env`：

```bash
cp .env.example .env
```

Windows PowerShell 可使用 `Copy-Item .env.example .env`。已有 `.env` 时直接编辑，保留原配置。

至少核对下面这些值：

```dotenv
# 示例为 DeepSeek 的 Anthropic 兼容入口。
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=替换为服务商当前支持的模型ID
ANTHROPIC_API_KEY=替换为该模型服务的Key

# 本地演示模式
EDUMIND_ENV=development
REDIS_PASSWORD=edumind123

# Docker 网络内的 ChromaDB 地址
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# 可选的在线意图向量；没有硅基流动 Key 也能运行。
INTENT_EMBEDDING_PROVIDER=auto
SILICONFLOW_API_KEY=
```

模型名必须是所选平台实际提供、账号可调用的 ID；示例文件中的值不保证适用于所有账号。使用 Anthropic 官方服务时可留空 `ANTHROPIC_BASE_URL`，并填写对应的模型名和 Key。只有 OpenAI Chat Completions 接口的服务，不能直接填到这里使用。

连接 `api.deepseek.com` 时，公共 LLM 调用层会显式关闭默认思考模式，覆盖意图分类、Agent 与汇总回复、查询改写、重排、画像、摘要和评测。这样可避免思考占满短输出预算而没有生成 JSON 或正文；其他模型服务保持原有请求参数。回复真正达到输出上限时仍会标记为 `partial`，不会伪装成完整成功。

### 3. 启动后端及依赖

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f edumind
```

根目录 Compose 启动 EduMind API、Redis、ChromaDB、Prometheus 和 Nginx。**它不包含 Web 前端页面**，Nginx 在这里代理后端接口。

| 入口 | 地址 | 用途 |
|---|---|---|
| Swagger | `http://localhost:8000/docs` | 查看接口、填写请求并执行 |
| 健康检查 | `http://localhost:8000/health` | 确认应用已完成初始化 |
| API | `http://localhost:8000` | 后端接口地址，如 `/chat` |
| Nginx | `http://localhost/docs` | 经反向代理访问 Swagger |
| Prometheus | `http://localhost:9090` | 查询监控指标 |

健康检查成功后，在 Swagger 中执行一次 `POST /chat`，确认模型也能正常调用。`/health` 不验证模型 Key、额度或模型名称。

### 4. 使用配套 Web 前端（可选）

本机已有根目录 Compose 数据环境时，可以直接叠加 Web 配置，复用现有 Python、Redis 和 ChromaDB 数据：

```bash
docker compose -f docker-compose.yml -f docker-compose.web.yml up -d --build
```

此配置同时启动前端和 Java 后端，页面入口为 `http://localhost:5174` 或 `http://localhost`；Python API 保持 `http://localhost:8000`，Java API 为 `http://localhost:8080`。前端经同源路径代理请求，模型 Key 只传入后端。Java 使用本机既有数据卷 `edumindjava_app-data`；没有该卷时先执行 `docker volume create edumindjava_app-data`，也可通过 `EDUMIND_JAVA_DATA_VOLUME` 指定另一个已有卷。后续启动、重建和停止这套环境时均携带上述两个 `-f` 参数。

如果同时拥有前端和 Java 项目，按配套 Compose 的相对路径摆放：

```text
同一个父目录/
  EduMind/           Python 后端，.env 放在这里
  EduMindFrontend/   Web 前端及全栈 Compose
  EduMindJava/       Java 后端及其 .env
```

先按各项目说明配置环境，再在 `EduMindFrontend` 目录执行：

```bash
docker compose --env-file ../EduMind/.env up -d --build
```

该部署的前端默认入口为 `http://localhost:5174`，也可经网关 `http://localhost` 访问。Python 服务名为 `edumind-python`，通过 `env_file` 读取 `../EduMind/.env`；命令中的 `--env-file` 同时让 Compose 插值使用该文件里的 Redis 密码等值。

两套 Compose 会使用重叠的端口和容器名，选择一种运行。切换部署方式前停止原来的那套；两套 Compose 的数据卷也相互独立，不会自动搬迁知识库和会话数据。

<a id="configuration"></a>

## 配置说明

所有后端凭据放在 **EduMind 根目录 `.env`** 或服务端环境变量中，不放入前端 `VITE_*` 变量。

### 常用配置

| 配置项 | 用途与默认行为 |
|---|---|
| `ANTHROPIC_API_KEY` | 必填，聊天、意图 LLM、摘要和 Judge 等使用的模型服务 Key。 |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容接口地址；留空时使用官方服务。 |
| `ANTHROPIC_MODEL` | 当前服务支持的模型 ID，建议显式填写。 |
| `REDIS_PASSWORD` | Compose 创建 Redis 时使用的密码。 |
| `REDIS_URL` | 应用连接地址；根目录 Compose 根据 `REDIS_PASSWORD` 覆盖为容器网络地址。 |
| `CHROMA_HOST` / `CHROMA_PORT` | Docker 使用 `chromadb:8000`；宿主机开发使用 `localhost:8001`。 |
| `CHROMA_PERSIST_DIRECTORY` | development 无法连接远端 ChromaDB 时的本地存储路径；容器内为 `/app/data/chroma`。production 连接失败时中止启动。 |
| `EDUMIND_SEED_DEMO_DOCS` | development 空库是否自动导入示例政策，默认 `true`；production 始终禁止自动导入。 |
| `EDUMIND_INTENT_TIMEOUT_S` / `EDUMIND_LLM_TIMEOUT_S` / `EDUMIND_COMPOSER_TIMEOUT_S` | 意图识别、Agent 每次模型请求和汇总模型请求的超时，默认 8 / 15 / 8 秒；均受聊天 50 秒总预算限制。 |
| `PROFILE_UPDATE_MIN_INTERVAL_S` | 普通后台画像最短更新间隔，默认 60 秒；明确表达偏好时立即更新。设为 0 恢复逐轮更新。会话消息和业务字段仍每轮保存。 |
| `LOG_LEVEL` | 日志级别，默认 `INFO`。 |
| `INTENT_EMBEDDING_ENABLED` | 是否执行向量分支，默认 `true`；设为 `false` 后仅融合 LLM 和规则。 |
| `INTENT_EMBEDDING_PROVIDER` | `auto`、`local` 或 `siliconflow`，默认 `auto`，见下一节。 |
| `SILICONFLOW_API_KEY` | 可选，在线意图向量使用的独立 Key。 |
| `MCP_EXTERNAL_ENABLED` | 默认 `true`；设为 `false` 关闭外部 MCP 客户端。 |
| `MCP_SERVERS_CONFIG` | 外部 MCP 配置文件路径，默认 `config/mcp_servers.json`。 |
| `EDUMIND_SKILLS_DIR` | Skills 根目录；Compose 挂载为 `/app/skills`。 |
| `EDUMIND_SKILLS_MAX_PROMPT_CHARS` | Skills 注入字符预算，默认 5000。 |
| `EDUMIND_ENV` | 鉴权部署模式，默认 `development`；`production` 必须配置访问 Token。 |
| `EDUMIND_API_TOKENS` | 访问 Token 与用户、角色的 JSON 映射，见部署章节。 |

`APP_ENV` 与 `EDUMIND_ENV` 用途不同：前者控制 `python api/main.py` 的开发热重载，后者控制生产鉴权要求。Docker 镜像通过 Uvicorn 启动，监听地址和端口由镜像命令指定；修改 `API_HOST`、`API_PORT` 不会改变 Compose 的端口映射。

### 修改配置后如何生效

在根目录部署中：

```bash
# 只改 .env：重新创建应用容器，让它读入环境变量。
docker compose up -d --force-recreate edumind

# 改了应用代码或依赖：重新构建并启动。
docker compose up -d --build edumind
```

修改 Redis 密码时，需要同时更新 Redis 和应用，可执行 `docker compose up -d --force-recreate redis edumind`。只执行 `restart` 不会重新读取容器的 `.env`。

从前端目录部署时，使用对应的服务名和配置来源：

```bash
docker compose --env-file ../EduMind/.env up -d --build --force-recreate edumind-python
```

<a id="intent-embedding-setup"></a>

## 在线 Embedding 配置

项目有两处不同的向量用途：

| 用途 | 模型与存储 |
|---|---|
| 三路意图识别中的向量匹配 | 默认本地字符 n-gram，可配置硅基流动 `BAAI/bge-m3`。 |
| RAG 知识库及 ChromaDB 记忆检索 | 使用 ChromaDB 默认的本地 `all-MiniLM-L6-v2` 向量模型。 |

本节只配置**意图识别的向量分支**，不会替换 RAG 模型，也不需要重建知识库。硅基流动的免费模型范围、账号权限和限流以[官方价格页](https://siliconflow.cn/pricing)为准；项目固定使用非 Pro 的 `BAAI/bge-m3`，不会自动切换到 `Pro/BAAI/bge-m3`。聊天及评测的 LLM 费用由对应服务商计算。

### 1. 申请 Key 并填写配置

1. 在[硅基流动中国站控制台](https://cloud.siliconflow.cn/)注册或登录，按平台要求开通账号。
2. 进入 API 密钥管理，创建 Key，并确认账号可使用 `BAAI/bge-m3`。
3. 在 **EduMind/.env** 中填写：

```dotenv
INTENT_EMBEDDING_ENABLED=true
INTENT_EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_API_KEY=替换为你自己的硅基流动Key
```

| 模式 | 行为 |
|---|---|
| `auto` | 有 Key 时尝试在线匹配，无 Key 时使用本地 n-gram。 |
| `local` | 始终使用本地 n-gram。 |
| `siliconflow` | 指定在线提供方；缺少 Key 或调用失败时仍会本地降级。 |

地址固定为 `https://api.siliconflow.cn/v1/embeddings`，接口格式见[官方 Embedding 文档](https://api-docs.siliconflow.cn/docs/api/embeddings-post)。硅基流动 Key、聊天模型 Key、EduMind API 访问 Token 是三种不同凭据，不能互换。

### 2. 重新创建容器

在 EduMind 根目录执行：

```bash
docker compose up -d --force-recreate edumind
```

如果代码尚未构建过，加上 `--build`。从前端目录运行的用户，使用上一节中的 `edumind-python` 命令。

### 3. 验证实际调用

先查看 `GET /integrations/embedding`：

```json
{
  "configured_provider": "auto",
  "online_configured": true,
  "model": "BAAI/bge-m3",
  "cooldown": false,
  "last_error": null
}
```

`online_configured=true` 只表示已读取 Key。使用独立检查接口验证模板向量和问题向量，不需要聊天模型余额：

```bash
curl -X POST http://localhost:8000/integrations/embedding/check \
  -H 'Content-Type: application/json' \
  -d '{"message":"这笔钱能退给我吗？"}'
```

只有实际使用在线模型时，结果中的 `success` 才为 `true`。`source` 显示实际向量来源，`status.last_dimensions` 显示维度；本地降级时，`status.fallback_reason` 区分 `missing_api_key`、`configured_local`、`http_401`、`http_429`、`TimeoutError` 等原因。`status.retry_after_s` 表示距离下一次允许在线重试的秒数。状态不包含 API Key。

本地脚本可独立检查，脚本会读取项目根目录 `.env`，环境中已设置的变量优先：

```bash
.venv/bin/python scripts/check_embedding.py --message '这笔钱能退给我吗？'
# 或在运行中的容器执行
docker compose exec -T edumind python scripts/check_embedding.py
```

本地脚本使用 Python 3.12 并安装项目依赖。在线检查成功退出码为 `0`，本地降级或分支关闭时为 `1`，便于在批量评测之前阻止误用本地向量。自己编写测试脚本时，同样需要在创建 `IntentRecognizer` 前加载 `.env`。

接着通过 `/chat` 发送一条新问题，例如“登录时提示 401，应该怎么处理？”，查看响应里的 `intent_embedding_source`：

- `siliconflow/BAAI/bge-m3`：本次匹配实际使用了在线向量。
- `local-ngram`：本次使用本地向量，可能是未配置、主动关闭或调用失败后降级。
- `disabled`：`INTENT_EMBEDDING_ENABLED=false`，没有执行向量分支；使用 LLM 0.85、规则 0.15 的权重。`provider=local` 仍执行向量分支，使用默认三路权重。
- `unavailable`：向量分支异常，未得到匹配结果。

启用鉴权后，状态接口需要管理员访问 Token，聊天接口可使用用户访问 Token。完整聊天还需要有效的 LLM 配置。

首次在线匹配会分批生成意图模板向量，每批最多 32 条，缓存在进程内；后续只需生成问题向量。在线匹配总预算为 8 秒，失败后冷却 60 秒，降级识别结果也会缓存 60 秒。降级时查询和模板一起切换，避免混用向量空间。接口发送当前问题和意图模板，不自动发送会话历史或用户画像。

<a id="usage"></a>

## 对话与知识库使用

以下示例按本地未启用鉴权的服务编写。启用后，HTTP 请求还需添加 `Authorization: Bearer <EduMind访问Token>`。

### 发送消息

在 Swagger 的 `POST /chat` 中填写：

```json
{
  "message": "数学导数这块不会，函数单调性怎么用导数判断？",
  "user_id": "demo_user",
  "client_request_id": "demo-derivative-001"
}
```

首次可不传 `conv_id`；后续对话携带响应中的 `conv_id`。每条新消息生成新的 `client_request_id`，同一消息因网络失败重试时复用原 ID。服务端回执保留 24 小时，同一 ID 携带不同消息返回 409。

响应中重点关注：

| 字段 | 如何使用 |
|---|---|
| `response`、`conv_id` | 展示回复、继续同一会话。 |
| `request_id` | 查询本次调用轨迹。 |
| `intent`、`intent_group`、`entities` | 查看细粒度意图、领域及提取实体。 |
| `intent_confidence`、`intent_source_scores` | 排查三路识别结果。 |
| `primary_agent`、`supporting_agents`、`routing_reason` | 了解主辅 Agent 的选择和路由依据。 |
| `tools_used`、`knowledge_used` | 查看工具使用情况；后者仅在知识库工具实际成功时为 true。 |
| `intent_embedding_source` | 确认本次向量分支使用在线模型还是本地 n-gram。 |
| `performance` | 本轮记忆读取、意图和 Agent 执行耗时，以及模型逻辑调用数、供应商返回的 token 用量和检索缓存事件；不包含问题正文。 |
| `status` | `success` 为完整回复，`partial` 表示回复截断或仅部分 Agent 成功，`failed` 为执行失败。 |
| `retryable`、`error_code` | 执行失败返回 HTTP 503；仅 `retryable=true` 时可以复用请求 ID 重新执行。 |
| `latency_ms` | 从进入聊天处理、等待会话锁到生成回复的耗时；完整 HTTP 耗时（含存储提交）见 `Server-Timing` 响应头。 |

检索会缓存“查询改写 → 召回 → 重排”的完整结果，并合并并发的相同检索。缓存按查询、Top-K、工具版本和传入的用户上下文隔离，沿用工具的 TTL；知识库更新会使线上及评测缓存失效。工具 TTL 为 0 时关闭完整流程缓存和并发合并，失败检索不缓存。`/search` 和知识库工具结果的 `cached` 表示完整流程命中，此时不再请求改写或重排模型。

默认回复提示约 200–400 个中文字符，多角色汇总约 300–500 个字符，保留必要依据、核验条件和下一步；用户明确要求详细说明时允许展开。这是生成时的篇幅指导，不是返回后裁剪，实际达到模型输出上限时仍返回 `partial`。结构化实体优先随角色输入包提供，避免在同一次模型请求中重复发送。

`performance.stages_ms` 是到生成回复为止的阶段耗时，嵌套阶段不可直接相加；`performance.llm.latency_ms` 是各次模型调用耗时之和，并行时可能大于实际墙钟时间。持久化提交在这些阶段之后，完整 HTTP 耗时仍以 `Server-Timing` 为准。幂等回放保留原响应中的统计。`/monitor` 的 `llm_usage.by_stage` 统计本进程各阶段用量，包括后台画像；进程重启后重新累计。调用数按 SDK 逻辑调用计算，不展开 SDK 内部重试；token 字段原样采用供应商统计口径，未报告的调用通过 `usage_reported_calls` 区分，不据此推算账单金额。

聊天总超时为 50 秒。同一会话串行处理，不同会话可并行；用户和助手消息一起写入 Redis。后台记忆压缩与画像更新不阻塞回复返回。

失败回执不会写入正常对话历史。可安全重试的失败在复用原 ID 时重新执行；成功和部分成功的回执直接返回。可能修改数据的工具执行前会保存不可重放标记，超时或连接中断后需要先核验业务结果，避免重复操作。自定义副作用工具必须将 `AgentToolSpec.retry_safe` 设置为 `false`；外部 MCP 工具按配置和只读声明确定。

学科、年级、章节、考试名称等槽位按会话保存，并在下一轮交给工具使用：换学科会清空上一门学科的章节，长期目标（如考研）则跨任务保留。字段来源可追踪，但这些槽位只表示"学生说过"，不代表已核验其身份或学习水平。

### 导入与检索知识库

在 `POST /knowledge/add` 中提交：

```json
{
  "documents": [
    {
      "document_id": "math-function-monotonic",
      "title": "函数单调性的定义与判定",
      "subject": "数学",
      "chapter": "函数",
      "difficulty": "基础",
      "grade": "高一",
      "doc_type": "知识点",
      "content": "设函数 f(x) 的定义域为 I，区间 D 是 I 的子区间。若对任意 x1 < x2 都有 f(x1) < f(x2)，则 f 在 D 上单调递增……"
    }
  ]
}
```

- `subject`、`chapter`、`difficulty`、`grade`、`doc_type`、`exam` 为可选结构化字段，会写入每个片段的 metadata，供按学科/章节过滤检索。
- 相同 `document_id` 表示更新同一文档；不传时按标题识别文档，同标题导入会覆盖。
- 同名不同来源的文档应指定不同 `document_id`。同批次相同身份保留最后一个版本。
- 文档会切片、向量化；更新会删除多余旧片段，并使相关检索缓存失效。接口返回的数量是**片段数**。
- `POST /knowledge/upload` 支持 txt、md、json，单文件最大 10 MB；JSON 格式可在 Swagger 查看。
- `GET /knowledge/stats` 查看片段总数与学科、文档类型分布。
- `POST /search` 使用查询参数 `query`、`top_k`，用于独立验证检索效果，不需要先经过 Agent 路由。
- `query` 最多 2000 字符，`top_k` 范围为 1–20；检索结果保留文档 ID、内容版本和片段位置。

先验证 `/search` 能召回文档，再通过 `/chat` 检查 Agent 是否按问题调用了知识库工具。

<a id="mcp"></a>

## MCP 外部数据接入

项目使用官方 `mcp==1.26.0` Python SDK，支持 Streamable HTTP 和 stdio 客户端。连接、工具发现及调用由 `tooling/mcp_client.py` 管理，工具执行继续经过缓存、超时、熔断和参数校验。

### 默认外部服务

[config/mcp_servers.json](config/mcp_servers.json) 配置了 Microsoft Learn 公开技术文档服务：

```json
{
  "servers": [
    {
      "name": "mslearn",
      "enabled": true,
      "transport": "streamable_http",
      "url": "https://learn.microsoft.com/api/mcp",
      "allowed_tools": ["microsoft_docs_search", "microsoft_docs_fetch"],
      "agents": ["general"],
      "read_only": true,
      "timeout_s": 25,
      "cache_ttl": 300,
      "max_result_chars": 12000
    }
  ]
}
```

General Agent 可以按需查询该服务；Tutor、Plan、Exam、Escalation 不获得这两个外部工具。服务说明见 [Microsoft Learn MCP](https://learn.microsoft.com/en-us/training/support/mcp)。

查看 `GET /integrations/mcp` 确认连接状态，再用 `POST /integrations/mcp/call` 独立测试工具：

```json
{
  "name": "mcp_mslearn_microsoft_docs_search",
  "arguments": {"query": "Azure Python authentication troubleshooting"}
}
```

此调试接口不经过 LLM。成功后可在聊天中提问，并用轨迹里的 `source=external_mcp`、`mcp_server`、`mcp_tool` 确认实际调用。

### 配置其他服务

- HTTP 服务填写 `name`、`url`、工具白名单和允许使用的 Agent。
- 需要认证时，通过 `headers_env` 将请求头映射到服务端环境变量，例如 `{"Authorization":"PRIVATE_MCP_AUTH_HEADER"}`；该变量的值填写完整的 `Bearer ...`。
- stdio 服务使用 `transport: "stdio"`、`command`、`args`，所需可执行程序必须存在于应用运行环境；附加环境变量通过 `env_vars` 显式配置。
- `read_only=true` 时，工具还必须声明只读且未声明破坏性操作；缺少相应声明的工具不会开放。
- 配置文件已只读挂载进容器。只修改 JSON 后可 `docker compose restart edumind`；同时修改环境变量时应重新创建容器。

外部服务只收到调用参数，不自动收到聊天历史、画像或模型 Key。连接失败会标记断开并重试，本地工具仍可使用。返回内容超出上限会标记 `truncated=true`，预览不能当成完整原文。

### 让其他 MCP 客户端访问 EduMind

连接地址为 **`http://localhost:8000/mcp/`**，保留末尾斜杠。当前提供：

- 工具：`search_local_knowledge(query, top_k)`。
- 资源：`edumind://knowledge/stats`。

这是标准 MCP 入口；`/integrations/mcp/*` 是外部客户端的 REST 调试接口。使用自定义域名时配置 `MCP_ALLOWED_HOSTS`，涉及浏览器来源时配置 `MCP_ALLOWED_ORIGINS`；启用接口鉴权后，MCP 客户端也需要发送访问 Token。

<a id="skills"></a>

## Skills 业务规则

内置学习服务台、学科辅导、学习计划、考试服务四类规则，位于 `skills/<名称>/SKILL.md`。修改业务处理步骤、升级条件或回复约束时，可直接编辑对应文件。

```bash
# 重新加载规则
curl -X POST http://localhost:8000/skills/reload

# 查看加载结果与解析错误
curl http://localhost:8000/skills
```

规则通过 Agent 类型和关键词匹配后注入，受字符预算限制；把关键约束放在前面。文件格式与字段说明见 [Skills 使用说明](skills/README.md)。Skills 用于处理规范，业务事实和产品资料更适合放入知识库。

<a id="evaluation"></a>

## 评测与监控

### 提交评测任务

`POST /eval/run` 返回 **HTTP 202 和 `job_id`**。不传请求体时使用内置用例，也可提供自己的用例：

```json
{
  "intent_cases": [
    {
      "message": "什么是函数的单调性？",
      "expected_intent": "knowledge_query"
    }
  ],
  "dialog_cases": [
    {
      "question": "导数这题我卡在第三步，能讲讲思路吗？",
      "reference_answer": "先确认题目与卡点，再给关键步骤和依据，不直接给可抄写的完整答案。",
      "evidence": "教学规范要求不代写作业；当前没有题目原文，无法给出具体答案。"
    }
  ]
}
```

通过 `GET /eval/jobs/{job_id}` 轮询：

```text
queued -> running -> completed
                  -> failed / cancelled
```

`completed`、`total` 字段表示进度；最终报告在 `result` 内。**任务执行完成不等于所有评分成功**，还要检查报告：

| 报告字段 | 含义 |
|---|---|
| `status` | `complete`、`partial`、`failed` 或 `empty`。 |
| `valid_total` / `evaluation_errors` | 有效用例数与评分执行失败数。 |
| `pass_rate` | 按有效用例计算；有效数为 0 时不能理解为正常的业务通过率。 |
| `results[].metadata.judge_error` | 评分失败原因；失败样本的 `scores` 为空，不以 0.5 代替失败。 |
| `suite_id` / `regression_note` | 回归比较使用的评测集合标识及跳过原因。 |

`intent_cases[].context.history` 可以提供历史消息；多轮对话用 `dialog_cases[].turns`。省略某一类用例时使用该类内置用例，明确传 `[]` 则跳过该类。字段结构以 Swagger 为准。

### Judge 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EVAL_JUDGE_MODEL` | 对话模型 | 仍使用相同的服务地址和模型 Key。 |
| `EVAL_JUDGE_MAX_TOKENS` | 4096 | 输出预算，截断重试时可加倍，最大 8192。 |
| `EVAL_JUDGE_TIMEOUT_S` | 45 | 每次评分请求超时，单位秒。 |
| `EVAL_BASELINE_PATH` | Compose 中为 `/app/data/eval/baseline.json` | 回归基线文件。 |

每条评分最多请求两次，401/403 等鉴权错误立即失败。解析支持代码围栏和前后说明文字，但缺字段、非法分值或不完整输出仍会记为失败。只有相同 `suite_id` 的完整评测参与基线比较；比较包含通过率，检测到回归的报告不会替换有效基线。

对话评分默认注入 5 个 Few-shot 校准示例，覆盖有依据的完整回答、诚实说明能力边界、笼统回答、编造操作成功及遗漏多轮已知信息。示例包含四维分数和评分依据，定义在 `evaluation/few_shot.py`；它们仅进入 Judge 提示词，不传给被测 Agent、不计入测试用例数量，也不新增模型请求。输入 token 会略有增加，Few-shot 用于校准评分尺度，不用于缩短评测时间。

每条对话结果记录 `judge_few_shot_version` 和 `judge_few_shot_count`。示例内容及版本纳入 `suite_id`，修改后首次完整评测会建立新的比较基线，不与旧评分标准直接比较。内置示例中的业务资料仅用于对应示例，当前用例仍以其参考答案、核验资料和对话历史为依据。

评测使用独立的 Orchestrator、统计、工具缓存和熔断状态，但共享真实知识库和外部服务，会消耗模型及外部接口额度。没有参考答案或证据的质量评分属于模型判断，应配合人工检查。

在线对话评测报告的 `scope=orchestrator`，覆盖编排、会话字段合并和工具调用；HTTP 鉴权、真实记忆生命周期及幂等回执由 API 回归和存储集成检查验证。内置对话用例提供参考答案和核验资料。自定义用例可设置 `expected_agent`、`expected_intent`、`required_tools`、`expected_entities`，对最后一轮进行确定性检查；执行失败、部分成功或检查未通过时，Judge 高分也不能使该轮通过。

评测 Agent 不获得可能修改数据的工具。生产部署的知识库和记忆服务连接失败时中止启动；新空库需要管理员导入实际业务文档。

### 在线监控与调用轨迹

| 接口 | 用途 |
|---|---|
| `GET /monitor` | 在线运行统计。 |
| `GET /metrics` | Prometheus 指标，抓取配置在 `config/prometheus.yml`。 |
| `GET /trace/tool/{request_id}` | 查看一次聊天的路由和工具轨迹。 |
| `GET /trace/tools?limit=20` | 查看最近的聊天轨迹。 |

监控降权使用最近 300 秒、最多 50 次已完成调用，至少 5 个样本参与计算。专业 Agent 达到降权条件后进入冷却，默认 60 秒；之后通过一次恢复探测决定恢复还是继续冷却。可用 `EDUMIND_MONITOR_COOLDOWN_SECONDS` 调整冷却时间。

聊天轨迹脱敏后存入 Redis，保留 7 天，最近列表最多 200 条；专业 Agent 失败回退时保留两次尝试的工具轨迹及回退事件。

`agent_latency_ms` 使用毫秒桶，`chat_latency_ms` 按 HTTP 状态码记录请求总耗时；将 `PROMETHEUS_PORT` 设为 0 可关闭独立指标端口，`/metrics` 仍提供业务指标。

<a id="development"></a>

## 本地开发与测试

推荐 Python 3.12，与 Docker 镜像一致。在项目根目录创建虚拟环境并安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 可使用 `py -3.12 -m venv .venv`，然后执行 `.venv\Scripts\Activate.ps1`。

只用 Docker 启动存储服务：

```bash
docker compose up -d redis chromadb
```

原生 Python 进程无法通过 Docker 服务名访问存储，在 `.env` 中改成宿主机地址：

```dotenv
REDIS_URL=redis://:edumind123@localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

Redis URL 中的密码应与实际密码一致。回到 Docker 部署时，把 ChromaDB 地址恢复为 `chromadb:8000`。

启动 API：

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

还可执行 `python api/main.py --cli` 进行终端交互。CLI 用于直接调试编排和工具；HTTP 鉴权、请求回执、异步评测等接口行为应通过 API 验证。

### 执行检查

```bash
# 测试依赖单独安装，不需要加到生产镜像中。
python -m pip install -r requirements-dev.txt
python -m pytest -q

# 真实 Redis / ChromaDB 的集成检查，在已启动的根目录部署中执行。
docker compose exec -T edumind python tests/verify_request_concurrency.py
docker compose exec -T edumind python tests/verify_chat_atomic.py
```

集成检查使用临时测试数据并清理。单元测试通过不代表模型平台鉴权成功，部署后仍需验证真实 `/chat`；在线 Embedding 和外部 MCP 分别按前文确认调用来源和工具结果。

<a id="deployment"></a>

## 部署与数据管理

### API 访问鉴权

本地演示可使用 `EDUMIND_ENV=development` 并留空 `EDUMIND_API_TOKENS`。对外部署设置：

```dotenv
EDUMIND_ENV=production
EDUMIND_API_TOKENS='{"替换为至少24字符的随机用户Token":{"user_id":"u1","role":"user"},"替换为至少24字符的随机管理员Token":{"user_id":"admin","role":"admin"}}'
```

用独立随机值替换示例中的 Token，可通过 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成。生产模式缺少或配置无效时启动失败。客户端发送：

```http
Authorization: Bearer <EduMind访问Token>
```

- 普通用户可访问聊天、共享知识查询、Skills 列表和本地 MCP；聊天身份来自 Token 中的 `user_id`。
- 管理员可访问评测、轨迹、知识库写入、监控、集成调试等接口。
- `/health` 无需认证；启用鉴权后不公开 Agent 明细。
- 配套前端在连接设置填写 Python API 访问 Token，仅保存在当前页面内存，刷新后需要重新填写。

启用鉴权后，直接打开 `/docs` 也需要认证头。可用支持设置请求头的 HTTP 客户端测试接口；不要把模型 Key 当作 API 访问 Token。

根目录 Compose 的端口绑定 `127.0.0.1`。对外访问应通过配置 HTTPS 的反向代理；Prometheus 抓取 `/metrics` 需要管理员凭据，可在抓取配置中通过 `authorization.type: Bearer` 和只读挂载的 `authorization.credentials_file` 提供。

### 数据保存位置

| 数据 | 根目录 Compose 中的位置 |
|---|---|
| 近期消息、摘要、回执、聊天轨迹 | Redis 命名卷 `redis-data`。 |
| 知识库、情景记忆和画像 | ChromaDB 命名卷 `chromadb-data`。 |
| ChromaDB 本地降级数据 | 项目 `data/chroma/` 挂载到 `/app/data/chroma`。 |
| 评测基线 | 项目 `data/eval/`。 |
| Skills 与 MCP 配置 | 项目 `skills/` 和 `config/mcp_servers.json`。 |
| Prometheus 时序数据 | 命名卷 `prometheus-data`。 |

常用管理命令：

```bash
docker compose ps
docker compose logs --tail=100 edumind
docker compose down
```

`down` 停止并移除容器，保留命名卷；`down -v` 会删除命名卷及其中数据，不能当作普通重启命令。备份应覆盖实际使用的数据卷和项目挂载目录；远端 ChromaDB 不可达时的本地降级数据不会自动合并回远端。

### 当前运行边界

- 推荐单应用进程部署。评测任务、路由状态和工具缓存都在进程内，不支持多 worker 共享。
- 评测最多同时保留 8 个未完成任务，串行执行，单任务执行上限 1800 秒，保留最近 50 个任务。应用重启后任务状态丢失，需要重新提交。
- 知识库写入锁是进程内锁，多副本同时导入需要另行协调；ChromaDB 的更新和删除不是跨进程原子事务。
- 重试回执避免已完成聊天重复写入，但无法保证进程崩溃前后的模型调用恰好执行一次。
- 当前知识库是共享库，用户会话隔离不等于知识库按租户隔离。接入真实业务数据时，应按实际访问范围扩展权限模型。

<a id="faq"></a>

## 常见问题

| 现象 | 检查与处理 |
|---|---|
| 容器构建失败或长时间停留在下载 | 查看构建日志，区分镜像拉取、Python 依赖安装和 ONNX 模型下载；首次构建需要这些网络资源。 |
| 健康检查正常，但聊天失败 | 查看应用日志和模型服务响应，核对 Key、Anthropic 兼容地址、模型 ID、额度及工具调用支持。 |
| 修改 `.env` 后不生效 | 重新创建应用容器；`restart` 只重启原容器。确认编辑的是 Python 项目的 `.env`。 |
| 前端选择 Python 后不可用 | 在前端目录查看 `docker compose logs --tail=100 edumind-python`，检查 Python 健康接口及连接设置；按前文带 `--env-file` 的命令启动。 |
| Redis 认证失败 | 检查 Redis 启动密码与应用 `REDIS_URL` 是否一致。改变密码后同时重新创建 Redis 和应用。 |
| Embedding 的 `online_configured=false` | 检查 Key 是否为空、模式是否为 `local`、容器是否重新创建。 |
| Embedding 返回 `http_401` / `http_403` | 检查是否使用硅基流动 Key，以及账号对当前模型的权限。 |
| Embedding 返回 `http_429`、超时或 `ValueError` | 分别检查限流、网络、响应向量格式；服务会本地降级。等待冷却和缓存过期后，用新问题验证。 |
| 知识库导入成功但召回异常 | 先检查片段数并独立调用 `/search`；核对文档内容、文档身份及实际连接的 ChromaDB 地址。 |
| MCP 工具没有出现 | 查看 `/integrations/mcp`、网络和白名单；核对 Agent 范围及只读声明。修改配置后重启应用。 |
| 评测只有任务 ID，没有分数 | 使用 `/eval/jobs/{job_id}` 轮询，在任务完成后读取 `result`。 |
| 评测失败较多 | 查看 `judge_error_type`、`judge_stop_reason`、`judge_attempts`；区分鉴权、超时、截断、JSON 格式和评分字段错误。 |
| 评测任务返回 404 / 提交返回 429 | 分别表示任务不存在或已过期、未完成队列已满；重启会清空任务状态，避免重复提交堆积。 |
| API 返回 401 / 403 | 检查 EduMind 访问 Token 和角色；管理员接口不能使用普通用户 Token。 |
| 端口或容器名冲突 | 检查是否同时启动了根目录和前端目录两套 Compose，停止不使用的一套。 |

<a id="documentation"></a>

## 项目结构与详细文档

```text
api/main.py                    HTTP API、应用初始化、CLI 入口
agents/agent_orchestrator.py    Agent 执行、主辅路由和回复合成
agents/tools.py                Agent 工具封装
core/intent_recognizer.py      三路意图识别与领域融合
core/intent_embedding.py       在线向量、本地降级与模板缓存
core/skill_loader.py           Skills 加载和匹配
core/access.py                 API 身份与角色校验
memory/conversation_memory.py  Redis / ChromaDB 分层记忆
tooling/                       工具执行与标准 MCP
  tool_manager.py              缓存、超时、熔断、参数校验
  knowledge_base.py            知识库导入、切片与检索
  mcp_client.py                外部 MCP 客户端
monitor/performance_monitor.py 运行统计与 Prometheus 指标
evaluation/evaluator.py        意图评测、Judge 与回归基线
evaluation/few_shot.py         Judge 评分校准示例
evaluation/jobs.py             异步评测任务队列
config/                        MCP、Nginx、Prometheus 配置
skills/                        业务处理规范
tests/                         单元测试与集成检查
wiki/                          使用、架构与代码文档
.env.example                   环境变量模板
docker-compose.yml             后端及依赖部署
```

| 文档 | 适合阅读的内容 |
|---|---|
| [01—12 源码教程](wiki/教程/README.md) | 从启动、请求链路到工具、知识库、记忆、监控与评测，附动手实验。 |
| [完整使用指南](wiki/教育培训完整使用指南.md) | 功能操作与使用细节。 |
| [业务流程说明](wiki/教育培训业务流程说明.md) | 一次请求从接收到响应的处理过程。 |
| [技术亮点](wiki/教育培训技术亮点.md) | 设计选择、实现方式和适用边界。 |
| [重点代码](wiki/教育培训重点代码.md) | 核心模块的代码阅读说明。 |
| [Skills 使用说明](skills/README.md) | 自定义业务规则的格式与加载方式。 |
| [场景扩展与改造指南](wiki/教育培训场景扩展与改造指南.md) | 记录从电商客服改造成教育培训场景的完整路径，以及其他可选场景。 |
| [文档中心](wiki/教育培训文档中心.md) | 其他学习和项目资料入口。 |

具体日期的改动记录见项目根目录的更新说明；本 README 介绍当前项目能力、配置和使用方式。
