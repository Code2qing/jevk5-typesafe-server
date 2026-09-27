# JevK5 TypeSafe Server

JevK5 TypeSafe Server 是一个独立的 HTTP 服务项目。它使用 [`jevk5`](https://github.com/allebee/jevk5) 官方提供的 `JevK5GGUF` 模块封装对 `llama-server` 的底层调用，对外提供与 [TypeSafe AI System One API](https://docs.typesafe.ai/api) 严格对齐、格式完全一致的 REST API。

无论是运行在 NVIDIA / AMD / Intel GPU、Mac (Apple Silicon) 还是普通 CPU 上，只要通过 `llama.cpp` 的 `llama-server` 加载 JevK5 GGUF 权重，即可无缝作为 TypeSafe API 的开源本地替代服务运行。

---

## 核心特性

- **严格兼容 TypeSafe API**：请求格式与响应格式与 [https://docs.typesafe.ai/api](https://docs.typesafe.ai/api) 保持 100% 一致。
- **使用 `JevK5GGUF` 模块**：直接依赖并使用 `from jevk5 import JevK5GGUF` 进行 token 级别的 logprob 提取与多选项处理。
- **完整支持三种决策类型**：
  - `noul`（Yes / No 概率决策，返回 `{"type": "noul", "noul": <float>}`）
  - `choice`（单选与概率分布，返回 `{"type": "choice", "choice": <key>, "probabilities": {...}, "confidence": <float>}`）
  - `score`（打分与评分层级，返回 `{"type": "score", "score": <float>, "legend": {...}, "probabilities": {...}, "confidence": <float>}`）
- **精确对齐 TypeSafe 置信度计算**：实现官方文档标准置信度公式 $\text{confidence} = \frac{N \times \max(P) - 1}{N - 1}$（单选与打分返回 `confidence`，`noul` 不附带）。
- **多选项自动锦标赛**：超过 16 个选项（最高 255 个）由 `JevK5GGUF` 自动执行淘汰制（Knockout）多轮 forward pass 并结合 `knockout_temperature` 校准。
- **开箱即用的生产特性**：
  - 异步非阻塞执行（支持高并发）
  - 支持可选的 Bearer API Key 鉴权
  - 具备 `/health` 健康检查与 `/v1/models` 模型列表端点
  - 标准错误状态码映射（401 Unauthorized, 422 Unprocessable Entity, 502 Bad Gateway）

---

## 架构示意

```
  Client (TypeSafe SDK / curl)
             │
             │ HTTP POST /v1/systemone
             ▼
┌──────────────────────────────────────────────┐
│          JevK5 TypeSafe Server               │
│                                              │
│   FastAPI Handler & Pydantic Schema          │
│                  │                           │
│                  ▼                           │
│      JevK5TypeSafeAdapter                    │
│                  │                           │
│                  ▼                           │
│     from jevk5 import JevK5GGUF              │
└──────────────────┬───────────────────────────┘
                   │ /tokenize & /completion
                   ▼
┌──────────────────────────────────────────────┐
│          llama-server (llama.cpp)            │
│         Model: JevK5-GGUF (Q8_0, etc.)       │
└──────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 启动 llama-server

参考 JevK5 Readme “Run it on any GPU, a Mac, or a CPU” 章节，使用 `llama-server` 启动 JevK5 GGUF 模型：

```bash
# 自动从 HuggingFace Hub 下载并加载 jevk5-4b-v0.3-Q8_0.gguf
llama-server --hf-repo alibiserikbay/JevK5-GGUF \
             --hf-file jevk5-4b-v0.3-Q8_0.gguf \
             -c 8192 \
             -ngl 99 \
             --port 8080
```

> **各模型推荐温度配置表**：
> | 模型 GGUF 文件 | `temperature` | `knockout_temperature` |
> |---|---|---|
> | `jevk5-4b-v0.3-Q8_0.gguf` | 1.22 | 0.93 |
> | `jevk5-4b-v0.3-Q5_K_M.gguf` | 1.22 | 0.93 |
> | `jevk5-9b-v0.3-Q8_0.gguf` | 1.049 | 1.2 |
> | `jevk5-2b-v0.2-Q8_0.gguf` | 1.42 | 0.77 |

### 2. 使用 uv 安装与运行 Server

```bash
cd jevk5-typesafe-server

# 使用 uv 同步依赖环境（秒级安装，自动排除重型 torch/cuda 依赖）
uv sync

# 启动服务（默认监听 0.0.0.0:8000，连接本地 8080 的 llama-server）
uv run python run.py --llama-url http://127.0.0.1:8080 --port 8000
```

#### 启动参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 绑定的 IP 地址 |
| `--port` | `8000` | 监听端口 |
| `--llama-url` | `http://127.0.0.1:8080` | llama-server 地址 |
| `--model` | `jevk5-4b-v0.3-Q8_0` | 对外宣告的模型标识 |
| `--temperature` | `1.22` | 基础概率校准温度（<=16 选项） |
| `--knockout-temperature` | `0.93` | 多选项淘汰制校准温度（>16 选项） |
| `--top-k` | `40` | 向 llama-server 请求的 top logprobs 数量 |
| `--api-key` | `None` | 可选的 Bearer Token 鉴权密钥 |
| `--reload` | `False` | 是否开启热重载开发模式 |

也可以通过 `.env` 文件或环境变量配置：`LLAMA_SERVER_URL`、`MODEL_NAME`、`TEMPERATURE`、`KNOCKOUT_TEMPERATURE`、`API_KEY`、`PORT` 等（可直接复制 `.env.example` 为 `.env`）。

配置优先级为：**命令行显式传参 > 环境变量 > .env 文件 > 代码内置默认值**。

### 3. 使用 Docker Compose 一键启动

项目内置了完整的 Docker Compose 编排方案，包含 `llama-server`（自动从 HuggingFace 拉取 GGUF 权重）与 `jevk5-typesafe-server` 双容器联动：

```bash
# 1. 复制配置文件与 Docker Compose 模板
cp .env.example .env
cp docker-compose.example.yml docker-compose.yml

# 2. 一键启动后端 llama-server 和 API 服务（llama-server 默认包含 -ngl 99 GPU 全层卸载）
docker compose up -d

# 3. 查看运行日志
docker compose logs -f
```

- API 服务端口：`http://localhost:8000`（宿主机映射端口可通过 `HOST_PORT` 自定义，容器内固定监听 8000；自动加载 `.env` 中的全部配置如 `API_KEY`、`TEMPERATURE` 等）
- llama-server 后端端口：`http://localhost:8080`（支持通过 `.env` 中的 `HF_REPO`、`HF_FILE`、`NGL` 灵活调整模型与 GPU 层数）
- HuggingFace 模型权重自动持久化挂载在数据卷 `jevk5-huggingface-cache` 中，避免重复下载。

---

## API 调用示例

### 1. 评估请求 `POST /v1/systemone`

#### 示例请求

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {
          "true": "Explicitly time-sensitive",
          "false": "No urgency expressed"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

#### 示例响应

```json
{
  "model": "jev-latest",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.95
    },
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.88,
        "technical": 0.12,
        "sales": 0.0
      },
      "confidence": 0.82
    },
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {
        "0": "Calm",
        "1": "Frustrated",
        "2": "Very angry"
      },
      "probabilities": {
        "0": 0.0,
        "1": 0.95,
        "2": 0.05
      },
      "confidence": 0.925
    }
  },
  "usage": {
    "input_tokens": 929,
    "output_tokens": 0
  }
}
```

### 2. 模型列表 `GET /v1/models`

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "models": [
    {
      "name": "jev-latest",
      "description": "The most recent stable release of JevK5.",
      "release_date": "2026-09-01"
    },
    {
      "name": "jev-preview",
      "description": "Preview build for JevK5 decisions.",
      "release_date": "2026-09-24"
    },
    {
      "name": "jevk5-4b-v0.3-Q8_0",
      "description": "Active local GGUF model running on llama-server.",
      "release_date": "2026-09-24"
    }
  ]
}
```

### 3. 健康检查 `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "ok": true,
  "status": "healthy",
  "backend_llama_connected": true,
  "model": "jevk5-4b-v0.3-Q8_0"
}
```

---

## 自动化测试

项目遵循严格的测试驱动开发（TDD）规范：

```bash
uv run pytest -v
```

包含全覆盖的测试用例：
- `test_confidence.py`：对齐 TypeSafe 规范的置信度公式数学边界验证
- `test_schemas.py`：Pydantic 请求体与响应体序列化及多态验证
- `test_adapter.py`：`JevK5GGUF` 模块适配层与输出构造逻辑
- `test_api.py`：HTTP 端点完整性、Bearer 鉴权及错误状态码测试
