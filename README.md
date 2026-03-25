```markdown
# Web Agent Planner (智能工作流执行引擎)

## 📖 项目简介
本项目是一个基于 LLM (Kimi/Moonshot) 和 MCP (Model Context Protocol) 规范的智能工作流编排与执行系统。它能够接收用户的自然语言目标，结合外部工作流生成 API，通过 LLM 逐步决策并调用外部 MCP 工具（如 SQL 查询、图表生成等）来完成复杂任务，并通过 Server-Sent Events (SSE) 向前端流式推送实时执行进度和 Markdown 报告。

## ✨ 核心特性
- **LLM 智能驱动**: 深度集成 Kimi (Moonshot AI)，利用 Function Calling / Tools 规范进行精准的下一步决策。
- **原生 MCP 支持**: 支持同时连接多个 MCP 服务器（如 SQL 节点、图表节点），具备动态工具发现和容错 Fallback 机制。
- **动态工具注册**: 内置可扩展的 `ToolRegistry`，支持 SQL 查询、数据可视化 (Chart)、文件处理等多种工具类别。
- **SSE 实时流输出**: 提供完整的执行进度追踪，前端可通过 SSE 接口实时接收 `thinking`、`message`、`error` 和 `report` 等标准化事件。
- **状态与记忆管理**: 自动维护执行历史 (`ExecutionMemory`) 和共享数据存储 (`shared_storage`)，支持复杂多步任务的数据上下文传递。

## 📂 目录结构
```text
planner3/
├── planner.py              # 核心工作流编排器，负责 LLM 决策、上下文管理与步骤流转
├── enhanced_executor.py    # 增强版 MCP 执行器，管理多服务器连接与工具调用
├── tool_registry.py        # 动态工具注册表，处理不同类型(SQL/Chart等)的工具逻辑
├── fastapi_main.py         # FastAPI 服务入口，提供 SSE 流式接口及本地图表静态代理
├── main.py                 # 本地命令行测试入口
├── llm_setup.py            # LLM API 统一封装 (专为 Kimi 优化)
├── utils.py                # 工具函数，包含工作流结构化解析
└── planner_prompt_mcp.yaml # LLM 编排器提示词模板

```

## 🛠️ 环境依赖与配置

**环境要求**: Python 3.10+

**依赖安装**:

```bash
pip install fastapi uvicorn pydantic requests fastmcp PyYAML openai

```

**环境变量**:
必须配置以下环境变量以启动服务：

* `MOONSHOT_API_KEY`: Kimi API Key (必填)
* `WORKFLOW_GENERATOR_API_URL`: 工作流生成服务 URL (默认: `http://localhost:8150/generate_workflow`)
* `SQL_EXECUTOR_URL`: SQL MCP 服务地址 (默认: `http://127.0.0.1:8000/mcp/`)
* `CHART_SERVER_URL`: 图表 MCP 服务地址 (默认: `http://172.16.1.114:1122/mcp`)
* `USE_AGENT_RUNTIME`: 是否启用 Agent Runtime 编排模式 (`true/false`，默认关闭)
* `AGENT_RUNTIME_BACKEND`: Agent Runtime 后端 (`builtin` 或 `openai_agents_sdk`，默认 `builtin`)

## 🧠 Agent Runtime 模式（新）

当 `USE_AGENT_RUNTIME=true` 时，服务将切换到 `AgentRuntimeOrchestrator` 执行链路，提供：

- **handoffs**：manager 自动分配到 `sql_specialist` / `viz_specialist`
- **guardrails**：输入与动作输出校验
- **structured output**：`AgentAction` / `TraceEvent` 强类型结构
- **run hooks + tracing**：内置 trace 事件流和最终报告中 trace 附录

> 说明：若设置 `AGENT_RUNTIME_BACKEND=openai_agents_sdk` 但环境未安装对应 SDK，将自动回退到 `builtin_fallback`。

## 🚀 快速启动

### 1. 启动 Web 服务 (FastAPI + SSE)

适用于需要与前端对接的生产/开发环境：

```bash
# 确保已设置环境变量
export MOONSHOT_API_KEY="your_api_key_here"

# 启动服务
python -m uvicorn planner3.fastapi_main:app --host 0.0.0.0 --port 9000

```

### 2. 本地命令行调试

适用于快速测试 MCP 连接和 LLM 规划逻辑：

```bash
export MOONSHOT_API_KEY="your_api_key_here"
python planner3/main.py

```

## 📡 核心 API 接口说明

### `POST /execute_workflow_sse`

流式执行工作流任务并返回进度。

**请求体**:

```json
{
  "content": "请分析缺陷和屏幕的空间聚集性"
}

```

**SSE 返回事件流 (Event Types)**:

* `thinking`: 任务初始化与生成计划阶段。
* `message`: 步骤执行的中间状态、工具调用详情及局部结果（Markdown 格式）。
* `ok`: 单个步骤成功完成。
* `error`: 执行异常信息。
* `report`: 全部任务完成后的最终人类可读总结报告。
* `done`: 会话安全结束标记。

## 📊 图表 DSL 规范 (Chart Tools)

当执行图表渲染时，系统标准化使用以下 DSL 传递给 MCP 节点：

```json
{
  "mark": "bar|line|pie|scatter...",
  "encoding": {
    "x": {"field": "x轴字段名"},
    "y": {"field": "y轴字段名"}
  },
  "title": "图表标题",
  "data_from_step": "step1" // 从共享存储动态注入数据
}

```

```

```
