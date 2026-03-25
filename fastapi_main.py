import requests
import json
import sys
import os
import asyncio
import uuid
import time
import logging
from typing import Dict, AsyncGenerator, Any, Optional
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.planner import TaskExecutorPlanner
from planner.llm_setup import QuickLLMAPI
from planner.utils import get_workflow_from_api
from planner.agent_runtime import AgentRuntimeOrchestrator

WORKFLOW_GENERATOR_API_URL = os.getenv("WORKFLOW_GENERATOR_API_URL", "http://localhost:8150/generate_workflow")

# MCP服务器配置 - 支持多个服务器
MCP_SERVERS = [
    os.getenv("SQL_EXECUTOR_URL", "http://127.0.0.1:8000/mcp/"),  # SQL服务器
    os.getenv("CHART_SERVER_URL", "http://172.16.1.114:1122/mcp")  # 图表服务器
]

# 过滤掉空的URL
MCP_SERVERS = [url for url in MCP_SERVERS if url and url.strip()]

# 仅输出数据传输相关日志
DATAFLOW_ONLY = os.getenv('DATAFLOW_ONLY', '').strip().lower() in ['1', 'true', 'yes', 'y', 'on']
USE_AGENT_RUNTIME = os.getenv('USE_AGENT_RUNTIME', '').strip().lower() in ['1', 'true', 'yes', 'y', 'on']

app = FastAPI(
    title="Workflow Executor SSE API",
    description="Web Agent - LLM驱动的智能工具流执行系统",
    version="1.0.0"
)

# 全局 executor 实例
global_executor = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化MCP连接"""
    global global_executor
    
    print("🚀 [FastAPI] 正在启动应用...")
    print(f"📡 [FastAPI] 配置的MCP服务器: {MCP_SERVERS}")
    
    try:
        from planner.enhanced_executor import EnhancedMcpExecutor
        global_executor = EnhancedMcpExecutor(MCP_SERVERS)
        connected = await global_executor.connect()
        
        if connected:
            print("✅ [FastAPI] MCP执行器初始化成功")
        else:
            print("⚠️ [FastAPI] MCP执行器部分初始化失败，但fallback工具可用")
            # 不要设置为None，因为fallback模式仍然可用
            
    except Exception as e:
        print(f"❌ [FastAPI] 启动时初始化失败: {e}")
        global_executor = None

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时断开连接"""
    global global_executor
    
    if global_executor:
        print("🔌 [FastAPI] 正在断开MCP连接...")
        await global_executor.disconnect()
        global_executor = None

# 静态文件挂载（/charts）——用于外部渲染服务若落地到本地目录时访问
charts_dir = Path("charts").resolve()
charts_dir.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(charts_dir)), name="charts")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该配置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


 


# ========== 工具函数 ==========
 


async def run_workflow(user_content: str, progress_callback=None) -> str:
    """执行完整工作流，支持实时进度回调"""
    global global_executor
    start_time = time.time()
    
    # 检查全局executor是否可用
    if global_executor is None:
        if progress_callback:
            await progress_callback({
                "type": "error",
                "content": "MCP执行器未初始化，请重启应用",
                "status": "failed"
            })
        return "MCP执行器未初始化"

    # 发送初始化进度（数据流日志：只保留事件与载荷）
    if progress_callback:
        payload = {
            "type": "initialization",
            "content": "正在初始化执行环境...",
            "status": "running"
        }
        if DATAFLOW_ONLY:
            try:
                print("[DATAFLOW] SSE -> client", json.dumps({"event": "initialization", "payload_keys": list(payload.keys())}, ensure_ascii=False))
            except Exception:
                pass
        await progress_callback(payload)

    workflow = get_workflow_from_api(user_content)
    if not workflow:
        if progress_callback:
            await progress_callback({
                "type": "error",
                "content": "无法从API获取工作流",
                "status": "failed"
            })
        return "无法从 API 获取工作流"

    task_input = {"query": user_content, "workflow": workflow}
    
    # 发送工作流信息
    if progress_callback:
        payload = {
            "type": "workflow_info",
            "content": f"已生成工作流，共{len(workflow)}个步骤：{list(workflow.keys())}",
            "total_steps": len(workflow),
            "workflow": workflow,
            "status": "ready"
        }
        if DATAFLOW_ONLY:
            try:
                print("[DATAFLOW] SSE -> client", json.dumps({"event": "workflow_info", "payload_keys": list(payload.keys())}, ensure_ascii=False))
            except Exception:
                pass
        await progress_callback(payload)
    
    # 支持实时回调
    try:
        if USE_AGENT_RUNTIME:
            runtime = AgentRuntimeOrchestrator(
                enhanced_executor=global_executor,
                tool_registry=global_executor.tool_registry,
                dataflow_only=DATAFLOW_ONLY
            )
            if progress_callback:
                await progress_callback({
                    "type": "runtime_mode",
                    "content": "使用 Agent Runtime 模式执行（handoff/guardrail/trace）",
                    "runtime_mode": "agent_runtime",
                    "status": "running"
                })
            final_report = await runtime.run(user_content, workflow, progress_callback)
        else:
            # 使用原有planner路径，保持向后兼容
            llm_api = QuickLLMAPI().api
            planner = TaskExecutorPlanner(
                llm_api=llm_api,
                tool_registry=global_executor.tool_registry,
                enhanced_executor=global_executor
            )
            final_report = await planner.execute_workflow(task_input, progress_callback)
        
        # 发送完成信息
        if progress_callback:
            duration = time.time() - start_time
            await progress_callback({
                "type": "completion",
                "content": f"工作流执行完成，耗时{duration:.2f}秒",
                "duration": duration,
                "status": "completed"
            })
            
        return final_report
    finally:
        # 清理资源 (全局executor会在应用关闭时断开)
        pass


# ========== Markdown 日志工具 ==========
class MarkdownLogger:
    """用于记录所有SSE事件到Markdown文件"""
    
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_file = None
        
    def start_session(self, conversation_id: str) -> Path:
        """开始新的会话日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_{conversation_id}_{timestamp}.md"
        self.current_file = self.base_dir / filename
        
        with open(self.current_file, 'w', encoding='utf-8') as f:
            f.write(f"# 工作流执行日志\n\n")
            f.write(f"**会话ID**: {conversation_id}\n")
            f.write(f"**开始时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
        
        return self.current_file
    
    def log_event(self, event_type: str, data: dict):
        """记录事件到Markdown文件"""
        if not self.current_file:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(f"## [{timestamp}] {event_type.upper()}\n\n")
            
            if "content" in data:
                f.write(f"**内容**: {data['content']}\n\n")
            
            if "step" in data and data["step"]:
                f.write(f"**步骤**: {data['step']}\n")
            
            if "step_number" in data and data["step_number"]:
                f.write(f"**进度**: {data['step_number']}/{data.get('total_steps', '?')}\n")
            
            if "function_name" in data and data["function_name"]:
                f.write(f"**函数**: {data['function_name']}\n")
            
            if "execution_time" in data and data["execution_time"]:
                f.write(f"**执行时间**: {data['execution_time']:.2f}秒\n")
            
            if "memory_usage" in data and data["memory_usage"]:
                f.write(f"**内存使用**: \n```json\n{json.dumps(data['memory_usage'], ensure_ascii=False, indent=2)}\n```\n")
            
            if "workflow" in data:
                f.write("**工作流详情**: \n")
                for step_key, instruction in data["workflow"].items():
                    f.write(f"- **{step_key}**: {instruction}\n")
                f.write("\n")
            
            # 添加完整数据
            f.write("**完整数据**: \n```json\n")
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            
            f.write("---\n\n")
    
    def complete_session(self, duration: float):
        """完成会话日志"""
        if not self.current_file:
            return
            
        with open(self.current_file, 'a', encoding='utf-8') as f:
            f.write(f"## 会话完成\n\n")
            f.write(f"**总耗时**: {duration:.2f}秒\n")
            f.write(f"**完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        logger.info(f"Markdown日志已保存: {self.current_file}")

# 初始化Markdown日志器
markdown_logger = MarkdownLogger()

# ========== SSE 工具 ==========
def format_sse(id_str: str, event: str, data: dict) -> str:
    """格式化为 SSE 消息"""
    return f"id:{id_str}\nevent:{event}\ndata:{json.dumps(data, ensure_ascii=False)}\n\n"


# ========== Markdown 构造工具 ==========
def _md_header(text: str, level: int = 3) -> str:
    level = max(1, min(level, 6))
    return f"{'#' * level} {text}"


def _md_kv_lines(pairs: dict) -> str:
    lines = []
    for k, v in pairs.items():
        if v is None or v == "":
            continue
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def _md_codeblock(text: str, lang: str = "") -> str:
    try:
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
    except Exception:
        s = str(text)
    return f"```{lang}\n{s}\n```"


def _md_workflow(workflow: Dict[str, str]) -> str:
    if not workflow:
        return "_未生成任何步骤_"
    lines = [f"- **{k}**: {v}" for k, v in workflow.items()]
    return "\n".join(lines)


def _md_wrap_block(md: str) -> str:
    """为Markdown块添加前后空行，避免在前后内容相邻时丢失换行。"""
    s = md or ""
    s = s.strip()
    return f"\n\n{s}\n\n"


def _extract_human_summary(md_text: str) -> str:
    """从报告文本中提取人类可读的摘要部分，去除尾部的大段机器可读JSON等。
    规则：遇到首个代码块（```）或 ```json 之前的内容作为摘要返回。
    """
    if not isinstance(md_text, str):
        return str(md_text)
    idx = md_text.find("```json")
    if idx == -1:
        idx = md_text.find("```")
    if idx != -1:
        return md_text[:idx].rstrip()
    return md_text


def _truncate_result_for_display(result_obj: Any, max_rows: int = 5) -> Any:
    """生成用于前端展示的精简版结果：
    - 若存在 query_data/preview_data/rows，则仅展示前 max_rows 条
    - 添加 display_truncated/display_rows_shown/display_total_rows 标记
    不修改原对象，返回一个浅拷贝结构。
    """
    try:
        # 直接列表：仅保留前 N
        if isinstance(result_obj, list):
            if len(result_obj) > max_rows:
                return {
                    "preview": result_obj[:max_rows],
                    "display_truncated": True,
                    "display_rows_shown": max_rows,
                    "display_total_rows": len(result_obj)
                }
            return result_obj

        # 顶层字典
        if isinstance(result_obj, dict):
            display = dict(result_obj)
            # 处理 data 下的内容
            data = display.get("data")
            if isinstance(data, dict):
                data_copy = dict(data)
                # query_data 优先
                qd = data_copy.get("query_data")
                if isinstance(qd, list) and len(qd) > max_rows:
                    data_copy["query_data"] = qd[:max_rows]
                    data_copy["display_truncated"] = True
                    data_copy["display_rows_shown"] = max_rows
                    data_copy["display_total_rows"] = len(qd)
                else:
                    # preview_data 次之
                    pd = data_copy.get("preview_data")
                    if isinstance(pd, list) and len(pd) > max_rows:
                        data_copy["preview_data"] = pd[:max_rows]
                        data_copy["display_truncated"] = True
                        data_copy["display_rows_shown"] = max_rows
                        data_copy["display_total_rows"] = len(pd)
                    else:
                        # rows/columns 结构
                        rows = data_copy.get("rows")
                        if isinstance(rows, list) and len(rows) > max_rows:
                            data_copy["rows"] = rows[:max_rows]
                            data_copy["display_truncated"] = True
                            data_copy["display_rows_shown"] = max_rows
                            data_copy["display_total_rows"] = len(rows)
                display["data"] = data_copy
                return display

            # 顶层 query_data
            qd_top = display.get("query_data")
            if isinstance(qd_top, list) and len(qd_top) > max_rows:
                display["query_data"] = qd_top[:max_rows]
                display["display_truncated"] = True
                display["display_rows_shown"] = max_rows
                display["display_total_rows"] = len(qd_top)
            else:
                # 顶层 preview_data
                pd_top = display.get("preview_data")
                if isinstance(pd_top, list) and len(pd_top) > max_rows:
                    display["preview_data"] = pd_top[:max_rows]
                    display["display_truncated"] = True
                    display["display_rows_shown"] = max_rows
                    display["display_total_rows"] = len(pd_top)
            return display
    except Exception:
        pass
    return result_obj


class WorkflowRequest(BaseModel):
    content: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    mcp_profiles: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health_check() -> JSONResponse:
    """健康检查端点，检查各服务状态"""
    services = {
        "workflow_generator": "unknown"
    }
    
    # 检查工作流生成器
    try:
        response = requests.get(WORKFLOW_GENERATOR_API_URL.replace("/generate_workflow", "/health"), timeout=5)
        services["workflow_generator"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["workflow_generator"] = "unreachable"
    
    # 检查所有MCP服务器
    for i, mcp_url in enumerate(MCP_SERVERS):
        service_name = f"mcp_server_{i+1}"
        try:
            response = requests.get(mcp_url, timeout=5)
            services[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services[service_name] = "unreachable"
    
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    mcp_profiles = None
    if global_executor and hasattr(global_executor, "get_server_profiles"):
        try:
            mcp_profiles = global_executor.get_server_profiles()
        except Exception:
            mcp_profiles = None

    return JSONResponse(content=HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services,
        mcp_profiles=mcp_profiles
    ).dict())


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return JSONResponse(content={
        "message": "Web Agent API 正在运行",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "execute_workflow": "/execute_workflow_sse",
            "docs": "/docs"
        }
    })


@app.post("/execute_workflow_sse")
async def execute_workflow_sse(req: WorkflowRequest):
    """SSE 接口：流式返回任务执行过程，适配前端事件类型"""

    async def event_stream() -> AsyncGenerator[str, None]:
        conversation_id = str(uuid.uuid4())  # 会话 ID
        start_time = datetime.now()
        start_id = str(int(time.time() * 1000))

        # 初始化Markdown日志
        log_file = markdown_logger.start_session(conversation_id)
        
        # 创建队列来收集进度信息
        progress_queue = asyncio.Queue()

        # 1. 发送初始化事件 - 使用 "thinking" 类型
        init_md = (
            f"{_md_header('开始处理工作流', 3)}\n\n"
            f"- **会话ID**: `{conversation_id}`\n"
            f"- **开始时间**: {start_time.isoformat()}\n\n"
            f"{_md_header('用户查询', 4)}\n"
            f"{_md_codeblock(req.content, '')}"
        )
        init_data = {
            "conversationId": conversation_id,
            "timestamp": start_time.isoformat(),
            "event": "thinking",  # 前端事件类型
            "content": init_md  # 前端显示内容（Markdown）
        }
        if DATAFLOW_ONLY:
            try:
                print("[DATAFLOW] SSE -> client", json.dumps({
                    "event": "thinking",
                    "keys": list(init_data.keys())
                }, ensure_ascii=False))
            except Exception:
                pass
        yield format_sse(start_id, "thinking", init_data)
        markdown_logger.log_event("thinking", init_data)

        # 2. 创建回调函数用于收集进度信息
        async def progress_callback(step_info: dict):
            logger.info(f"[Step Progress] {step_info}")
            await progress_queue.put(step_info)

        # 3. 运行任务，支持实时回调
        try:
            if not DATAFLOW_ONLY:
                logger.info(f"开始执行工作流: {req.content}")
            
            # 获取工作流 - 使用 "thinking" 类型（Markdown 格式）
            thinking_data = {
                "conversationId": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "event": "thinking",
                "content": _md_wrap_block(f"{_md_header('正在生成工作流...', 3)}\n\n_请稍候，正在与工作流服务通信_")
            }
            if DATAFLOW_ONLY:
                try:
                    print("[DATAFLOW] SSE -> client", json.dumps({"event": "thinking", "keys": list(thinking_data.keys())}, ensure_ascii=False))
                except Exception:
                    pass
            yield format_sse(str(int(time.time() * 1000)), "thinking", thinking_data)
            markdown_logger.log_event("thinking", thinking_data)
            
            workflow = get_workflow_from_api(req.content)
            if not workflow:
                error_data = {
                    "conversationId": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "event": "error",
                    "content": f"{_md_header('工作流生成失败', 3)}\n\n- **原因**: 无法从 API 获取工作流（workflow_generation_failed）"
                }
                yield format_sse(str(int(time.time() * 1000)), "error", error_data)
                markdown_logger.log_event("error", error_data)
                return

            # 发送工作流信息 - 使用 "ok" 类型（轻量标记，表示准备就绪）
            workflow_data = {
                "conversationId": conversation_id,
                "event": "ok",
                "content": _md_wrap_block(f"{_md_header('工作流已就绪', 3)}\n\n- **步骤数**: {len(workflow)}")
            }
            if DATAFLOW_ONLY:
                try:
                    print("[DATAFLOW] SSE -> client", json.dumps({"event": "ok", "keys": list(workflow_data.keys())}, ensure_ascii=False))
                except Exception:
                    pass
            yield format_sse(str(int(time.time() * 1000)), "ok", workflow_data)

            # 详细工作流信息 - 使用 "message" 类型（Markdown 展示完整步骤）
            workflow_data = {
                "conversationId": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "event": "message",
                "content": _md_wrap_block(f"{_md_header('工作流已生成', 3)}\n\n- **步骤数**: {len(workflow)}\n\n{_md_header('步骤详情', 4)}\n{_md_workflow(workflow)}"),
                "workflow": workflow,
                "total_steps": len(workflow)
            }
            if DATAFLOW_ONLY:
                try:
                    print("[DATAFLOW] SSE -> client", json.dumps({"event": "message", "keys": list(workflow_data.keys())}, ensure_ascii=False))
                except Exception:
                    pass
            yield format_sse(str(int(time.time() * 1000)), "message", workflow_data)
            markdown_logger.log_event("ok", workflow_data)

            # 启动工作流执行和进度收集
            async def run_workflow_task():
                try:
                    return await run_workflow(req.content, progress_callback)
                except Exception as e:
                    logger.error(f"工作流执行错误: {str(e)}")
                    raise e

            # 并发执行工作流和发送进度
            workflow_task = asyncio.create_task(run_workflow_task())
            
            # 发送进度信息
            while not workflow_task.done():
                try:
                    step_info = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    # 构建内容信息，整合message和其他详细信息
                    step_message = step_info.get("message", "")
                    step_type = step_info.get("type", "info")
                    step_name = step_info.get("step", "")
                    step_number = step_info.get("step_number", 0)
                    total_steps = step_info.get("total_steps", 0)
                    status = step_info.get("status", "running")
                    details = step_info.get("details", {})
                    function_name = step_info.get("function", None)
                    execution_time = step_info.get("execution_time", None)
                    
                    # 构建 Markdown 详细内容
                    md_lines = []
                    title = step_name or step_type or '信息'
                    md_lines.append(_md_header(str(title), 4))
                    if step_message:
                        md_lines.append(step_message)
                    meta_pairs = {
                        "进度": f"{step_number}/{total_steps}" if step_number and total_steps else None,
                        "任务类型": step_info.get("task_type"),
                        "工具": step_info.get("used_tool") or function_name,
                        "执行时间": f"{execution_time:.2f}s" if execution_time else None,
                    }
                    meta_block = _md_kv_lines({k: v for k, v in meta_pairs.items() if v})
                    if meta_block:
                        md_lines.append(meta_block)
                    # 指令（如果有）
                    instruction = step_info.get("instruction")
                    if instruction:
                        md_lines.append(_md_header("指令", 5))
                        md_lines.append(_md_codeblock(instruction, ''))

                    # LLM 决策（如果有）
                    llm_reasoning = step_info.get("llm_reasoning")
                    if llm_reasoning and not DATAFLOW_ONLY:
                        md_lines.append(_md_header("LLM 决策", 5))
                        md_lines.append(llm_reasoning)

                    # 结果或详情（优先显示 result）
                    result_obj = step_info.get("result")
                    if result_obj is not None:
                        # 结果精简展示（前端显示只保留前5条）
                        result_display = _truncate_result_for_display(result_obj, max_rows=5)
                        md_lines.append(_md_header("结果", 5))
                        try:
                            md_lines.append(_md_codeblock(json.dumps(result_display, ensure_ascii=False, indent=2), 'json'))
                        except Exception:
                            md_lines.append(_md_codeblock(str(result_display), ''))
                        # LLM 总结（若存在）
                        try:
                            llm_summary = step_info.get("llm_summary")
                            if not llm_summary and isinstance(result_obj, dict):
                                llm_summary = result_obj.get("llm_summary")
                            if llm_summary:
                                md_lines.append(_md_header("LLM 总结", 5))
                                md_lines.append(_md_codeblock(llm_summary, ''))
                        except Exception:
                            pass
                        # 如果是图表结果，追加图片预览（Markdown 图片）
                        try:
                            data_block = result_display.get("data", {}) if isinstance(result_display, dict) else {}
                            chart_url = data_block.get("chart_url") or (data_block.get("structured_content", {}) or {}).get("url")
                            chart_title = data_block.get("title", "图表")
                            if chart_url:
                                md_lines.append(_md_header("图表预览", 5))
                                md_lines.append(f"![{chart_title}]({chart_url})")
                        except Exception:
                            pass
                    elif details:
                        md_lines.append(_md_header("详情", 5))
                        try:
                            md_lines.append(_md_codeblock(json.dumps(details, ensure_ascii=False, indent=2), 'json'))
                        except Exception:
                            md_lines.append(_md_codeblock(str(details), ''))

                    # 错误（如果有）
                    err = step_info.get("error")
                    if err:
                        md_lines.append(_md_header("错误", 5))
                        md_lines.append(_md_codeblock(err, ''))
                    content = _md_wrap_block("\n\n".join([x for x in md_lines if x]))
                    
                    progress_data = {
                        "conversationId": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "event": "message",
                        "type": step_type,
                        "step": step_name,
                        "step_number": step_number,
                        "total_steps": total_steps,
                        "status": status,
                        "content": content,
                        "message": content,
                        "instruction": step_info.get("instruction"),
                        "used_tool": step_info.get("used_tool"),
                        "task_type": step_info.get("task_type"),
                        "result": step_info.get("result"),
                        "function_name": function_name,
                        "execution_time": execution_time,
                        "memory_usage": step_info.get("memory_usage", {})
                    }
                    
                    # 根据步骤类型选择事件类型
                    event_type = "message"
                    
                    # 检查是否为画图任务
                    result_obj = step_info.get("result")
                    if result_obj and isinstance(result_obj, dict):
                        data_block = result_obj.get("data", {})
                        structured_content = data_block.get("structured_content", {}) if isinstance(data_block, dict) else {}
                        if structured_content.get("type") == "draw":
                            event_type = "draw"
                    
                    # 其他类型检查
                    if event_type == "message":  # 只有在未检测到draw时才检查其他类型
                        if step_info.get("type") == "image_generation":
                            event_type = "img"
                        elif step_info.get("type") == "report":
                            event_type = "report"
                        elif step_info.get("status") == "completed":
                            event_type = "ok"
                    
                    # 同步 JSON 中的事件类型，便于前端按 data.event 分发
                    progress_data["event"] = event_type
                    
                    # 如果是draw事件，添加DSL数据
                    if event_type == "draw" and result_obj and isinstance(result_obj, dict):
                        data_block = result_obj.get("data", {})
                        if isinstance(data_block, dict):
                            dsl_data = data_block.get("dsl", {})
                            structured_content = data_block.get("structured_content", {})
                            progress_data["dsl"] = dsl_data
                            progress_data["chart_title"] = structured_content.get("title", data_block.get("title", "图表"))
                            
                    
                    if DATAFLOW_ONLY:
                        try:
                            print("[DATAFLOW] SSE -> client", json.dumps({
                                "event": event_type,
                                "step": step_name,
                                "status": status,
                                "has_result": step_info.get("result") is not None
                            }, ensure_ascii=False))
                        except Exception:
                            pass
                    yield format_sse(str(int(time.time() * 1000)), event_type, progress_data)
                    markdown_logger.log_event(event_type, progress_data)
                    # 追加一条仅结果的消息，确保前端能看到结果主体
                    if event_type == "ok" and step_info.get("result") is not None:
                        try:
                            result_display = _truncate_result_for_display(step_info.get("result"), max_rows=5)
                            parts = [
                                _md_header("结果", 4),
                                _md_codeblock(json.dumps(result_display, ensure_ascii=False, indent=2), 'json')
                            ]
                            # 附带图表图片
                            try:
                                data_block = result_display.get("data", {}) if isinstance(result_display, dict) else {}
                                chart_url = data_block.get("chart_url") or (data_block.get("structured_content", {}) or {}).get("url")
                                chart_title = data_block.get("title", "图表")
                                if chart_url:
                                    parts.append(_md_header("图表预览", 5))
                                    parts.append(f"![{chart_title}]({chart_url})")
                            except Exception:
                                pass
                            result_only_md = _md_wrap_block("\n\n".join(parts))
                        except Exception:
                            result_only_md = _md_wrap_block(
                                _md_header("结果", 4) + "\n\n" + _md_codeblock(str(step_info.get("result")), '')
                            )
                        result_event = {
                            "conversationId": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "event": "message",
                            "type": "result",
                            "step": step_name,
                            "step_number": step_number,
                            "total_steps": total_steps,
                            "status": status,
                            "content": result_only_md,
                            "message": result_only_md
                        }
                        if DATAFLOW_ONLY:
                            try:
                                print("[DATAFLOW] SSE -> client", json.dumps({
                                    "event": "message",
                                    "type": "result",
                                    "step": step_name
                                }, ensure_ascii=False))
                            except Exception:
                                pass
                        yield format_sse(str(int(time.time() * 1000)), "message", result_event)
                        markdown_logger.log_event("message", result_event)
                except asyncio.TimeoutError:
                    continue

            # 获取最终结果
            result_text = await workflow_task
            
            # 4. 最终消息 - 使用 "report" 类型（Markdown 格式）
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 仅向前端发送人类摘要，避免外部自定义渲染卡片/props等冗长内容
            human_summary = _extract_human_summary(result_text)
            report_md = (
                f"{_md_header('执行完成', 3)}\n\n"
                f"- **耗时**: {duration:.2f}秒\n\n"
                f"{_md_header('报告', 4)}\n"
                f"{human_summary}"
            )
            report_data = {
                "conversationId": conversation_id,
                "timestamp": end_time.isoformat(),
                "event": "report",
                "content": report_md,
                "duration": duration
            }
            
            if DATAFLOW_ONLY:
                try:
                    print("[DATAFLOW] SSE -> client", json.dumps({"event": "report", "keys": list(report_data.keys())}, ensure_ascii=False))
                except Exception:
                    pass
            yield format_sse(str(int(time.time() * 1000)), "report", report_data)
            markdown_logger.log_event("report", report_data)

        except Exception as e:
            logger.error(f"工作流执行错误: {str(e)}")
            error_data = {
                "conversationId": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "event": "error",
                "content": f"执行错误: {str(e)} (错误类型: {type(e).__name__})"
            }
            yield format_sse(
                str(int(time.time() * 1000)),
                "error",
                error_data
            )
            markdown_logger.log_event("error", error_data)

        finally:
            # 完成会话日志
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            markdown_logger.complete_session(duration)

            # 5. 发送完成事件（Markdown 格式）
            done_data = {
                "conversationId": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "event": "done",
                "content": f"{_md_header('会话结束', 3)}\n\n_感谢使用 Web Agent_"
            }
            if DATAFLOW_ONLY:
                try:
                    print("[DATAFLOW] SSE -> client", json.dumps({"event": "done", "keys": list(done_data.keys())}, ensure_ascii=False))
                except Exception:
                    pass
            yield format_sse(str(int(time.time() * 1000)), "done", done_data)
            markdown_logger.log_event("done", done_data)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ===================== 命令行启动入口 =====================
if __name__ == "__main__":

    # 启动 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
