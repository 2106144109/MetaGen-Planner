import os
import json
import datetime
import importlib
from typing import Dict, Any, Optional, List, Callable, Awaitable

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    step_id: str
    specialist: str = Field(description="manager|sql_specialist|viz_specialist")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class TraceEvent(BaseModel):
    event_type: str
    timestamp: str
    step_id: Optional[str] = None
    specialist: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


ProgressCallback = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


class AgentRuntimeOrchestrator:
    """轻量 Agent Runtime 编排器：
    - handoffs: manager -> specialist
    - guardrails: 输入/输出校验
    - structured output: AgentAction / TraceEvent
    - hooks + tracing: 全链路trace事件
    """

    def __init__(self, enhanced_executor, tool_registry=None, dataflow_only: bool = False):
        self.enhanced_executor = enhanced_executor
        self.tool_registry = tool_registry
        self.dataflow_only = dataflow_only
        self.trace_events: List[TraceEvent] = []
        self.runtime_backend = self._detect_backend()

    def _detect_backend(self) -> str:
        preferred = os.getenv("AGENT_RUNTIME_BACKEND", "builtin").strip().lower()
        if preferred == "openai_agents_sdk":
            spec = importlib.util.find_spec("agents")
            if spec is not None:
                return "openai_agents_sdk"
            return "builtin_fallback"
        return "builtin"

    def _trace(self, event_type: str, step_id: Optional[str] = None, specialist: Optional[str] = None,
               payload: Optional[Dict[str, Any]] = None) -> TraceEvent:
        evt = TraceEvent(
            event_type=event_type,
            timestamp=datetime.datetime.now().isoformat(),
            step_id=step_id,
            specialist=specialist,
            payload=payload or {}
        )
        self.trace_events.append(evt)
        if self.dataflow_only:
            print("[DATAFLOW] TRACE", json.dumps(evt.model_dump(), ensure_ascii=False))
        return evt

    def _guardrail_validate_input(self, query: str, workflow: Dict[str, str]) -> Optional[str]:
        if not isinstance(query, str) or not query.strip():
            return "query不能为空"
        if not isinstance(workflow, dict) or not workflow:
            return "workflow不能为空"
        for k, v in workflow.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return "workflow格式错误：step_id和instruction必须为字符串"
        return None

    def _select_specialist(self, step_id: str, instruction: str) -> str:
        text = f"{step_id} {instruction}".lower()
        if any(x in text for x in ["图", "chart", "可视化", "line", "bar", "pie", "scatter"]):
            return "viz_specialist"
        return "sql_specialist"

    def _build_action(self, step_id: str, instruction: str, specialist: str, shared_storage: Dict[str, Any]) -> AgentAction:
        if specialist == "viz_specialist":
            prev_step = None
            for key in reversed(list(shared_storage.keys())):
                if key.endswith("_result"):
                    prev_step = key[:-7]
                    break
            tool_name = "generate_column_chart"
            arguments: Dict[str, Any] = {"title": f"{step_id} 可视化结果"}
            if prev_step:
                arguments["data_from_step"] = prev_step
            return AgentAction(
                step_id=step_id,
                specialist=specialist,
                tool_name=tool_name,
                arguments=arguments,
                rationale="检测到可视化步骤，handoff到viz_specialist并引用上游数据。"
            )

        return AgentAction(
            step_id=step_id,
            specialist=specialist,
            tool_name="sql_query",
            arguments={"natural_language": instruction},
            rationale="默认由sql_specialist执行结构化查询步骤。"
        )

    def _guardrail_validate_action(self, action: AgentAction) -> Optional[str]:
        if not action.tool_name:
            return "tool_name不能为空"
        if not isinstance(action.arguments, dict):
            return "arguments必须为对象"
        return None

    async def _emit(self, callback: ProgressCallback, payload: Dict[str, Any]):
        if callback:
            await callback(payload)

    async def run(self, query: str, workflow: Dict[str, str], progress_callback: ProgressCallback = None) -> str:
        err = self._guardrail_validate_input(query, workflow)
        if err:
            self._trace("guardrail.input.failed", payload={"error": err})
            await self._emit(progress_callback, {"type": "error", "message": err, "status": "failed"})
            return f"Agent Runtime 输入校验失败：{err}"

        self._trace("runtime.start", payload={"backend": self.runtime_backend, "steps": len(workflow)})
        await self._emit(progress_callback, {
            "type": "runtime_start",
            "status": "running",
            "runtime_backend": self.runtime_backend,
            "message": f"Agent Runtime 启动，backend={self.runtime_backend}"
        })

        shared_storage: Dict[str, Any] = {}
        completed = 0
        total = len(workflow)
        final_sections: List[str] = []

        for idx, (step_id, instruction) in enumerate(workflow.items(), 1):
            specialist = self._select_specialist(step_id, instruction)
            self._trace("handoff", step_id=step_id, specialist=specialist, payload={"instruction": instruction})
            await self._emit(progress_callback, {
                "type": "handoff",
                "step": step_id,
                "step_number": idx,
                "total_steps": total,
                "specialist": specialist,
                "message": f"manager -> {specialist}"
            })

            action = self._build_action(step_id, instruction, specialist, shared_storage)
            action_err = self._guardrail_validate_action(action)
            if action_err:
                self._trace("guardrail.output.failed", step_id=step_id, specialist=specialist, payload={"error": action_err})
                await self._emit(progress_callback, {
                    "type": "step_error",
                    "step": step_id,
                    "step_number": idx,
                    "total_steps": total,
                    "message": action_err,
                    "status": "failed"
                })
                continue

            self._trace("tool.start", step_id=step_id, specialist=specialist, payload=action.model_dump())
            await self._emit(progress_callback, {
                "type": "step_start",
                "step": step_id,
                "step_number": idx,
                "total_steps": total,
                "specialist": specialist,
                "message": f"调用工具: {action.tool_name}",
                "instruction": instruction
            })

            context = {"shared_storage": shared_storage, "progress": {"completed": completed, "total": total}}
            result = await self.enhanced_executor.execute(step_id, instruction, context, action.tool_name, action.arguments)

            if isinstance(result, dict) and result.get("success"):
                completed += 1
                shared_storage[f"{step_id}_result"] = result.get("data")
                self._trace("tool.success", step_id=step_id, specialist=specialist, payload={"tool": action.tool_name})
                await self._emit(progress_callback, {
                    "type": "step_complete",
                    "step": step_id,
                    "step_number": idx,
                    "total_steps": total,
                    "specialist": specialist,
                    "used_tool": action.tool_name,
                    "result": result,
                    "status": "completed"
                })
                final_sections.append(f"- ✅ **{step_id}** ({specialist}/{action.tool_name})")
            else:
                err_msg = (result or {}).get("error", "unknown error") if isinstance(result, dict) else str(result)
                self._trace("tool.failed", step_id=step_id, specialist=specialist, payload={"error": err_msg})
                await self._emit(progress_callback, {
                    "type": "step_error",
                    "step": step_id,
                    "step_number": idx,
                    "total_steps": total,
                    "specialist": specialist,
                    "used_tool": action.tool_name,
                    "error": err_msg,
                    "status": "failed"
                })
                final_sections.append(f"- ❌ **{step_id}** ({specialist}/{action.tool_name}): {err_msg}")

        self._trace("runtime.complete", payload={"completed": completed, "total": total})
        await self._emit(progress_callback, {
            "type": "runtime_complete",
            "status": "completed",
            "completed_steps": completed,
            "total_steps": total
        })

        trace_json = json.dumps([evt.model_dump() for evt in self.trace_events], ensure_ascii=False, indent=2)
        report = (
            "## Agent Runtime 执行报告\n\n"
            f"- backend: `{self.runtime_backend}`\n"
            f"- completed: `{completed}/{total}`\n\n"
            "### Steps\n"
            + "\n".join(final_sections)
            + "\n\n### Trace\n```json\n"
            + trace_json
            + "\n```"
        )
        return report
