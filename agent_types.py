from typing import Dict, Any, Optional, List, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    step_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class NextAction(BaseModel):
    reasoning: str = ""
    current_step: str = "unknown"
    step_status: Literal["pending", "executing", "completed", "failed", "unknown"] = "unknown"
    progress_report: Dict[str, Any] = Field(default_factory=dict)
    tool_call: Optional[ToolCall] = None


class StepResult(BaseModel):
    step_id: str
    success: bool
    task_type: str = "unknown"
    used_tool: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    data: Any = None
    llm_summary: Optional[str] = None


class FinalReport(BaseModel):
    query: str
    total_steps: int
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    shared_storage_keys: List[str] = Field(default_factory=list)
    step_results: List[StepResult] = Field(default_factory=list)
    summary: str = ""
