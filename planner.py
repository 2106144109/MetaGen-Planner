# ==========================================================
# 文件: planner.py (V15 - 简化MCP专用版)
# 职责: 基于LLM的智能工作流编排器，仅支持MCP原生工具调用
# 特点: 删除了所有传统executor分支，只保留MCP服务器工具调用流程
# ==========================================================
import json
import os
import yaml
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# 引入 llm_setup 模块
from .llm_setup import QuickLLMAPI, BaseLLMBackend
# 引入工具注册表

# (数据类定义保持不变)
@dataclass
class LLMDecision:
    reasoning: str
    current_step: str
    step_status: str
    progress_report: Dict[str, Any]
    tool_call: Optional[Dict[str, Any]] = None  # {step_id, tool_name, arguments}
    @classmethod
    def from_dict(cls, data: Dict):
        response_data = data.get('response', {})
        return cls(
            reasoning=response_data.get('reasoning', 'No reasoning provided.'),
            current_step=(
                (response_data.get('tool_call') or {}).get('step_id')
                or response_data.get('current_step', 'unknown')
            ),
            step_status=response_data.get('step_status', 'unknown'),
            progress_report=response_data.get('progress_report', {}),
            tool_call=response_data.get('tool_call')
        )

@dataclass
class Post:
    send_from: str
    send_to: str
    message: str
    attachments: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class Round:
    round_id: int
    posts: List[Post] = field(default_factory=list)
    status: str = "active"

@dataclass
class StepRecord:
    instruction: str = ""
    structured_content: Any = None
    status: str = "pending"  # pending|executing|completed|failed|skipped
    retries: int = 0

@dataclass
class ExecutionMemory:
    shared_storage: Dict[str, Any] = field(default_factory=dict)
    step_status: Dict[str, str] = field(default_factory=dict)
    step_retries: Dict[str, int] = field(default_factory=dict)
    rounds: List[Round] = field(default_factory=list)
    initial_plan: Dict[str, str] = field(default_factory=dict)
    step_records: Dict[str, StepRecord] = field(default_factory=dict)
    
    def get_progress(self) -> Dict[str, List[str]]:
        completed = [step for step, status in self.step_status.items() if status == "completed"]
        failed = [step for step, status in self.step_status.items() if status == "failed"]
        all_steps = list(self.initial_plan.keys())
        processed_steps = set(completed + failed)
        pending = [step for step in all_steps if step not in processed_steps]
        return { "completed_steps": completed, "pending_steps": pending, "failed_steps": failed }


class TaskExecutorPlanner:
    def __init__(self, llm_api: Optional[BaseLLMBackend] = None, max_retries=3, tool_registry=None, enhanced_executor=None, **kwargs):
        # 仅支持MCP原生工具调用模式
        self.llm_api = llm_api if llm_api is not None else QuickLLMAPI().api
        self.max_retries = max_retries
        self.memory = ExecutionMemory()
        self.round_counter = 0
        self.max_rounds = 20
        # 仅输出数据传输日志
        self.dataflow_only = self._is_truthy(os.getenv('DATAFLOW_ONLY', kwargs.get('dataflow_only', '')))
        self.instruction_template = self._load_prompt_template()
        # Chat tools 上下文缓存（用于在工具执行后回传 role:"tool"）
        self._chat_tools_ctx: Optional[Dict[str, Any]] = None
        # 工具注册表 - 用于动态获取工具信息
        self.tool_registry = tool_registry
        # 增强执行器 - 用于执行所有工具
        self.enhanced_executor = enhanced_executor

    def _load_prompt_template(self) -> str:
        """加载 MCP 原生提示词模板，若失败则回退固定模板。"""
        try:
            current_dir = Path(__file__).parent
            mcp_yaml_path = current_dir / "planner_prompt_mcp.yaml"
            if mcp_yaml_path.exists():
                with open(mcp_yaml_path, 'r', encoding='utf-8') as f:
                    prompt_config = yaml.safe_load(f)
                    template = prompt_config.get('instruction_template', '')
                    if not self.dataflow_only:
                        print("✅ [Planner] 使用原生工具调用提示词模板 planner_prompt_mcp.yaml")
                    return template
            if not self.dataflow_only:
                print("⚠️ [Planner] 未找到 planner_prompt_mcp.yaml，使用默认模板")
            return self._get_fallback_template()
        except Exception as e:
            if not self.dataflow_only:
                print(f"❌ [Planner] 加载提示词模板失败: {e}，使用默认模板")
            return self._get_fallback_template()

    def _get_fallback_template(self) -> str:
        """备用提示词模板"""
        return """
You are an expert AI Project Manager. Your task is to manage a workflow by analyzing a conversation history of instructions and their results. Based on this history, you will decide on the next action to take.

The user will provide the history, concluding with the next instruction to be considered. Your response must be a JSON object that either confirms the execution of this next step or provides a corrected plan.
"""

    # ----------------------- MCP Native Helpers -----------------------
    def _is_truthy(self, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        s = str(v or '').strip().lower()
        return s in ['1', 'true', 'yes', 'y', 'on']
    
    def _get_dynamic_tools_spec(self) -> List[Dict[str, Any]]:
        """动态获取所有可用工具的规格，供LLM使用"""
        if self.tool_registry is None:
            # 如果没有工具注册表，返回默认的SQL工具
            print("  ⚠️ [Planner] 工具注册表未初始化，使用默认SQL工具")
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "sql_query",
                        "description": "执行SQL（自然语言描述）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "natural_language": {"type": "string"}
                            },
                            "required": ["natural_language"]
                        }
                    }
                }
            ]
        
        # 从工具注册表获取所有工具规格
        tools_spec = self.tool_registry.get_all_tools_spec()
        
        if not self.dataflow_only:
            print(f"  🔧 [Planner] 动态构建工具规格，共 {len(tools_spec)} 个工具")
        
        return tools_spec

    # ----------------------- Guards & Utils -----------------------
    def _is_probable_sql_text(self, text: str) -> bool:
        """Heuristic check: determine whether a text looks like raw SQL.
        We treat as SQL if it contains common SQL keywords or patterns.
        """
        if not isinstance(text, str):
            return False
        s = text.strip()
        if not s:
            return False
        lowered = s.lower()
        keywords = [
            "select ", " from ", " where ", " join ", " group by ", " order by ",
            " insert ", " update ", " delete ", " create table", " drop table",
            " having ", " union ", " limit ", " offset ", " into ", " values "
        ]
        if any(k in lowered for k in keywords):
            return True
        # simple pattern: ends with semicolon and has typical SQL tokens
        if s.endswith(';') and ("select" in lowered or "insert" in lowered or "update" in lowered or "delete" in lowered):
            return True
        return False

    def _build_execution_history(self) -> str:
        """构建执行历史文本"""
        if not self.memory.rounds:
            return "No execution history yet. This is the first step."
        
        history_parts = []
        for i, round_obj in enumerate(self.memory.rounds, 1):
            planner_post = next((p for p in round_obj.posts if p.send_from == "Planner"), None)
            executor_post = next((p for p in round_obj.posts if p.send_from == "Executor"), None)
            
            if planner_post and executor_post:
                step_id = planner_post.attachments.get('step_id', f'step{i}')
                instruction = planner_post.attachments.get('instruction', planner_post.message)
                
                # 获取结构化内容
                rec = self.memory.step_records.get(step_id)
                if rec and rec.structured_content:
                    # 🔧 为图表生成提供详细的字段信息，不截断SQL结果
                    content = rec.structured_content
                    if isinstance(content, dict):
                        # 如果有query_data，显示字段信息和数据样本
                        query_data = content.get('query_data', [])
                        if isinstance(query_data, list) and query_data:
                            available_fields = list(query_data[0].keys()) if query_data[0] and isinstance(query_data[0], dict) else []
                            sample_data = query_data[:3]  # 显示前3条数据作为样本
                            result_summary = f"""SQL execution completed successfully.
- Available fields: {available_fields}
- Row count: {content.get('row_count', len(query_data))}
- Sample data: {json.dumps(sample_data, ensure_ascii=False)}"""
                        else:
                            result_summary = json.dumps(content, ensure_ascii=False)[:300] + "..."
                    else:
                        result_summary = json.dumps(content, ensure_ascii=False)[:300] + "..."
                else:
                    result_data = executor_post.attachments.get('result', {})
                    if result_data.get('success'):
                        result_summary = "Execution successful"
                    else:
                        result_summary = f"Execution failed: {result_data.get('error', 'Unknown error')}"
                
                history_parts.append(f"""
**Round {i} - {step_id}**
- Instruction: {instruction}
- Result: {result_summary}
- Status: {rec.status if rec else 'unknown'}
""")
        
        return "\n".join(history_parts)

    def _get_llm_decision(self, query: str, last_post: Post) -> LLMDecision:
        """
        使用YAML模板获取LLM决策（Chat tools 或 MCP responses 或 Chat JSON）
        """
        progress = self.memory.get_progress()
        pending_steps = progress.get("pending_steps", [])

        if not pending_steps:
            return LLMDecision.from_dict({"response": {"reasoning": "所有步骤已完成，任务结束。"}})

        # 获取下一步信息
        next_step_id = pending_steps[0]
        next_step_instruction = self.memory.initial_plan.get(next_step_id, "无此步骤的指令。")
        
        # 构建执行历史
        execution_history = self._build_execution_history()
        
        # 填充模板占位符
        formatted_prompt = self.instruction_template.format(
            query=query,
            workflow=json.dumps(self.memory.initial_plan, ensure_ascii=False, indent=2),
            history=execution_history
        )
        
        # 添加当前步骤信息 + MCP 工具规则
        final_prompt = f"""{formatted_prompt}

## Next Step to Execute
- **Step ID**: {next_step_id}
- **Instruction**: {next_step_instruction}

## MCP Tool Usage Rules
You MUST select exactly one tool and respond ONLY with a `tool_call` JSON.
The `sql_query` MCP tool接受自然语言描述；请提供精简参数。

Additional Requirements:
- 若需图表，先使用 `sql_query` 获取更新后的字段，再在下一轮使用图表类工具。
- 坐标轴字段必须取自最近成功SQL结果的键。

Please analyze the above context and provide your decision.
"""
        
        if not self.dataflow_only:
            print(f"  🤔 [Planner] 正在向 LLM 请求决策 (使用YAML模板)...")
            print(f"  - 对话历史深度: {len(self.memory.rounds)} 轮")
            print(f"  - 下一步: {next_step_id}")

        # 构建消息（先走 Chat tools 路径）
        messages = [ {"role": "user", "content": final_prompt} ]
        # 动态构建工具集：从ToolRegistry获取所有可用工具
        tools_spec = self._get_dynamic_tools_spec()

        # 仅使用Chat tools原生MCP调用
        try:
            full_resp = self.llm_api.chat_completion_full(messages, tools=tools_spec, stream=False)
            response_str = json.dumps(full_resp, ensure_ascii=False)
        except Exception as e:
            if not self.dataflow_only:
                print(f"❌ [Planner] MCP Chat tools 调用失败: {e}")
            return LLMDecision.from_dict({"response": {"reasoning": f"MCP工具调用失败: {e}"}})
        
        # 仅数据流日志：打印请求与响应摘要
        if self.dataflow_only:
            try:
                print("[DATAFLOW] LLM -> request", json.dumps({
                    "messages_len": len(messages),
                    "tools": [t.get('function', {}).get('name') for t in (tools_spec or [])],
                    "stream": False,
                    "step_id": next_step_id
                }, ensure_ascii=False))
            except Exception:
                print("[DATAFLOW] LLM -> request (summary)")
        else:
            pass
        
        try:
            # 解析LLM响应
            parsed = json.loads(response_str)
            choice0 = (parsed.get('choices') or [{}])[0]
            msg = choice0.get('message') or {}
            
            tool_call = None
            reasoning = 'MCP工具调用'
            
            # 优先检查 OpenAI function calling 格式
            tool_calls = msg.get('tool_calls') or []
            if tool_calls:
                tc = tool_calls[0]
                name = (tc.get('function') or {}).get('name')
                args_text = (tc.get('function') or {}).get('arguments') or '{}'
                try:
                    args = json.loads(args_text)
                except Exception:
                    args = { 'raw': args_text }
                
                tool_call = {
                    'step_id': next_step_id,
                    'tool_name': name,
                    'arguments': args,
                    'tool_call_id': tc.get('id')
                }
                reasoning = msg.get('content') or reasoning
                
                # 缓存 chat tools 上下文，便于执行后回传 role:"tool"
                self._chat_tools_ctx = {
                    'user_prompt': final_prompt,
                    'assistant_tool_call': tc,
                    'tools_spec': tools_spec
                }
            
            # 当 tool_calls 为空时，尝试从 message.content 解析 JSON
            else:
                content = msg.get('content', '')
                
                if content:
                    # 方法1: 提取 ```json ... ``` 格式
                    import re
                    json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                    
                    if json_match:
                        try:
                            json_content = json_match.group(1).strip()
                            content_data = json.loads(json_content)
                            response_data = content_data.get('response', {})
                            tool_call_data = response_data.get('tool_call')
                            reasoning = response_data.get('reasoning', reasoning)
                            
                            if tool_call_data and isinstance(tool_call_data, dict):
                                # 验证必需字段
                                if tool_call_data.get('tool_name'):
                                    tool_call = {
                                        'step_id': tool_call_data.get('step_id', next_step_id),
                                        'tool_name': tool_call_data.get('tool_name'),
                                        'arguments': tool_call_data.get('arguments', {}),
                                        'tool_call_id': None  # 文本格式没有call_id
                                    }
                                    
                        except (json.JSONDecodeError, KeyError) as e:
                            pass
                    
                    # 方法2: 尝试直接解析整个 content 作为 JSON (备用方案)
                    elif content.strip().startswith('{'):
                        try:
                            
                            content_data = json.loads(content.strip())
                            response_data = content_data.get('response', {})
                            tool_call_data = response_data.get('tool_call')
                            reasoning = response_data.get('reasoning', reasoning)
                            
                            if tool_call_data and isinstance(tool_call_data, dict) and tool_call_data.get('tool_name'):
                                tool_call = {
                                    'step_id': tool_call_data.get('step_id', next_step_id),
                                    'tool_name': tool_call_data.get('tool_name'),
                                    'arguments': tool_call_data.get('arguments', {}),
                                    'tool_call_id': None
                                }
                                
                                    
                        except (json.JSONDecodeError, KeyError) as e:
                            pass
            
            return LLMDecision(
                reasoning=reasoning,
                current_step=next_step_id,
                step_status='executing' if tool_call else 'unknown',
                progress_report=progress,
                tool_call=tool_call
            )
        except (json.JSONDecodeError, KeyError) as e:
            return LLMDecision.from_dict({"response": {"reasoning": "MCP响应解析失败"}})

    async def execute_workflow(self, task_input: Dict, progress_callback=None) -> str:
        query = task_input['query']
        workflow = task_input['workflow']
        self.memory.initial_plan = workflow
        total_steps = len(workflow)
        
        for step_id in workflow:
            self.memory.step_status[step_id] = "pending"
            if step_id not in self.memory.step_records:
                self.memory.step_records[step_id] = StepRecord(status="pending")

        current_post = Post("User", "Planner", query, {"result": {"success": True, "data": "任务开始"}})

        if progress_callback:
            await progress_callback({
                "type": "workflow_start",
                "message": f"开始执行工作流，共{total_steps}个步骤",
                "total_steps": total_steps,
                "status": "running"
            })

        completed_steps = 0
        
        while self.round_counter < self.max_rounds:
            self.round_counter += 1
            if not self.dataflow_only:
                print(f"\n--- 🔄 Round {self.round_counter} ---")
            
            round_obj = Round(round_id=self.round_counter)

            llm_decision = self._get_llm_decision(query, current_post)
            if self.dataflow_only:
                print("[DATAFLOW] LLM <- response", json.dumps({
                    "step_id": llm_decision.current_step,
                    "has_tool_call": bool(llm_decision.tool_call),
                    "tool_name": (llm_decision.tool_call or {}).get('tool_name')
                }, ensure_ascii=False))
            else:
                print(f"🧠 LLM决策: {llm_decision.reasoning}")

            has_tool_call = bool(llm_decision.tool_call and llm_decision.tool_call.get('step_id'))
            if not has_tool_call:
                if not self.dataflow_only:
                    print("✅ LLM决定任务结束。")
                if progress_callback:
                    await progress_callback({
                        "type": "workflow_complete",
                        "message": "所有步骤已完成",
                        "completed_steps": completed_steps,
                        "total_steps": total_steps,
                        "status": "completed"
                    })
                break

            selected_step_id = (llm_decision.tool_call or {}).get('step_id')
            # 归一化重试步骤ID：支持 step1_retry 与 step1_retry_1
            if isinstance(selected_step_id, str):
                if '_retry_' in selected_step_id:
                    original_step_id = selected_step_id.split('_retry_')[0]
                elif selected_step_id.endswith('_retry'):
                    original_step_id = selected_step_id[:-6]
                else:
                    original_step_id = selected_step_id
            else:
                original_step_id = selected_step_id
            current_step_number = list(workflow.keys()).index(original_step_id) + 1
            
            if original_step_id in self.memory.step_status:
                 self.memory.step_status[original_step_id] = "executing"

            # 记录实际执行的指令到 step_records
            rec = self.memory.step_records.get(original_step_id) or StepRecord()
            # 指令来源：优先工具调用参数，否则回退初始计划文本
            instruction_text = None
            tool_args = (llm_decision.tool_call or {}).get('arguments') or {}
            if isinstance(tool_args, dict):
                for k in ['natural_language', 'nl', 'query', 'instruction']:
                    if k in tool_args and isinstance(tool_args[k], str) and tool_args[k].strip():
                        instruction_text = tool_args[k].strip()
                        break
            if not instruction_text:
                instruction_text = self.memory.initial_plan.get(original_step_id, '')

            # 防护：若指令疑似为 SQL，则回退为初始工作流的中文自然语言描述
            if self._is_probable_sql_text(instruction_text):
                fallback_instruction = self.memory.initial_plan.get(original_step_id, '')
                if isinstance(fallback_instruction, str) and fallback_instruction.strip() and not self._is_probable_sql_text(fallback_instruction):
                    instruction_text = fallback_instruction.strip()

            rec.instruction = instruction_text
            rec.status = "executing"
            self.memory.step_records[original_step_id] = rec

            # 发送进度回调
            if progress_callback:
                await progress_callback({
                    "type": "step_start",
                    "step": original_step_id,
                    "step_number": current_step_number,
                    "total_steps": total_steps,
                    "message": f"开始执行步骤 {current_step_number}/{total_steps}: {workflow[original_step_id]}",
                    "instruction": instruction_text,
                    "status": "running",
                    "llm_reasoning": llm_decision.reasoning,
                    "llm_current_step": llm_decision.current_step,
                    "task_type": "step"
                })

            planner_attachments = {
                "step_id": original_step_id,
                "instruction": instruction_text,
                # 工具调用（若有）
                "tool_name": (llm_decision.tool_call or {}).get('tool_name'),
                "arguments": (llm_decision.tool_call or {}).get('arguments'),
            }
            planner_post_to_executor = Post(
                "Planner", "Executor", 
                f"请执行: {instruction_text}",
                planner_attachments
            )
            round_obj.posts.append(planner_post_to_executor)
            
            if self.dataflow_only:
                print("[DATAFLOW] Planner -> Executor", json.dumps({
                    "step_id": original_step_id,
                    "tool": planner_attachments.get('tool_name'),
                    "arguments": planner_attachments.get('arguments')
                }, ensure_ascii=False))
            else:
                print(f"🚀 Planner -> Executor: 执行步骤 '{original_step_id}'")
                print(f"📤 Planner发送给Executor的完整消息:")
                print(f"   Step ID: {original_step_id}")
                print(f"   Instruction: {instruction_text}")
                try:
                    print(f"   Full attachments: {json.dumps(planner_attachments, ensure_ascii=False, indent=2)}")
                except Exception:
                    print(f"   Full attachments: {planner_attachments}")
            
            # MCP 工具调用流程
            tool_name = (llm_decision.tool_call or {}).get('tool_name')
            arguments = (llm_decision.tool_call or {}).get('arguments') or {}
            tool_call_id = (llm_decision.tool_call or {}).get('tool_call_id')
            
            # 使用增强执行器执行所有工具，包括SQL和图表工具
            if self.enhanced_executor:
                # 构建执行上下文，包含共享存储
                context = {
                    'shared_storage': self.memory.shared_storage,
                    'step_records': self.memory.step_records,
                    'progress': self.memory.get_progress()
                }
                result_payload = await self.enhanced_executor.execute(
                    original_step_id, instruction_text, context, tool_name, arguments
                )
            else:
                # 回退到简化模式（仅支持SQL工具）
                result_payload = await self._execute_natively(original_step_id, instruction_text, tool_name, arguments)

            # 将工具结果回传给 LLM（role:"tool"），促使其给出总结/下一步（仅当存在 tool_call_id 时）
            followup_result = None
            try:
                if self._chat_tools_ctx and tool_call_id:
                    followup_result = await self._chat_tools_followup_loop(
                        original_step_id,
                        self._chat_tools_ctx['user_prompt'],
                        self._chat_tools_ctx['assistant_tool_call'],
                        self._chat_tools_ctx['tools_spec'],
                        tool_name,
                        tool_call_id,
                        result_payload.get('data') if isinstance(result_payload, dict) else result_payload,
                        max_rounds=3
                    )
                    # 提取最终总结
                    try:
                        choice0 = (followup_result.get('choices') or [{}])[0]
                        msg2 = choice0.get('message') or {}
                        llm_text = msg2.get('content')
                        if llm_text:
                            result_payload['llm_summary'] = llm_text
                    except Exception:
                        pass
            except Exception as e:
                if not self.dataflow_only:
                    print(f"⚠️ [Planner] 回传工具结果给LLM失败: {e}")

            # 附带精简后的 followup 片段，便于调试
            slim_followup = None
            try:
                if isinstance(followup_result, dict):
                    slim = {
                        'finish_reason': ((followup_result.get('choices') or [{}])[0] or {}).get('finish_reason'),
                        'message_preview': (((followup_result.get('choices') or [{}])[0] or {}).get('message') or {}).get('content')
                    }
                    slim_followup = slim
            except Exception:
                slim_followup = None

            executor_feedback_post = Post("Executor", "Planner", f"步骤 '{original_step_id}' 执行完毕。", {"step_id": original_step_id, "result": result_payload, "llm_followup": slim_followup})

            # 将本轮的 Executor 反馈加入当前回合
            round_obj.posts.append(executor_feedback_post)

            if self.dataflow_only:
                try:
                    print("[DATAFLOW] Executor -> Planner", json.dumps({
                        "step_id": executor_feedback_post.attachments.get('step_id'),
                        "success": executor_feedback_post.attachments.get('result', {}).get('success'),
                        "used_tool": executor_feedback_post.attachments.get('result', {}).get('used_tool')
                    }, ensure_ascii=False))
                except Exception:
                    print("[DATAFLOW] Executor -> Planner (summary)")
            else:
                print(f"📥 Executor -> Planner: 收到反馈")
                print(f"   Step ID: {executor_feedback_post.attachments.get('step_id', 'unknown')}")
                # 只显示精炼后的数据预览
                result_data = executor_feedback_post.attachments.get('result', {})
                compressed_data = self._extract_structured_content(result_data)
                # 🔧 精简结果预览，截断长数据
                display_data = self._truncate_for_debug_display(compressed_data)
                # 调试输出改为更简洁的成功/失败提示
                if isinstance(result_data, dict) and result_data.get('success') is True:
                    print("   执行结果: success")
                elif isinstance(result_data, dict) and result_data.get('success') is False:
                    print(f"   执行结果: failed (error={result_data.get('error', 'unknown')})")
            
            self._update_memory_from_feedback(executor_feedback_post, llm_decision)

            # 如果步骤成功完成，更新进度
            result_data = executor_feedback_post.attachments.get('result', {})
            if result_data.get('success', False):
                completed_steps += 1
                if progress_callback:
                    used_tool = result_data.get('used_tool') if isinstance(result_data, dict) else None
                    llm_summary = result_data.get('llm_summary') if isinstance(result_data, dict) else None
                    await progress_callback({
                        "type": "step_complete",
                        "step": original_step_id,
                        "step_number": current_step_number,
                        "total_steps": total_steps,
                        "message": f"步骤 {current_step_number} 完成",
                        "result": result_data,
                        "completed_steps": completed_steps,
                        "status": "completed",
                        "used_tool": used_tool,
                        "task_type": result_data.get('task_type') if isinstance(result_data, dict) else None,
                        "function": used_tool,
                        "llm_summary": llm_summary
                    })
            else:
                if progress_callback:
                    used_tool = result_data.get('used_tool') if isinstance(result_data, dict) else None
                    llm_summary = result_data.get('llm_summary') if isinstance(result_data, dict) else None
                    await progress_callback({
                        "type": "step_error",
                        "step": original_step_id,
                        "step_number": current_step_number,
                        "total_steps": total_steps,
                        "message": f"步骤 {current_step_number} 执行失败",
                        "error": result_data.get('error', '未知错误'),
                        "status": "failed",
                        "used_tool": used_tool,
                        "task_type": result_data.get('task_type') if isinstance(result_data, dict) else None,
                        "function": used_tool,
                        "llm_summary": llm_summary
                    })

            current_post = executor_feedback_post
            self.memory.rounds.append(round_obj) # 将完整的当前回合存入记忆

            # MCP 原生模式下：若在同一轮 outputs 已经直接返回了 tool_result 并被存入 shared_storage，
            # 此处无需额外动作；仍保持一致的回合推进。

        return self._generate_final_report(query)
    
    async def _execute_natively(self, step_id: str, instruction: str, tool_name: Optional[str], arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """MCP工具代理：支持SQL相关工具的多种名称"""
        try:
            # 支持多种SQL工具名称
            sql_tool_names = ['sql_query', 'generate_sql', 'execute_sql']
            if (tool_name or '').lower() in sql_tool_names:
                natural_language = (arguments or {}).get('natural_language') or instruction
                limit = (arguments or {}).get('limit', 10)
                data = await self._mcp_sql_call(step_id, natural_language, limit)
                return {
                    'success': True,
                    'task_type': 'sql',
                    'data': data,
                    'used_tool': tool_name  # 使用实际的工具名
                }
            
            # 对于其他工具，返回不支持的错误
            return {
                'success': False,
                'error_type': 'unsupported_tool',
                'error': f"工具 '{tool_name}' 需要通过增强执行器处理",
                'step_id': step_id
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': 'execution_error',
                'error': str(e),
                'step_id': step_id
            }

    async def _mcp_sql_call(self, step_id: str, natural_language: str, limit: int = 10) -> Dict[str, Any]:
        """调用 MCP SQL 工具，返回可序列化结果。
        注意：服务端工具（如 generate_sql）不接受 limit 字段，这里不向服务端传递 limit，仅保留自然语言。
        """
        try:
            from fastmcp import Client
        except Exception as e:
            raise RuntimeError(f"缺少 fastmcp 依赖或导入失败: {e}")

        base_url = os.getenv('SQL_EXECUTOR_URL', 'http://127.0.0.1:8000/mcp/')
        sql_args = {
            'natural_language': natural_language,
            'include_preview': True,
            'output_format': 'json',
            'return_json_directly': True
        }
        if self.dataflow_only:
            try:
                print("[DATAFLOW] MCP -> request", json.dumps({
                    "base_url": base_url,
                    "step_id": step_id,
                    "args": {"natural_language": natural_language}
                }, ensure_ascii=False))
            except Exception:
                print("[DATAFLOW] MCP -> request (summary)")

        async with Client(base_url) as session:
            tools = await session.list_tools()
            mcp_tool_name = None
            for t in tools:
                name = getattr(t, 'name', str(t))
                if 'sql' in name.lower():
                    mcp_tool_name = name
                    break
            if not mcp_tool_name and tools:
                mcp_tool_name = getattr(tools[0], 'name', str(tools[0]))
            if not mcp_tool_name:
                raise RuntimeError('MCP 未发现可用工具')
            result = await session.call_tool(mcp_tool_name, sql_args)
            # fastmcp CallToolResult
            if str(type(result)).find('CallToolResult') != -1:
                content = getattr(result, 'content', None)
                if content and len(content) > 0 and hasattr(content[0], 'text'):
                    text_data = content[0].text
                    try:
                        parsed = json.loads(text_data)
                        if self.dataflow_only:
                            try:
                                print("[DATAFLOW] MCP <- response", json.dumps({
                                    "step_id": step_id,
                                    "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                                    "row_count": parsed.get('row_count') if isinstance(parsed, dict) else None
                                }, ensure_ascii=False))
                            except Exception:
                                print("[DATAFLOW] MCP <- response (summary)")
                        return parsed
                    except Exception:
                        return {'raw': text_data}
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            if hasattr(result, 'dict'):
                return result.dict()
            return {'raw_result': str(result)}

    async def _chat_tools_followup_loop(
        self,
        original_step_id: str,
        user_prompt: str,
        first_assistant_tool_call: Dict[str, Any],
        tools_spec: List[Dict[str, Any]],
        first_tool_name: str,
        first_tool_call_id: str,
        first_tool_data: Any,
        max_rounds: int = 3
    ) -> Dict[str, Any]:
        """在 Chat+tools 模式下，执行多轮 工具调用→回传→继续 对话，直到模型返回最终文本或达到上限。返回最后一次完整响应。"""
        # 初始消息（用户 + 第一次 assistant 的 tool_calls + 第一次 tool 结果）
        try:
            content_str = json.dumps(first_tool_data, ensure_ascii=False) if not isinstance(first_tool_data, str) else first_tool_data
        except Exception:
            content_str = str(first_tool_data)

        if len(content_str) > 50000:
            content_str = content_str[:50000] + "... (truncated)"

        messages: List[Dict[str, Any]] = [
            { 'role': 'user', 'content': user_prompt },
            { 'role': 'assistant', 'tool_calls': [ first_assistant_tool_call ] },
            { 'role': 'tool', 'tool_call_id': first_tool_call_id, 'name': first_tool_name, 'content': content_str }
        ]

        final_resp: Dict[str, Any] = {}
        turns = 0
        while turns < max_rounds:
            turns += 1
            resp = self.llm_api.chat_completion_full(messages, tools=tools_spec, stream=False)
            final_resp = resp
            try:
                choice0 = (resp.get('choices') or [{}])[0]
                finish_reason = choice0.get('finish_reason')
                msg = choice0.get('message') or {}
                if finish_reason == 'tool_calls':
                    # 继续执行新一轮工具
                    tool_calls = msg.get('tool_calls') or []
                    if not tool_calls:
                        break
                    # 暂时仅执行第一个
                    tc = tool_calls[0]
                    name = (tc.get('function') or {}).get('name')
                    args_text = (tc.get('function') or {}).get('arguments') or '{}'
                    try:
                        args = json.loads(args_text)
                    except Exception:
                        args = { 'raw': args_text }

                    # 执行工具（通过增强执行器）
                    if self.enhanced_executor:
                        context = {
                            'shared_storage': self.memory.shared_storage,
                            'step_records': self.memory.step_records,
                            'progress': self.memory.get_progress()
                        }
                        result_payload = await self.enhanced_executor.execute(
                            original_step_id, user_prompt, context, name, args
                        )
                    else:
                        result_payload = await self._execute_natively(original_step_id, user_prompt, name, args)
                    tool_content_obj = result_payload.get('data') if isinstance(result_payload, dict) else result_payload
                    try:
                        tool_content_str = json.dumps(tool_content_obj, ensure_ascii=False)
                    except Exception:
                        tool_content_str = str(tool_content_obj)
                    if len(tool_content_str) > 50000:
                        tool_content_str = tool_content_str[:50000] + "... (truncated)"

                    # 追加 assistant 的 tool_calls 与 tool 的结果
                    messages.append({ 'role': 'assistant', 'tool_calls': tool_calls })
                    messages.append({ 'role': 'tool', 'tool_call_id': tc.get('id'), 'name': name, 'content': tool_content_str })
                    continue
                else:
                    # 收到最终文本/停止
                    break
            except Exception:
                break
        return final_resp
    
    def _truncate_for_debug_display(self, data: Any, max_rows: int = 3) -> Any:
        """为调试输出截断长数据，保持可读性"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'query_data' and isinstance(value, list):
                    # SQL结果数据：只显示前N行
                    if len(value) > max_rows:
                        result[key] = value[:max_rows] + [f"... (+{len(value) - max_rows} more rows)"]
                    else:
                        result[key] = value
                elif key in ['data'] and isinstance(value, list) and len(value) > max_rows:
                    # 图表数据：只显示前N行
                    result[key] = value[:max_rows] + [f"... (+{len(value) - max_rows} more rows)"]
                else:
                    result[key] = value
            return result
        elif isinstance(data, list) and len(data) > max_rows:
            # 纯列表：只显示前N项
            return data[:max_rows] + [f"... (+{len(data) - max_rows} more items)"]
        else:
            return data

    def _extract_structured_content(self, result: Dict[str, Any]) -> Any:
        if not result:
            return None

        # 失败：返回错误结构
        if result.get('success') is False:
            return {"error": result.get('error', '执行失败')}

        data = result.get('data')
        
        # 处理CallToolResult对象（直接从对象提取）
        if str(type(data)).find("CallToolResult") != -1:
            try:
                content = data.content
                if content and len(content) > 0:
                    text_data = content[0].text if hasattr(content[0], 'text') else str(content[0])
                    if isinstance(text_data, str):
                        inner_data = json.loads(text_data)
                        return inner_data  # 返回原始字典，不再压缩
            except Exception as e:
                return {"error": str(e)}

        # 🔧 处理CallToolResult格式：先提取真正的JSON内容
        if isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
            # 提取实际的文本内容
            for item in data['content']:
                if isinstance(item, dict) and 'text' in item:
                    try:
                        inner_data = json.loads(item['text'])
                        return inner_data  # 返回原始字典
                    except json.JSONDecodeError as e:
                        return {"error": f"JSON解析失败: {str(e)}"}

        # 字符串情况：尽量解析JSON
        if isinstance(data, str):
            s = data.strip()
            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                try:
                    parsed = json.loads(s)
                    return parsed  # 返回解析后的数据
                except json.JSONDecodeError as e:
                    return {"error": f"JSON解析失败: {str(e)}"}
            return {"text": data}  # 不截断

        # 已经是字典或列表，直接返回
        if isinstance(data, (dict, list)):
            return data

        # 其他类型，包装成标准格式
        return {"data": data}

    # 只保留MCP原生调用，已删除传统executor方法
            
    def _update_memory_from_feedback(self, feedback_post: Post, llm_decision: LLMDecision):
        step_id = feedback_post.attachments['step_id']
        result = feedback_post.attachments['result']
        # 归一化重试步骤ID：支持 step1_retry 与 step1_retry_1
        if isinstance(step_id, str):
            if '_retry_' in step_id:
                original_step_id = step_id.split('_retry_')[0]
            elif step_id.endswith('_retry'):
                original_step_id = step_id[:-6]
            else:
                original_step_id = step_id
        else:
            original_step_id = step_id

        # 提取并记录 structured_content
        sc = self._extract_structured_content(result)
        # 若有 LLM 总结，合并到 structured_content，便于回显
        try:
            if isinstance(sc, dict) and isinstance(result, dict) and result.get('llm_summary'):
                sc = { **sc, 'llm_summary': result.get('llm_summary'), 'summary_text': result.get('llm_summary') }
        except Exception:
            pass
        rec = self.memory.step_records.get(original_step_id) or StepRecord()
        rec.structured_content = sc

        if result and result.get('success'):
            data = result.get('data')
            skipped = isinstance(data, dict) and data.get('status') == 'skipped'
            if skipped:
                self.memory.step_status[original_step_id] = "skipped"
                rec.status = "skipped"
            else:
                self.memory.step_status[original_step_id] = "completed"
                rec.status = "completed"

            # 自动存储成功的结果到共享存储
            key = f"{original_step_id}_result"
            original_data = result.get('data')
            
            # 创建标准存储格式
            if isinstance(original_data, dict):
                stored_data = {
                    "success": original_data.get("success", True),
                    "sql_query": original_data.get("sql_query", ""),
                    "query_data": original_data.get("query_data", original_data.get("preview_data", [])),
                    "row_count": original_data.get("row_count", original_data.get("total_rows", 0)),
                    "task_type": result.get('task_type', 'unknown')
                }
                
                # 针对图表结果的特殊处理
                if result.get('task_type') == 'chart' or 'dsl' in original_data:
                    stored_data.update({
                        "task_type": "chart",
                        "chart_dsl": original_data.get('dsl', {}),
                        "title": original_data.get('title'),
                        "axis": original_data.get('axis')
                    })
                    
            elif isinstance(original_data, list):
                stored_data = {
                    "success": True,
                    "sql_query": "",
                    "query_data": original_data,
                    "row_count": len(original_data),
                    "task_type": result.get('task_type', 'unknown')
                }
            else:
                stored_data = {
                    "success": True,
                    "sql_query": "",
                    "query_data": [],
                    "row_count": 0,
                    "task_type": result.get('task_type', 'unknown')
                }

            # 追加 LLM 总结文本（若存在）
            if isinstance(result, dict) and result.get('llm_summary'):
                stored_data['llm_summary'] = result.get('llm_summary')
                stored_data['summary_text'] = result.get('llm_summary')
            
            self.memory.shared_storage[key] = stored_data
            
            if self.dataflow_only:
                try:
                    print("[DATAFLOW] STORAGE <- write", json.dumps({
                        "key": key,
                        "row_count": stored_data.get('row_count'),
                        "has_query_data": bool(stored_data.get('query_data')),
                        "task_type": stored_data.get('task_type', 'sql')
                    }, ensure_ascii=False))
                except Exception:
                    print("[DATAFLOW] STORAGE <- write (summary)")
            else:
                # 保留最简日志，避免在终端输出完整内容
                print(f"💾 结果已存入共享存储: key='{key}', type={type(stored_data).__name__}")
        else:
            self.memory.step_status[original_step_id] = "failed"
            self.memory.step_retries[original_step_id] = self.memory.step_retries.get(original_step_id, 0) + 1
            rec.status = "failed"
            rec.retries = self.memory.step_retries[original_step_id]
            error_msg = result.get('error', '未知错误') if result else "执行器返回了空结果"
            print(f"🔥 步骤 '{original_step_id}' 失败。错误: {error_msg}")

        self.memory.step_records[original_step_id] = rec

    def _generate_final_report(self, query: str) -> str:
        print("\n--- 📊 生成最终报告 ---")
        progress = self.memory.get_progress()
        # 人类可读的简短摘要（不包含大段JSON，避免前端渲染卡片噪音）
        report = (
            "## 任务执行报告\n"
            f"**原始查询:** {query}\n"
            "**最终摘要:**\n"
            f"- ✅ 已完成步骤: {progress.get('completed_steps', [])}\n"
            f"- ❌ 已失败步骤: {progress.get('failed_steps', [])}\n"
        ).strip()

        # 仅展示共享存储的键摘要，避免在摘要中内联海量JSON
        storage_keys = list(self.memory.shared_storage.keys())
        if storage_keys:
            storage_summary = "**共享存储摘要（键）:**\n" + "\n".join([f"- {k}" for k in storage_keys])
        else:
            storage_summary = "**共享存储摘要（键）:** _空_"

        # 步骤摘要（若存在 summary_text/llm_summary）
        step_summary_lines: List[str] = []
        try:
            def _step_order(s: str) -> int:
                try:
                    return int(''.join([c for c in s if c.isdigit()]) or 0)
                except Exception:
                    return 0
            for step_id in sorted(self.memory.initial_plan.keys(), key=_step_order):
                rec = self.memory.step_records.get(step_id)
                sv = None
                if rec and isinstance(rec.structured_content, dict):
                    sv = rec.structured_content.get('summary_text') or rec.structured_content.get('llm_summary')
                if sv:
                    step_summary_lines.append(f"- {step_id}: {sv}")
        except Exception:
            pass
        step_summaries_md = ("### 步骤摘要\n" + "\n".join(step_summary_lines)) if step_summary_lines else ""

        # 详细共享存储放入代码块，便于被上层 human_summary 截断
        shared_storage_json = json.dumps(self.memory.shared_storage, indent=2, ensure_ascii=False)

        # 追加可机读记忆 JSON（每步实际指令 + structured_content）
        memory_data = {}
        for step_id in sorted(self.memory.initial_plan.keys(), key=_step_order):
            rec = self.memory.step_records.get(step_id)
            memory_data[step_id] = {
                "instruction": (rec.instruction if rec and rec.instruction else self.memory.initial_plan.get(step_id)),
                "structured_content": (rec.structured_content if rec else None),
                "status": (rec.status if rec else self.memory.step_status.get(step_id, "pending")),
                "retries": (rec.retries if rec else self.memory.step_retries.get(step_id, 0)),
            }

        json_tail = json.dumps({"memory": memory_data}, ensure_ascii=False, indent=2)

        # 返回：人类摘要 + 步骤摘要 + 共享存储键摘要 + 详细共享存储（代码块）+ 机读记忆（代码块）
        return (
            f"{report}\n\n"
            f"{step_summaries_md}\n\n"
            f"{storage_summary}\n\n"
            f"### 共享存储详细内容\n"
            f"```json\n{shared_storage_json}\n```\n\n"
            f"### 执行记忆（机读）\n"
            f"```json\n{json_tail}\n```"
        )

