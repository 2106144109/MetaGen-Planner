# ==========================================================
# 文件: enhanced_executor.py (简化MCP专用版)
# 职责: 简化的MCP执行器，只保留核心工具调用功能
# ==========================================================

from fastmcp import Client
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
import time

from .tool_registry import ToolRegistry


@dataclass
class ServerCapabilityProfile:
    server_url: str
    mcp_protocol_version: str = "unknown"
    discovered_tools: List[str] = field(default_factory=list)
    declared_categories: List[str] = field(default_factory=list)
    health_score: float = 1.0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_error: Optional[str] = None
    last_checked_at: float = 0.0


class EnhancedMcpExecutor:
    """使用工具注册表的增强MCP执行器 - 支持多个MCP服务器"""
    
    def __init__(self, mcp_servers: List[str]):
        """
        初始化执行器
        Args:
            mcp_servers: MCP服务器URL列表，例如：
                ["http://127.0.0.1:8000/mcp/", "http://192.168.90.146:1122/mcp"]
        """
        self.mcp_servers = mcp_servers
        self.mcp_sessions: Dict[str, Client] = {}
        self.server_profiles: Dict[str, ServerCapabilityProfile] = {}
        
        # 动态工具注册表，完全从MCP服务器获取工具信息
        self.tool_registry = ToolRegistry()
        self.is_connected = False
        self.max_connect_retries = int(os.getenv("MCP_CONNECT_MAX_RETRIES", "3"))
        self.retry_base_seconds = float(os.getenv("MCP_RETRY_BASE_SECONDS", "0.5"))
        self.circuit_failure_threshold = int(os.getenv("MCP_CIRCUIT_FAILURE_THRESHOLD", "3"))
        self.circuit_open_seconds = float(os.getenv("MCP_CIRCUIT_OPEN_SECONDS", "30"))
        # MCP 协议版本（日期制）
        self.expected_mcp_version = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18")

    def _infer_categories_from_tools(self, tool_names: List[str]) -> List[str]:
        categories = set()
        for name in tool_names:
            low = (name or "").lower()
            if "sql" in low:
                categories.add("sql")
            if any(x in low for x in ["chart", "plot", "bar", "line", "pie", "histogram", "scatter"]):
                categories.add("chart")
            if "file" in low:
                categories.add("file")
            if "web" in low:
                categories.add("web")
        return sorted(list(categories))

    async def _probe_server_profile(self, server_url: str, session) -> ServerCapabilityProfile:
        tools = await session.list_tools()
        tool_names: List[str] = []
        for t in tools or []:
            tool_names.append(getattr(t, "name", str(t)))

        categories = self._infer_categories_from_tools(tool_names)
        profile = self.server_profiles.get(server_url) or ServerCapabilityProfile(server_url=server_url)
        profile.discovered_tools = tool_names
        profile.declared_categories = categories
        profile.last_checked_at = time.time()
        profile.last_error = None
        # 协议协商（日期制）：当前 fastmcp 客户端未暴露明确字段时，记录期望版本作为会话协商版本
        profile.mcp_protocol_version = self.expected_mcp_version
        self.server_profiles[server_url] = profile
        return profile

    def _mark_server_failure(self, server_url: str, error: str):
        profile = self.server_profiles.get(server_url) or ServerCapabilityProfile(server_url=server_url)
        profile.consecutive_failures += 1
        profile.last_error = error
        profile.health_score = max(0.05, profile.health_score * 0.7)
        if profile.consecutive_failures >= self.circuit_failure_threshold:
            profile.circuit_open_until = time.time() + self.circuit_open_seconds
        self.server_profiles[server_url] = profile

    def _mark_server_success(self, server_url: str):
        profile = self.server_profiles.get(server_url) or ServerCapabilityProfile(server_url=server_url)
        profile.consecutive_failures = 0
        profile.circuit_open_until = 0.0
        profile.last_error = None
        profile.health_score = min(1.0, profile.health_score + 0.1)
        profile.last_checked_at = time.time()
        self.server_profiles[server_url] = profile

    def _is_circuit_open(self, server_url: str) -> bool:
        profile = self.server_profiles.get(server_url)
        if not profile:
            return False
        return profile.circuit_open_until > time.time()
    
    async def connect(self) -> bool:
        """连接所有MCP服务器并发现工具"""
        if self.is_connected:
            return True
            
        print(f"🔌 [MCP Executor] 正在连接 {len(self.mcp_servers)} 个MCP服务器...")
        
        total_discovered = 0
        connected_servers = 0
        
        for server_url in self.mcp_servers:
            print(f"  📡 正在连接服务器: {server_url}")
            connected = False
            for attempt in range(1, self.max_connect_retries + 1):
                try:
                    client = Client(server_url)
                    self.mcp_sessions[server_url] = client
                    async with client as session:
                        profile = await self._probe_server_profile(server_url, session)
                        discovered_count = await self.tool_registry.register_from_mcp_server(session)
                        total_discovered += discovered_count
                        connected_servers += 1
                        connected = True
                        self._mark_server_success(server_url)
                        print(
                            f"  ✅ 服务器 {server_url} 连接成功，发现 {discovered_count} 个工具，"
                            f"categories={profile.declared_categories}, mcp_version={profile.mcp_protocol_version}"
                        )
                    break
                except Exception as e:
                    self._mark_server_failure(server_url, str(e))
                    wait_seconds = self.retry_base_seconds * (2 ** (attempt - 1))
                    print(f"  ⚠️ 服务器 {server_url} 连接失败 (attempt {attempt}/{self.max_connect_retries}): {e}")
                    if attempt < self.max_connect_retries:
                        time.sleep(wait_seconds)
            if not connected:
                print(f"  ❌ 服务器 {server_url} 最终连接失败，已跳过")
        
        if connected_servers == 0:
            print("❌ [MCP Executor] 所有MCP服务器连接失败，启用fallback模式")
            # 使用fallback工具
            fallback_count = self.tool_registry.register_fallback_tools()
            print(f"🔧 [MCP Executor] Fallback模式：注册了 {fallback_count} 个基本工具")
            
            if fallback_count == 0:
                print("❌ [MCP Executor] Fallback工具注册也失败")
                return False
        else:
            print(f"🎯 [MCP Executor] 成功连接 {connected_servers}/{len(self.mcp_servers)} 个服务器，共注册 {total_discovered} 个工具")
            
            # 即使连接成功，也可以注册一些fallback工具作为补充
            if total_discovered == 0:
                print("⚠️ [MCP Executor] 未发现任何工具，注册fallback工具")
                fallback_count = self.tool_registry.register_fallback_tools()
                print(f"🔧 [MCP Executor] 补充注册了 {fallback_count} 个fallback工具")
        
        # 3. 设置执行器上下文
        for handler in self.tool_registry.handlers.values():
            handler.executor_context = self
        
        # 4. 准备所有工具
        for tool_name, handler in self.tool_registry.handlers.items():
            try:
                await handler.prepare()
            except Exception as e:
                print(f"  ⚠️ 工具 '{tool_name}' 准备失败: {e}")
        
        self.is_connected = True
        self._print_tool_summary()
        return True
    
    async def disconnect(self):
        """断开所有连接"""
        for server_url, client in self.mcp_sessions.items():
            try:
                # fastmcp Client使用上下文管理器，自动管理连接
                # 这里我们只需要清理引用
                await client.close()
                print(f"🔌 [MCP Executor] 已断开服务器: {server_url}")
            except Exception as e:
                print(f"⚠️ [MCP Executor] 断开服务器 {server_url} 时出错: {e}")
        
        self.mcp_sessions.clear()
        self.is_connected = False
    
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     tool_name: Optional[str] = None, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行工作流步骤"""
        if not self.is_connected:
            raise ConnectionError("执行器未连接，请先调用 connect()")
            
        print(f"⚡️ [MCP Executor] 执行步骤 '{step_id}': {tool_name or '自动选择工具'}")
        
        try:
            # 1. 确定使用的工具
            selected_tool = self._select_tool(tool_name, instruction, arguments)
            if not selected_tool:
                raise ValueError(f"无法找到合适的工具处理: {tool_name or instruction}")
            
            # 2. 获取工具处理器
            handler = self.tool_registry.get_handler(selected_tool)
            if not handler:
                raise ValueError(f"工具 '{selected_tool}' 没有注册处理器")
            
            # 3. 通过MCP执行工具任务
            
            # 选择合适的MCP session
            if not self.mcp_sessions:
                # Fallback模式：没有真实的MCP连接，使用模拟执行
                print(f"🔧 [MCP Executor] Fallback模式执行工具 '{selected_tool}'")
                try:
                    result = await self._execute_fallback(step_id, instruction, context, arguments, handler)
                    return result
                except Exception as fallback_error:
                    print(f"❌ [MCP Executor] Fallback执行失败: {fallback_error}")
                    return {
                        'success': False,
                        'error_type': 'fallback_error',
                        'error': str(fallback_error),
                        'step_id': step_id
                    }
            
            server_url = self._select_mcp_server_url(selected_tool)
            
            # 🔧 修复会话嵌套问题：每次都创建新的client实例
            client = Client(server_url)
            
            async with client as session:
                # 设置session给handler使用
                original_session = getattr(handler.executor_context, 'sql_session', None) if handler.executor_context else None
                if handler.executor_context:
                    handler.executor_context.sql_session = session
                try:
                    result = await handler.execute(step_id, instruction, context, arguments)
                    if isinstance(result, dict) and 'success' in result and not result.get('success'):
                        print(f"  ❌ [MCP Executor] 工具执行失败: {result.get('error', 'N/A')}")
                        self._mark_server_failure(server_url, str(result.get('error', 'tool_failed')))
                    else:
                        self._mark_server_success(server_url)
                except Exception as handler_error:
                    print(f"  ❌ [MCP Executor] Handler执行异常: {handler_error}")
                    self._mark_server_failure(server_url, str(handler_error))
                    raise
                finally:
                    # 恢复原来的session
                    if handler.executor_context and original_session is not None:
                        handler.executor_context.sql_session = original_session
            
            # 4. 标准化结果
            if isinstance(result, dict) and 'success' in result:
                return result
            else:
                return {
                    'success': True,
                    'task_type': 'generic',
                    'data': result,
                    'used_tool': selected_tool
                }
                
        except Exception as e:
            print(f"❌ [MCP Executor] 执行步骤失败: {e}")
            return {
                'success': False,
                'error_type': 'execution_error',
                'error': str(e),
                'step_id': step_id
            }
    
    def _select_tool(self, tool_name: Optional[str], instruction: str, 
                    arguments: Optional[Dict[str, Any]]) -> Optional[str]:
        """简化的工具选择：只支持LLM明确指定的工具名"""
        if not tool_name:
            return None
            
        # 1. 检查工具注册表中是否有此工具
        if tool_name in self.tool_registry.tools:
            return tool_name
            
        # 1.5. 特殊处理：图表工具名称映射
        chart_tool_mapping = {
            'generate_column_chart': 'generate_column_chart',
            'generate_bar_chart': 'generate_bar_chart',
            'generate_pie_chart': 'generate_pie_chart',
            'generate_line_chart': 'generate_line_chart'
        }
        if tool_name in chart_tool_mapping:
            mapped_name = chart_tool_mapping[tool_name]
            if mapped_name in self.tool_registry.tools:
                return mapped_name
            
        # 2. 检查是否是MCP工具名，需要映射
        mapped_tool = self.tool_registry.get_tool_by_mcp_name(tool_name)
        if mapped_tool:
            return mapped_tool
            
        # 3. 未找到匹配的工具
        print(f"  ❌ [MCP Executor] 未找到工具: '{tool_name}'")
        return None
    
    def _select_mcp_server_url(self, tool_name: str) -> str:
        """根据工具能力 + 健康评分 + 熔断状态选择MCP服务器URL"""
        if not self.mcp_sessions:
            raise RuntimeError("没有可用的MCP连接")
        
        tool_def = self.tool_registry.tools.get(tool_name)
        desired_category = tool_def.category if tool_def else None
        candidate_scores: List[tuple[float, str]] = []

        for server_url in self.mcp_sessions.keys():
            if self._is_circuit_open(server_url):
                continue

            profile = self.server_profiles.get(server_url)
            health = profile.health_score if profile else 0.5
            capability_bonus = 0.0
            if desired_category and profile and desired_category in profile.declared_categories:
                capability_bonus += 1.0
            # 兼容未声明能力时的弱启发式
            if desired_category == "chart" and (":1122" in server_url or "chart" in server_url.lower()):
                capability_bonus += 0.3
            if desired_category == "sql" and (":8000" in server_url or "sql" in server_url.lower()):
                capability_bonus += 0.3
            score = health + capability_bonus
            candidate_scores.append((score, server_url))

        if not candidate_scores:
            # 若全部熔断，选择最早恢复的一个，避免彻底不可用
            if self.server_profiles:
                fallback_url = min(
                    self.server_profiles.keys(),
                    key=lambda url: self.server_profiles[url].circuit_open_until or 0.0
                )
                print(f"  ⚠️ [MCP Executor] 所有节点熔断，使用最早恢复节点: {fallback_url}")
                return fallback_url
            return list(self.mcp_sessions.keys())[0]

        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        selected_score, selected_url = candidate_scores[0]
        profile = self.server_profiles.get(selected_url)
        print(
            f"  🎯 [MCP Executor] 选择服务器: {selected_url} "
            f"(score={selected_score:.2f}, health={profile.health_score if profile else 0.5:.2f}, "
            f"categories={profile.declared_categories if profile else []})"
        )
        return selected_url
    
    def _select_mcp_session(self, tool_name: str) -> Client:
        """根据工具选择合适的MCP session (已弃用，保留兼容性)"""
        server_url = self._select_mcp_server_url(tool_name)
        return self.mcp_sessions[server_url]
    
    async def _execute_fallback(self, step_id: str, instruction: str, context: Dict[str, Any], 
                               arguments: Optional[Dict[str, Any]], handler) -> Dict[str, Any]:
        """Fallback模式执行：在没有MCP连接时的模拟执行"""
        tool_def = handler.tool_def
        
        if tool_def.category == 'sql':
            # 模拟SQL执行
            return {
                'success': True,
                'task_type': 'sql',
                'data': {
                    'success': True,
                    'sql_query': f"-- 模拟SQL查询: {instruction}",
                    'query_data': [
                        {'示例列1': '模拟数据1', '示例列2': 100},
                        {'示例列1': '模拟数据2', '示例列2': 200},
                        {'示例列1': '模拟数据3', '示例列2': 300}
                    ],
                    'row_count': 3,
                    'message': 'Fallback模式：这是模拟的SQL执行结果'
                },
                'used_tool': 'sql_query_fallback'
            }
        elif tool_def.category == 'chart':
            # 模拟图表生成
            return {
                'success': True,
                'task_type': 'chart',
                'data': {
                    'chart_type': tool_def.config.get('chart_type', 'bar'),
                    'title': '模拟图表',
                    'dsl': {
                        'mark': tool_def.config.get('chart_type', 'bar'),
                        'encoding': {
                            'x': {'field': '示例列1'},
                            'y': {'field': '示例列2'}
                        },
                        'title': '模拟图表标题',
                        'data': [
                            {'示例列1': '类别A', '示例列2': 100},
                            {'示例列1': '类别B', '示例列2': 200},
                            {'示例列1': '类别C', '示例列2': 150}
                        ]
                    },
                    'structured_content': {
                        'type': 'chart_render',
                        'payload': {
                            'message': 'Fallback模式：这是模拟的图表生成结果',
                            'chart_url': '/charts/fallback_chart.png'
                        }
                    }
                },
                'used_tool': f'{tool_def.name}_fallback'
            }
        else:
            # 其他工具类型的模拟执行
            return {
                'success': True,
                'task_type': tool_def.category,
                'data': {
                    'message': f'Fallback模式执行: {instruction}',
                    'tool_category': tool_def.category,
                    'simulated': True
                },
                'used_tool': f'{tool_def.name}_fallback'
            }
    
    def _print_tool_summary(self):
        """打印工具摘要（简化版）"""
        enabled_tools = [(name, tool_def) for name, tool_def in self.tool_registry.tools.items() if tool_def.enabled]
        
        if self.mcp_sessions:
            print(f"📋 [MCP Executor] 已连接 {len(self.mcp_sessions)} 个服务器，共 {len(enabled_tools)} 个工具就绪")
            for server_url, profile in self.server_profiles.items():
                print(
                    f"   - {server_url} | version={profile.mcp_protocol_version} "
                    f"| categories={profile.declared_categories} | health={profile.health_score:.2f} "
                    f"| failures={profile.consecutive_failures}"
                )
        else:
            print(f"📋 [MCP Executor] Fallback模式，共 {len(enabled_tools)} 个工具就绪")

    def get_server_profiles(self) -> Dict[str, Dict[str, Any]]:
        """导出MCP服务器能力/健康画像，便于上层观测。"""
        result: Dict[str, Dict[str, Any]] = {}
        for url, profile in self.server_profiles.items():
            result[url] = {
                "mcp_protocol_version": profile.mcp_protocol_version,
                "declared_categories": profile.declared_categories,
                "discovered_tools": profile.discovered_tools,
                "health_score": profile.health_score,
                "consecutive_failures": profile.consecutive_failures,
                "circuit_open_until": profile.circuit_open_until,
                "last_error": profile.last_error,
                "last_checked_at": profile.last_checked_at
            }
        return result
