# ==========================================================
# 文件: tool_registry.py
# 职责: 可扩展的工具注册表管理系统
# ==========================================================

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class ToolDefinition:
    """工具定义数据类"""
    name: str                           # 工具名称
    category: str                       # 工具分类 (sql, chart, file, web, etc.)
    description: str                    # 工具描述
    discovery_pattern: Optional[str] = None    # MCP工具发现模式（如 "*sql*"）
    handler_class: Optional[str] = None        # 处理器类名
    config: Dict[str, Any] = field(default_factory=dict)  # 工具特定配置
    enabled: bool = True                       # 是否启用
    priority: int = 100                        # 优先级（数字越小优先级越高）
    
    def matches_mcp_tool(self, mcp_tool_name: str) -> bool:
        """检查MCP工具名是否匹配此工具定义"""
        if not self.discovery_pattern:
            return self.name.lower() == mcp_tool_name.lower()
        
        import fnmatch
        return fnmatch.fnmatch(mcp_tool_name.lower(), self.discovery_pattern.lower())

class BaseToolHandler(ABC):
    """工具处理器基类"""
    
    def __init__(self, tool_def: ToolDefinition, executor_context: Any = None):
        self.tool_def = tool_def
        self.executor_context = executor_context
        self.mcp_tool_name: Optional[str] = None
    
    @abstractmethod
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行工具任务"""
        pass
    
    def set_mcp_tool_name(self, mcp_tool_name: str):
        """设置发现的MCP工具名称"""
        self.mcp_tool_name = mcp_tool_name
        
    async def prepare(self) -> bool:
        """工具准备（可选重写）"""
        return True

class SqlToolHandler(BaseToolHandler):
    """SQL工具处理器"""
    
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行SQL任务"""
        if not self.mcp_tool_name:
            raise ValueError("SQL工具未发现MCP名称")
            
        sql_session = self.executor_context.sql_session
        
        sql_args = {
            "natural_language": instruction,
            "evidence": f"这是工作流的步骤 {step_id}。",
            "include_preview": True,
            "output_format": "json",
            "return_json_directly": True
        }
        
        # 调用SQL工具
        result = await sql_session.call_tool(self.mcp_tool_name, sql_args)
        
        # 标准化返回格式
        return {
            'success': True,
            'task_type': 'sql',
            'data': self._parse_sql_result(result),
            'used_tool': self.mcp_tool_name
        }
    
    def _parse_sql_result(self, result: Any) -> Dict[str, Any]:
        """解析SQL结果"""
        serializable_data = {}
        if str(type(result)).find("CallToolResult") != -1:
            content = result.content
            if content and len(content) > 0 and hasattr(content[0], 'text'):
                text_data = content[0].text
                if isinstance(text_data, str):
                    try:
                        serializable_data = json.loads(text_data)
                    except json.JSONDecodeError as e:
                        serializable_data = {"error": str(e)}
        
        return serializable_data

class ChartToolHandler(BaseToolHandler):
    """图表工具处理器 - 支持25种不同类型的图表"""
    
    def _validate_encoding_fields(self, mcp_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """校验 encoding 中引用的字段是否在 data 中存在。
        返回 None 表示通过；返回 dict 表示校验失败并给出可读错误信息。
        """
        data = mcp_args.get('data')
        encoding = mcp_args.get('encoding')

        if not isinstance(data, list) or not data:
            return None
        if not isinstance(encoding, dict) or not encoding:
            return None

        dict_rows = [row for row in data if isinstance(row, dict)]
        if not dict_rows:
            return None

        # 使用多行并集字段，避免仅看首行导致误报
        available_fields_set = set()
        for row in dict_rows[:50]:
            available_fields_set.update(row.keys())
        available_fields = sorted(list(available_fields_set))
        missing_fields: List[str] = []

        for axis, field_info in encoding.items():
            if isinstance(field_info, dict):
                requested_field = field_info.get('field')
                if isinstance(requested_field, str) and requested_field and requested_field not in available_fields_set:
                    missing_fields.append(f"{axis}:{requested_field}")

        if not missing_fields:
            return None

        return {
            'success': False,
            'error_type': 'invalid_encoding_fields',
            'error': '图表encoding字段不存在于数据中',
            'details': {
                'missing_fields': missing_fields,
                'available_fields': available_fields
            }
        }

    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行图表任务"""
        chart_type = self.tool_def.config.get('chart_type', 'bar')
        max_data_points = self.tool_def.config.get('max_data_points', 1000)
        
        # 🎯 处理数据：检查是否需要从步骤引用获取数据
        mcp_args = arguments or {}
        
        # 如果LLM使用了data_from_step引用，需要解析实际数据
        if 'data_from_step' in mcp_args and 'data' not in mcp_args:
            step_data = self._attach_data_from_context(mcp_args.copy(), context, max_data_points)
            if 'data' in step_data:
                actual_data = step_data['data']
                mcp_args['data'] = actual_data
                mcp_args.pop('data_from_step', None)
                
                # 🔍 DEBUG: 输出数据结构信息
                print(f"  📊 [ChartToolHandler] 实际数据结构调试:")
                if isinstance(actual_data, list) and len(actual_data) > 0:
                    sample_row = actual_data[0]
                    if isinstance(sample_row, dict):
                        available_fields = list(sample_row.keys())
                        print(f"    ✅ 可用字段: {available_fields}")
                        print(f"    📋 数据样本: {sample_row}")
                    else:
                        print(f"    ⚠️ 数据行不是字典格式: {type(sample_row)}")
                else:
                    print(f"    ⚠️ 数据格式异常: {type(actual_data)}, length: {len(actual_data) if hasattr(actual_data, '__len__') else 'N/A'}")
                
                # 🔍 DEBUG: 输出LLM请求的encoding字段
                if 'encoding' in mcp_args:
                    encoding = mcp_args['encoding']
                    print(f"    🎯 LLM请求的encoding字段:")
                    for axis, field_info in encoding.items():
                        if isinstance(field_info, dict) and 'field' in field_info:
                            requested_field = field_info['field']
                            print(f"      {axis}: '{requested_field}' ({field_info.get('type', 'unknown')})")
                print(f"    📤 最终发送给MCP的参数keys: {list(mcp_args.keys())}")

        # Guardrail：图表字段校验，避免把错误字段发送给图表MCP工具
        validate_error = self._validate_encoding_fields(mcp_args)
        if validate_error:
            validate_error['step_id'] = step_id
            return validate_error
        
        # 通过 MCP 工具执行（强制要求，不再回退本地）
        sql_session = getattr(self.executor_context, 'sql_session', None)
        
        if not self.mcp_tool_name:
            return {
                'success': False,
                'error_type': 'tool_not_mapped',
                'error': f"未发现图表MCP工具映射: {self.tool_def.name}",
                'step_id': step_id
            }
        if not sql_session:
            return {
                'success': False,
                'error_type': 'no_session',
                'error': 'MCP 会话未就绪，无法调用图表工具',
                'step_id': step_id
            }
        
        try:
            result = await sql_session.call_tool(self.mcp_tool_name, mcp_args)
            parsed = self._parse_chart_result(result, fallback_payload={'chart_type': chart_type, 'dsl': {}})
            return {
                'success': True,
                'task_type': 'chart',
                'data': parsed,
                'used_tool': self.mcp_tool_name
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': 'execution_error',
                'error': str(e),
                'step_id': step_id
            }
    
    def _normalize_chart_dsl(self, hints: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """图表DSL规范化 - 支持25种图表类型"""
        hints = dict(hints or {})
        standard = {
            'chart_type': chart_type,
            'tool_name': self.tool_def.name
        }
        
        # 图表类型到标准格式的映射
        chart_specs = {
            # 基础图表
            'bar': {'mark': 'bar', 'requires': ['x', 'y']},
            'column': {'mark': 'column', 'requires': ['x', 'y']},
            'line': {'mark': 'line', 'requires': ['x', 'y']},
            'area': {'mark': 'area', 'requires': ['x', 'y']},
            'pie': {'mark': 'pie', 'requires': ['label', 'value']},
            'scatter': {'mark': 'scatter', 'requires': ['x', 'y']},
            
            # 统计图表
            'histogram': {'mark': 'histogram', 'requires': ['x']},
            'boxplot': {'mark': 'boxplot', 'requires': ['x', 'y']},
            'violin': {'mark': 'violin', 'requires': ['x', 'y']},
            
            # 多维图表
            'radar': {'mark': 'radar', 'requires': ['dimensions', 'values']},
            'dual_axes': {'mark': 'dual_axes', 'requires': ['x', 'y1', 'y2']},
            
            # 层次图表
            'treemap': {'mark': 'treemap', 'requires': ['hierarchy', 'value']},
            'sankey': {'mark': 'sankey', 'requires': ['source', 'target', 'value']},
            'funnel': {'mark': 'funnel', 'requires': ['stage', 'value']},
            
            # 特殊图表
            'liquid': {'mark': 'liquid', 'requires': ['value', 'max']},
            'word_cloud': {'mark': 'word_cloud', 'requires': ['words', 'frequency']},
            
            # 关系图表
            'network': {'mark': 'network', 'requires': ['nodes', 'edges']},
            'venn': {'mark': 'venn', 'requires': ['sets', 'overlaps']},
            
            # 流程图表
            'flow': {'mark': 'flow', 'requires': ['nodes', 'connections']},
            'mind_map': {'mark': 'mind_map', 'requires': ['central_topic', 'branches']},
            'org_chart': {'mark': 'org_chart', 'requires': ['hierarchy', 'positions']},
            'fishbone': {'mark': 'fishbone', 'requires': ['problem', 'causes']},
            
            # 地图图表
            'district_map': {'mark': 'district_map', 'requires': ['regions', 'values']},
            'pin_map': {'mark': 'pin_map', 'requires': ['locations', 'coordinates']},
            'path_map': {'mark': 'path_map', 'requires': ['routes', 'waypoints']}
        }
        
        spec = chart_specs.get(chart_type, {'mark': chart_type, 'requires': ['x', 'y']})
        standard['mark'] = spec['mark']
        standard['required_fields'] = spec['requires']
        
        # 处理编码信息
        encoding = hints.get('encoding', {})
        if not encoding:
            # 如果有chart_data字段，使用其配置
            if 'chart_data' in hints:
                chart_data = hints['chart_data']
                # 根据图表类型生成默认编码（使用MCP schema中的实际字段名）
                if chart_type in ['bar', 'column', 'line', 'area', 'scatter']:
                    encoding = {
                        'x': {'field': chart_data.get('xField', 'category')},  # MCP期望category字段
                        'y': {'field': chart_data.get('yField', 'value')}      # MCP期望value字段
                    }
                elif chart_type == 'pie':
                    encoding = {
                        'label': {'field': chart_data.get('labelField', 'label')},
                        'value': {'field': chart_data.get('valueField', 'value')}
                    }
            # 如果直接有data字段，生成对应的默认编码
            elif 'data' in hints:
                if chart_type in ['bar', 'column', 'line', 'area', 'scatter']:
                    encoding = {
                        'x': {'field': 'category'},  # MCP schema期望的字段名
                        'y': {'field': 'value'}      # MCP schema期望的字段名
                    }
                elif chart_type == 'pie':
                    encoding = {
                        'label': {'field': 'label'},
                        'value': {'field': 'value'}
                    }
        
        standard['encoding'] = encoding
        
        # 处理图表特定配置
        if chart_type in ['word_cloud']:
            standard['text_config'] = hints.get('text_config', {})
        elif chart_type in ['network', 'mind_map', 'org_chart']:
            standard['layout_config'] = hints.get('layout_config', {})
        elif chart_type in ['district_map', 'pin_map', 'path_map']:
            standard['map_config'] = hints.get('map_config', {})
        
        # 透传通用字段
        for key in ['title', 'data', 'data_from_step', 'chart_data', 'subtitle', 'description']:
            if key in hints:
                standard[key] = hints[key]
                
        return standard
        
    def _attach_data_from_context(self, dsl: Dict[str, Any], shared_storage: Dict[str, Any], max_data_points: int = 1000) -> Dict[str, Any]:
        """从共享存储附加数据"""
        result_dsl = dict(dsl)
        
        # 如果已经有数据，限制数据量后返回
        if 'data' in result_dsl and result_dsl['data']:
            if isinstance(result_dsl['data'], list):
                result_dsl['data'] = result_dsl['data'][:max_data_points]
            return result_dsl
        
        # 从data_from_step引用数据
        data_from_step = result_dsl.get('data_from_step')
        if data_from_step and shared_storage:
            step_key = f"{data_from_step}_result"
            if step_key in shared_storage:
                step_result = shared_storage[step_key]
                if isinstance(step_result, dict):
                    # 兼容两种存储格式：顶层 query_data 或 data.query_data
                    query_data = []
                    if 'query_data' in step_result and isinstance(step_result['query_data'], list):
                        query_data = step_result['query_data']
                    else:
                        query_data = step_result.get('data', {}).get('query_data', [])
                    if isinstance(query_data, list) and len(query_data) > 0:
                        result_dsl['data'] = query_data[:max_data_points]
        
        # 如果仍然没有数据，生成示例数据（用于演示）
        if 'data' not in result_dsl or not result_dsl['data']:
            chart_type = dsl.get('chart_type', 'bar')
            result_dsl['data'] = self._generate_sample_data(chart_type, min(10, max_data_points))
        
        return result_dsl
    
    def _parse_chart_result(self, result: Any, fallback_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """解析 MCP 图表结果，尽可能标准化。"""
        payload: Dict[str, Any] = {
            'chart_type': (fallback_payload or {}).get('chart_type'),
            'dsl': (fallback_payload or {}).get('dsl'),
            'structured_content': None
        }
        try:
            # 常见 fastmcp CallToolResult 结构
            if str(type(result)).find("CallToolResult") != -1:
                content = getattr(result, 'content', None)
                if content and len(content) > 0 and hasattr(content[0], 'text'):
                    text_data = content[0].text
                    if isinstance(text_data, str):
                        try:
                            parsed = json.loads(text_data)
                            # 期望远端返回 { url | dsl | image | structured_content }
                            if isinstance(parsed, dict):
                                # 优先 structured_content
                                if 'structured_content' in parsed:
                                    payload['structured_content'] = parsed['structured_content']
                                else:
                                    # 将远端返回封装到 structured_content 中，类型为 'chart_render'
                                    payload['structured_content'] = {
                                        'type': 'chart_render',
                                        'payload': parsed
                                    }
                                return payload
                        except json.JSONDecodeError:
                            # 非 JSON，作为纯文本封装
                            payload['structured_content'] = {
                                'type': 'chart_render',
                                'payload': {'text': text_data}
                            }
                            return payload
            # 其他返回结构尽量序列化
            if hasattr(result, 'model_dump'):
                payload['structured_content'] = {
                    'type': 'chart_render',
                    'payload': result.model_dump()
                }
            elif hasattr(result, 'dict'):
                payload['structured_content'] = {
                    'type': 'chart_render',
                    'payload': result.dict()
                }
            else:
                payload['structured_content'] = {
                    'type': 'chart_render',
                    'payload': {'raw_result': str(result)}
                }
        except Exception as e:
            payload['structured_content'] = {
                'type': 'chart_render_error',
                'error': str(e)
            }
        return payload

    def _generate_sample_data(self, chart_type: str, count: int = 10) -> List[Dict[str, Any]]:
        """为不同图表类型生成示例数据"""
        import random
        
        if chart_type in ['bar', 'column', 'line', 'area']:
            return [
                {'x': f'Category {i+1}', 'y': random.randint(10, 100)}
                for i in range(count)
            ]
        elif chart_type == 'pie':
            return [
                {'label': f'Segment {i+1}', 'value': random.randint(10, 50)}
                for i in range(min(count, 8))
            ]
        elif chart_type == 'scatter':
            return [
                {'x': random.randint(1, 100), 'y': random.randint(1, 100)}
                for i in range(count)
            ]
        elif chart_type == 'histogram':
            return [
                {'value': random.gauss(50, 15)}
                for i in range(count * 10)
            ]
        elif chart_type in ['boxplot', 'violin']:
            return [
                {'category': f'Group {(i//10)+1}', 'value': random.gauss(50, 15)}
                for i in range(count * 10)
            ]
        elif chart_type == 'word_cloud':
            words = ['data', 'analysis', 'chart', 'visualization', 'insight', 'trend', 'pattern', 'discovery']
            return [
                {'word': word, 'frequency': random.randint(5, 50)}
                for word in words[:count]
            ]
        elif chart_type == 'funnel':
            stages = ['Awareness', 'Interest', 'Consideration', 'Purchase', 'Retention']
            return [
                {'stage': stage, 'value': 1000 - i*150}
                for i, stage in enumerate(stages[:count])
            ]
        else:
            # 默认返回简单的x-y数据
            return [
                {'x': f'Item {i+1}', 'y': random.randint(10, 100)}
                for i in range(count)
            ]

class FileToolHandler(BaseToolHandler):
    """文件处理工具处理器"""
    
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行文件处理任务"""
        print(f"  📁 [File Handler] 处理文件操作...")
        
        if not self.mcp_tool_name:
            # 如果没有MCP工具，返回模拟结果
            return {
                'success': True,
                'task_type': 'file',
                'data': {
                    'message': f"文件处理任务: {instruction}",
                    'structured_content': {
                        'type': 'file_result',
                        'content': "文件处理完成（模拟）"
                    }
                },
                'used_tool': 'file_processor'
            }
        
        # 实际MCP调用逻辑
        mcp_session = self.executor_context.sql_session  # 复用连接
        result = await mcp_session.call_tool(self.mcp_tool_name, arguments or {})
        
        return {
            'success': True,
            'task_type': 'file',
            'data': self._parse_file_result(result),
            'used_tool': self.mcp_tool_name
        }
    
    def _parse_file_result(self, result: Any) -> Dict[str, Any]:
        """解析文件处理结果"""
        return {'raw_result': str(result)}

class WebToolHandler(BaseToolHandler):
    """Web抓取工具处理器"""
    
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行Web抓取任务"""
        print(f"  🌐 [Web Handler] 执行Web抓取...")
        
        # 模拟Web抓取
        url = (arguments or {}).get('url', 'https://example.com')
        
        return {
            'success': True,
            'task_type': 'web',
            'data': {
                'url': url,
                'content': f"从 {url} 抓取的内容（模拟）",
                'structured_content': {
                    'type': 'web_content',
                    'content': f"Web抓取完成: {instruction}"
                }
            },
            'used_tool': 'web_scraper'
        }

class AnalysisToolHandler(BaseToolHandler):
    """数据分析工具处理器"""
    
    async def execute(self, step_id: str, instruction: str, context: Dict[str, Any], 
                     arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行数据分析任务"""
        print(f"  🔬 [Analysis Handler] 执行数据分析...")
        
        # 从共享存储获取数据进行分析
        shared_storage = (context or {}).get('shared_storage', {})
        analysis_type = (arguments or {}).get('type', 'correlation')
        
        return {
            'success': True,
            'task_type': 'analysis',
            'data': {
                'analysis_type': analysis_type,
                'results': f"数据分析结果: {analysis_type} 分析完成",
                'structured_content': {
                    'type': 'analysis_result',
                    'content': f"分析完成: {instruction}"
                }
            },
            'used_tool': 'data_analyzer'
        }



class ToolRegistry:
    """动态工具注册表 - 从MCP服务器获取工具信息"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.tools: Dict[str, ToolDefinition] = {}
        self.handlers: Dict[str, BaseToolHandler] = {}
        self.mcp_mappings: Dict[str, str] = {}  # MCP工具名 -> 注册工具名
        self.tool_parameters: Dict[str, Dict[str, Any]] = {}  # 工具名 -> 参数schema
        # 维护category到Handler类的映射
        self.category_handler_map = {
            'sql': SqlToolHandler,
            'chart': ChartToolHandler,
            'file': FileToolHandler,
            'web': WebToolHandler,
            'analysis': AnalysisToolHandler,
        }
        
    def register_tool(self, tool_def: ToolDefinition, handler: BaseToolHandler):
        """注册工具"""
        self.tools[tool_def.name] = tool_def
        self.handlers[tool_def.name] = handler
    
    async def register_from_mcp_server(self, mcp_session) -> int:
        """从MCP服务器动态注册工具"""
        discovered_count = 0
        
        try:
            mcp_tools = await mcp_session.list_tools()
            
            for mcp_tool in mcp_tools:
                try:
                    # 解析MCP工具信息
                    tool_info = self._parse_mcp_tool(mcp_tool)
                    if not tool_info:
                        continue
                    
                    # 创建工具定义
                    tool_def = ToolDefinition(
                        name=tool_info['name'],
                        category=tool_info['category'],
                        description=tool_info['description'],
                        enabled=True,
                        priority=tool_info.get('priority', 100),
                        config=tool_info.get('config', {})
                    )
                    
                    # 创建对应的Handler
                    handler = self._create_handler(tool_def)
                    if handler:
                        # 设置MCP工具名称
                        handler.set_mcp_tool_name(tool_info['name'])
                        
                        # 注册工具
                        self.register_tool(tool_def, handler)
                        
                        # 保存参数信息
                        if 'parameters' in tool_info:
                            self.tool_parameters[tool_info['name']] = tool_info['parameters']
                        
                        # 建立映射
                        self.mcp_mappings[tool_info['name']] = tool_info['name']
                        
                        discovered_count += 1
                    
                except Exception as e:
                    print(f"  ⚠️ [ToolRegistry] 解析工具失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"  ❌ [ToolRegistry] MCP工具发现失败: {e}")
            
        return discovered_count
    
    def register_fallback_tools(self):
        """注册基本的fallback工具，确保即使MCP服务器不可用也有基本功能"""
        
        fallback_tools = [
            {
                'name': 'sql_query',
                'category': 'sql',
                'description': '执行SQL查询（自然语言描述）',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "natural_language": {"type": "string"}
                    },
                    "required": ["natural_language"]
                },
                'config': {},
                'priority': 10
            },
            {
                'name': 'generate_bar_chart',
                'category': 'chart',
                'description': '生成柱状图',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "data_from_step": {"type": "string"},
                        "encoding": {"type": "object"}
                    },
                    "required": []
                },
                'config': {'chart_type': 'bar', 'max_data_points': 1000},
                'priority': 20
            }
        ]
        
        fallback_count = 0
        for tool_info in fallback_tools:
            try:
                # 创建工具定义
                tool_def = ToolDefinition(
                    name=tool_info['name'],
                    category=tool_info['category'],
                    description=tool_info['description'],
                    enabled=True,
                    priority=tool_info.get('priority', 100),
                    config=tool_info.get('config', {})
                )
                
                # 创建对应的Handler
                handler = self._create_handler(tool_def)
                if handler:
                    # 注册工具
                    self.register_tool(tool_def, handler)
                    
                    # 保存参数信息
                    if 'parameters' in tool_info:
                        self.tool_parameters[tool_info['name']] = tool_info['parameters']
                    
                    # 建立映射（fallback工具没有真实的MCP名称）
                    self.mcp_mappings[tool_info['name']] = tool_info['name']
                    
                    fallback_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ 注册fallback工具失败 {tool_info['name']}: {e}")
                continue
        
        return fallback_count
    
    def _parse_mcp_tool(self, mcp_tool) -> Optional[Dict[str, Any]]:
        """解析MCP工具信息，返回标准化的工具信息字典"""
        try:
            tool_name = mcp_tool.name if hasattr(mcp_tool, 'name') else str(mcp_tool)
            description = getattr(mcp_tool, 'description', '') or ''
            
            # 从工具名称和描述推断category
            category = self._infer_tool_category(tool_name, description)
            if not category:
                return None
            
            # 解析参数信息
            parameters = {}
            if hasattr(mcp_tool, 'inputSchema'):
                try:
                    # MCP工具的inputSchema通常是一个JSON Schema
                    schema = mcp_tool.inputSchema
                    if hasattr(schema, 'properties'):
                        # 转换为OpenAI tools格式
                        parameters = {
                            "type": "object",
                            "properties": schema.properties,
                            "required": getattr(schema, 'required', [])
                        }
                    elif isinstance(schema, dict):
                        parameters = schema
                except Exception as e:
                    print(f"  ⚠️ [ToolRegistry] 解析参数失败 {tool_name}: {e}")
            
            # 根据工具类型设置特殊配置
            config = {}
            if category == 'chart':
                # 从工具名称推断图表类型
                chart_type = self._infer_chart_type(tool_name)
                config = {
                    'chart_type': chart_type,
                    'max_data_points': 1000
                }
            
            return {
                'name': tool_name,
                'category': category,
                'description': description,
                'parameters': parameters,
                'config': config,
                'priority': self._get_tool_priority(category)
            }
            
        except Exception as e:
            print(f"  ❌ [ToolRegistry] 解析MCP工具失败: {e}")
            return None
    
    def _infer_tool_category(self, tool_name: str, description: str) -> Optional[str]:
        """根据工具名称和描述推断工具类型"""
        name_lower = tool_name.lower()
        desc_lower = description.lower()
        
        # SQL相关
        if 'sql' in name_lower or 'query' in name_lower:
            return 'sql'
        if 'sql' in desc_lower or 'database' in desc_lower or 'query' in desc_lower:
            return 'sql'
            
        # 图表相关
        if name_lower.startswith('generate_') and ('chart' in name_lower or 'plot' in name_lower):
            return 'chart'
        if 'chart' in desc_lower or 'visualization' in desc_lower or 'plot' in desc_lower:
            return 'chart'
            
        # 文件相关
        if 'file' in name_lower or 'read' in name_lower or 'write' in name_lower:
            return 'file'
        if 'file' in desc_lower or 'document' in desc_lower:
            return 'file'
            
        # Web相关
        if 'web' in name_lower or 'http' in name_lower or 'url' in name_lower:
            return 'web'
        if 'web' in desc_lower or 'scrape' in desc_lower or 'crawl' in desc_lower:
            return 'web'
            
        # 分析相关
        if 'analysis' in name_lower or 'analyze' in name_lower:
            return 'analysis'
        if 'analysis' in desc_lower or 'statistics' in desc_lower:
            return 'analysis'
            
        return None
    
    def _infer_chart_type(self, tool_name: str) -> str:
        """从工具名称推断图表类型"""
        name_lower = tool_name.lower()
        
        # 移除generate_前缀
        if name_lower.startswith('generate_'):
            name_lower = name_lower[9:]
        
        # 移除_chart/_plot后缀
        for suffix in ['_chart', '_plot', '_graph', '_diagram']:
            if name_lower.endswith(suffix):
                name_lower = name_lower[:-len(suffix)]
                break
                
        # 常见图表类型映射
        chart_type_map = {
            'bar': 'bar',
            'column': 'column', 
            'line': 'line',
            'area': 'area',
            'pie': 'pie',
            'scatter': 'scatter',
            'histogram': 'histogram',
            'boxplot': 'boxplot',
            'violin': 'violin',
            'radar': 'radar',
            'treemap': 'treemap',
            'sankey': 'sankey',
            'funnel': 'funnel',
            'liquid': 'liquid',
            'word_cloud': 'word_cloud',
            'network': 'network',
            'venn': 'venn',
            'flow': 'flow',
            'mind_map': 'mind_map',
            'org': 'org_chart',
            'fishbone': 'fishbone',
            'district_map': 'district_map',
            'pin_map': 'pin_map',
            'path_map': 'path_map'
        }
        
        return chart_type_map.get(name_lower, 'bar')  # 默认为bar图
    
    def _get_tool_priority(self, category: str) -> int:
        """根据工具类型返回默认优先级"""
        priority_map = {
            'sql': 10,
            'chart': 20,
            'file': 30,
            'web': 40,
            'analysis': 50
        }
        return priority_map.get(category, 100)
    
    def _create_handler(self, tool_def: ToolDefinition) -> Optional[BaseToolHandler]:
        """根据工具定义创建处理器"""
        handler_class = self.category_handler_map.get(tool_def.category)
        if not handler_class:
            return None
            
        return handler_class(tool_def)
    
    def get_all_tools_spec(self) -> List[Dict[str, Any]]:
        """获取所有工具的OpenAI tools格式规格，供LLM使用"""
        tools_spec = []
        
        for tool_name, tool_def in self.tools.items():
            if not tool_def.enabled:
                continue
                
            # 获取工具参数
            parameters = self.tool_parameters.get(tool_name, {
                "type": "object",
                "properties": {},
                "required": []
            })
            
            # 构建OpenAI tools格式
            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_def.description,
                    "parameters": parameters
                }
            }
            
            tools_spec.append(tool_spec)
            
        return tools_spec
    
    def get_handler(self, tool_name: str) -> Optional[BaseToolHandler]:
        """获取工具处理器"""
        return self.handlers.get(tool_name)
    
    def get_tool_by_mcp_name(self, mcp_name: str) -> Optional[str]:
        """通过MCP名称获取工具名"""
        return self.mcp_mappings.get(mcp_name)
    
    def list_enabled_tools(self) -> List[str]:
        """列出启用的工具"""
        return [name for name, tool_def in self.tools.items() if tool_def.enabled]
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """按分类获取工具"""
        return [name for name, tool_def in self.tools.items() 
                if tool_def.category == category and tool_def.enabled]

# 全局工具注册表实例
default_registry = ToolRegistry()
