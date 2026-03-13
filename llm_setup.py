"""
LLM API 统一接口 - Kimi专用版
只支持 Kimi (Moonshot AI) API
"""

import os
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):
    """LLM后端基类"""
    
    @abstractmethod
    def call(self, prompt: str) -> str:
        """简单调用接口"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict], json_schema: Optional[Dict] = None) -> str:
        """聊天完成接口"""
        pass

    # 新增：返回完整响应（用于 tools/function calling）
    def chat_completion_full(self, messages: List[Dict], tools: Optional[List[Dict]] = None, stream: bool = False) -> Dict:
        """聊天完成接口（返回完整响应对象的可序列化形式）。"""
        raise NotImplementedError("当前后端未实现 chat_completion_full 支持")

    # 可选：支持 Responses（原生输出数组与连接器/工具）
    def responses_create(self, *, model: Optional[str] = None, input_text: Optional[str] = None,
                         messages: Optional[List[Dict]] = None, connectors: Optional[List[Dict]] = None,
                         extra_args: Optional[Dict] = None) -> Dict:
        """默认不实现。子类可覆盖以支持 Responses API。
        返回值约定为一个 dict，至少包含 'output'（列表）或完整响应对象的可序列化表示。
        """
        raise NotImplementedError("当前后端未实现 Responses 支持")


class KimiBackend(BaseLLMBackend):
    """Kimi (Moonshot AI) 专用后端"""
    
    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )
            self.model = "kimi-k2-0905-preview"
            # Responses 模式默认与 chat 使用相同模型，允许外部覆写
            self.responses_model = None
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
    
    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    def chat_completion(self, messages: List[Dict], json_schema: Optional[Dict] = None) -> str:
        if json_schema:
            messages = messages.copy()
            messages.append({
                "role": "system",
                "content": f"请严格按照以下JSON Schema格式回复：\n{json.dumps(json_schema, indent=2)}"
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def chat_completion_full(self, messages: List[Dict], tools: Optional[List[Dict]] = None, stream: bool = False) -> Dict:
        """返回完整响应，用于 tools/function-calling。"""
        kwargs: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'stream': stream
        }
        if tools:
            kwargs['tools'] = tools
            # 可选：通过环境变量强制要求模型触发工具调用
            # FORCE_TOOL_CHOICE_REQUIRED=true/1/yes/y/on 时启用
            try:
                force_required = os.getenv('FORCE_TOOL_CHOICE_REQUIRED', '').strip().lower() in ['1', 'true', 'yes', 'y', 'on']
            except Exception:
                force_required = False
            if force_required:
                kwargs['tool_choice'] = 'required'
        resp = self.client.chat.completions.create(**kwargs)
        try:
            if hasattr(resp, 'model_dump'):
                return resp.model_dump()
            if hasattr(resp, 'to_dict'):
                return resp.to_dict()
            return json.loads(json.dumps(resp, default=lambda o: getattr(o, '__dict__', str(o))))
        except Exception:
            return { 'raw': str(resp) }

    def responses_create(self, *, model: Optional[str] = None, input_text: Optional[str] = None,
                         messages: Optional[List[Dict]] = None, connectors: Optional[List[Dict]] = None,
                         extra_args: Optional[Dict] = None) -> Dict:
        """Kimi Responses API 包装（用于原生 MCP 连接器）。
        - 优先使用 messages，其次使用 input_text。
        - connectors 按原样透传给底层（若支持）。
        - 返回尽量转成内置 dict，包含 'output'（若可用）；否则返回可序列化整体。
        """
        # 动态检测 client 是否具备 responses 接口
        responses_client = getattr(self.client, 'responses', None)
        if responses_client is None or not hasattr(responses_client, 'create'):
            raise RuntimeError("Kimi 客户端不支持 responses 接口，请升级 SDK 或关闭 USE_MCP_NATIVE 模式")

        create_kwargs: Dict[str, Any] = {}
        create_kwargs['model'] = model or self.responses_model or self.model

        if messages:
            # 一些 SDK 支持以 messages 形式输入
            create_kwargs['messages'] = messages
        elif input_text is not None:
            # 也有实现用 input (单段文本)
            create_kwargs['input'] = input_text

        if connectors:
            create_kwargs['connectors'] = connectors

        if extra_args:
            create_kwargs.update(extra_args)

        resp = responses_client.create(**create_kwargs)

        # 转为 python dict，兼容不同 SDK 对象
        try:
            # OpenAI 风格对象通常有 model_dump_json / model_dump
            if hasattr(resp, 'model_dump'):
                data = resp.model_dump()
            elif hasattr(resp, 'to_dict'):
                data = resp.to_dict()
            else:
                # 尝试 json 序列化回退
                data = json.loads(json.dumps(resp, default=lambda o: getattr(o, '__dict__', str(o))))
        except Exception:
            data = { 'raw': str(resp) }

        # 规范 output 字段（若 SDK 已有则透传）
        if isinstance(data, dict) and 'output' in data:
            return data
        # 一些实现使用 outputs / content 等字段
        if isinstance(data, dict) and 'outputs' in data:
            return { **data, 'output': data.get('outputs') }
        if isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
            return { **data, 'output': data['content'] }
        return data


class QuickLLMAPI:
    """统一的LLM API接口 - Kimi专用版"""
    
    def __init__(self):
        self.backend = "kimi"
        self.api = None
        self._setup_kimi()
    
    def _setup_kimi(self):
        """设置Kimi API"""
        api_key = os.getenv('MOONSHOT_API_KEY')
        if not api_key:
            raise ValueError(
                "未找到Kimi API Key！\n"
                "请设置环境变量：export MOONSHOT_API_KEY=your_kimi_api_key"
            )
        
        try:
            self.api = KimiBackend(api_key=api_key)
            print("✅ 使用 Kimi (Moonshot AI) API")
        except Exception as e:
            raise RuntimeError(f"Kimi API 初始化失败: {e}")
    
    def call(self, prompt: str) -> str:
        """简单调用接口"""
        return self.api.call(prompt)
    
    def chat_completion(self, messages: List[Dict], json_schema: Optional[Dict] = None) -> str:
        """聊天完成接口"""
        return self.api.chat_completion(messages, json_schema)

    def chat_completion_full(self, messages: List[Dict], tools: Optional[List[Dict]] = None, stream: bool = False) -> Dict:
        """聊天完成接口（返回完整响应对象）。"""
        return self.api.chat_completion_full(messages, tools, stream)

    def responses_create(self, *, model: Optional[str] = None, input_text: Optional[str] = None,
                         messages: Optional[List[Dict]] = None, connectors: Optional[List[Dict]] = None,
                         extra_args: Optional[Dict] = None) -> Dict:
        """Responses 接口（支持原生 MCP 连接器）。"""
        return self.api.responses_create(model=model, input_text=input_text, messages=messages,
                                         connectors=connectors, extra_args=extra_args)