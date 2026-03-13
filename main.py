# ==========================================================
# 文件: main.py (V14 - 简化MCP专用版)
# 职责: 演示简化后的MCP工具调用流程
# ==========================================================

import json
import sys
import os
import asyncio

# 动态调整Python的搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的模块
from planner.planner import TaskExecutorPlanner
from planner.llm_setup import QuickLLMAPI
from planner.utils import get_workflow_from_api
async def main():
    """主函数"""
    print("--- 🚀 [Main] 系统启动 ---")
    
    # 🔍 调试模式检查
    debug_mode = os.getenv('DEBUG_MODE', '').lower() in ['true', '1', 'yes']
    if debug_mode:
        print("🔍 [Main] 调试模式已开启")
        # 设置更详细的日志级别
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # MCP服务器配置 - 支持多个服务器
    mcp_servers = [
        os.getenv('SQL_EXECUTOR_URL', 'http://172.16.1.114:8000/mcp/'),  # SQL服务器
        os.getenv('CHART_SERVER_URL', 'http://172.16.1.114:1122/mcp')  # 图表服务器
    ]
    
    # 过滤掉空的URL
    mcp_servers = [url for url in mcp_servers if url and url.strip()]
    print(f"🔗 [Main] 配置的MCP服务器: {mcp_servers}")
    
    kimi_api_key = os.getenv('MOONSHOT_API_KEY')
    if kimi_api_key:
        print("✅ [Main] 检测到 MOONSHOT_API_KEY 环境变量")
    else:
        print("⚠️ [Main] 未检测到 MOONSHOT_API_KEY 环境变量，LLM 初始化可能失败")
    
    # 使用简化的MCP专用Planner（仅支持MCP工具调用）
    llm_api = QuickLLMAPI().api  # Planner使用的LLM API
    
    # 创建增强执行器来初始化工具注册表
    from planner.enhanced_executor import EnhancedMcpExecutor
    executor = EnhancedMcpExecutor(mcp_servers)
    connected = await executor.connect()  # 这会初始化工具注册表
    
    if not connected:
        print("❌ [Main] 无法连接到任何MCP服务器，程序退出")
        return
    
    # 创建Planner并传入工具注册表和增强执行器
    planner = TaskExecutorPlanner(llm_api=llm_api, tool_registry=executor.tool_registry, enhanced_executor=executor)
    print("✅ [Main] 多服务器MCP Planner 已准备就绪。")

    # --- 执行工作流 ---
    try:
        # MCP模式：Planner内部使用MCP工具调用

        user_query = "请你分析缺陷和屏幕的空间聚集性"
        print(f"➡️  [Main] 步骤 2: 准备获取工作流。查询: '{user_query}'")
        workflow = get_workflow_from_api(user_query)
        if workflow:
            print("✅ [Main] 步骤 2: 工作流已成功获取并解析。")
        else:
            print("❌ [Main] 步骤 2: 工作流获取/解析失败。")

        if not workflow:
            print("🛑 [Main] 无法从 API 获取工作流，任务终止。")
            return
            
        print("\n📋 [Main] 获取到的原始工作流计划如下:")
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        print("-" * 40)

        task_input = { "query": user_query, "workflow": workflow }
        
        print(f"\n➡️  [Main] 步骤 3: 准备将任务交由简化MCP Planner执行...")
        final_report = await planner.execute_workflow(task_input)
        print("✅ [Main] 步骤 3: 简化MCP Planner已完成工作流执行。")

        print("\n\n--- 🎉 任务执行完毕 (简化MCP模式) ---")
        print(final_report)

    except Exception as e:
        print(f"\n🔥🔥🔥 [Main] 在主流程中捕获到未处理的异常: {e}")
    finally:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
