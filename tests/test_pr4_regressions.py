import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("metagen_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package

def load(name):
    spec = importlib.util.spec_from_file_location(f"metagen_test.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

runtime = load("agent_runtime")
# No network or installed MCP client is required for these connection tests.
fastmcp = types.ModuleType("fastmcp")
fastmcp.Client = None
with patch.dict(sys.modules, {"fastmcp": fastmcp}):
    executor = load("enhanced_executor")


class ChartTests(unittest.IsolatedAsyncioTestCase):
    async def test_requested_chart_reaches_executor(self):
        cases = {
            "Create a line chart": "generate_line_chart",
            "绘制折线图": "generate_line_chart",
            "Create a PIE chart": "generate_pie_chart",
            "绘制饼图": "generate_pie_chart",
            "Create a scatter plot": "generate_scatter_chart",
            "绘制散点图": "generate_scatter_chart",
            "Create a bar chart": "generate_bar_chart",
            "绘制柱状图": "generate_column_chart",
            "Use generate_scatter_plot": "generate_scatter_plot",
            "Visualize pipeline totals in a chart": "generate_column_chart",
        }
        for instruction, expected in cases.items():
            with self.subTest(instruction=instruction):
                calls = []
                class FakeExecutor:
                    async def execute(self, *args):
                        calls.append(args)
                        return {"success": True, "data": [{"value": 2}]}
                runner = runtime.AgentRuntimeOrchestrator(FakeExecutor())
                await runner.run("Analyze sales", {"step1": "Query sales", "step2": instruction})
                self.assertEqual(calls[0][3], "sql_query")
                self.assertEqual(calls[1][3], expected)
                self.assertEqual(calls[1][4]["data_from_step"], "step1")
                self.assertEqual(calls[1][1], instruction)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_backoff_yields_and_exhaustion_falls_back(self):
        attempts, delays, ticks = [], [], []
        class FailingClient:
            def __init__(self, url):
                attempts.append(url)
            async def __aenter__(self):
                raise ConnectionError("unavailable")
            async def __aexit__(self, *args):
                pass
        real_sleep = asyncio.sleep
        async def sleep(delay):
            delays.append(delay)
            await real_sleep(0)
        async def heartbeat():
            ticks.append(len(attempts))
        worker = executor.EnhancedMcpExecutor(["http://unavailable/mcp"])
        worker.max_connect_retries = 3
        worker.retry_base_seconds = 0.5
        with patch.object(executor, "Client", FailingClient), patch.object(executor.asyncio, "sleep", sleep), patch.object(executor.time, "sleep", side_effect=AssertionError("blocking sleep")):
            result, _ = await asyncio.gather(worker.connect(), heartbeat())
        self.assertTrue(result)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertEqual(ticks, [1])
        self.assertTrue(worker.tool_registry.tools)

    async def test_cancellation_interrupts_backoff(self):
        entered = asyncio.Event()
        class FailingClient:
            def __init__(self, url):
                pass
            async def __aenter__(self):
                raise ConnectionError("unavailable")
            async def __aexit__(self, *args):
                pass
        async def sleep(delay):
            entered.set()
            await asyncio.Future()
        worker = executor.EnhancedMcpExecutor(["http://unavailable/mcp"])
        worker.max_connect_retries = 3
        with patch.object(executor, "Client", FailingClient), patch.object(executor.asyncio, "sleep", sleep):
            task = asyncio.create_task(worker.connect())
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(worker.is_connected)


if __name__ == "__main__":
    unittest.main()
