import os
import json
import re
from typing import Any, Dict, Optional


def normalize_to_workflow(flows: Any) -> Dict[str, str]:
    """将外部返回的多形态工作流数据规范化为 {stepN: instruction}。"""
    combined: Dict[str, str] = {}
    collected_instructions: list[str] = []

    def split_multi_step_text(text: str) -> list[str]:
        parts = re.findall(r"\[STEP\s*\d+\]:\s*(.*?)(?=\[STEP\s*\d+\]:|$)", text, re.DOTALL)
        return [p.strip() for p in parts if isinstance(p, str) and p.strip()]

    def add_instruction(item: Any) -> None:
        if item is None:
            return
        s = str(item).strip()
        if not s:
            return
        if "[STEP" in s:
            for seg in split_multi_step_text(s):
                collected_instructions.append(seg)
        else:
            collected_instructions.append(s)

    def handle(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            s = obj.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    handle(parsed)
                    return
                except Exception:
                    pass
            add_instruction(s)
            return
        if isinstance(obj, dict):
            workflows_obj = obj.get("workflows") if isinstance(obj.get("workflows"), dict) else None
            if workflows_obj is not None:
                def wf_key(k: str):
                    m = re.search(r"(\d+)", k)
                    return (0, int(m.group(1))) if m else (1, k)

                for k in sorted(workflows_obj.keys(), key=wf_key):
                    handle(workflows_obj[k])
                return

            def step_key(k: str):
                m = re.fullmatch(r"step(\d+)", k.strip().lower())
                return (0, int(m.group(1))) if m else (1, k)

            for k in sorted(obj.keys(), key=step_key):
                handle(obj[k])
            return
        if isinstance(obj, list):
            for item in obj:
                handle(item)
            return
        add_instruction(obj)

    if isinstance(flows, str):
        s = flows.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                flows = json.loads(s)
            except Exception:
                pass

    handle(flows)

    for i, instr in enumerate(collected_instructions, 1):
        combined[f"step{i}"] = instr

    return combined


def get_workflow_from_api(content: str, api_url: Optional[str] = None, timeout: int = 300) -> Optional[Dict[str, str]]:
    """调用外部 API 生成工作流（健壮解析）。"""
    import requests

    workflow_url = api_url or os.getenv("WORKFLOW_GENERATOR_API_URL", "http://localhost:8150/generate_workflow")
    try:
        resp = requests.post(
            workflow_url,
            json={"query": content},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        resp.raise_for_status()
        try:
            resp_json = resp.json()
        except json.JSONDecodeError:
            return None

        data = resp_json.get("data")

        def contains_step_like_keys(obj: Any) -> bool:
            if not isinstance(obj, dict):
                return False
            for k in obj.keys():
                if isinstance(k, str) and k.lower().startswith("step"):
                    m = re.fullmatch(r"step\d+", k.lower().strip())
                    if m:
                        return True
            return False

        flows_candidate: Any = None
        if isinstance(data, dict):
            flows_candidate = (
                data.get("workflows")
                or data.get("workflow")
                or data.get("steps")
                or data.get("subquery")
            )
            if flows_candidate is None and contains_step_like_keys(data):
                flows_candidate = data
            if flows_candidate is None and isinstance(data.get("subquery"), dict) and contains_step_like_keys(data["subquery"]):
                flows_candidate = data["subquery"]
        elif isinstance(data, (list, str)):
            flows_candidate = data

        if flows_candidate is None:
            flows_candidate = (
                resp_json.get("workflows")
                or resp_json.get("workflow")
                or resp_json.get("steps")
                or resp_json.get("subquery")
            )
        if flows_candidate is None and contains_step_like_keys(resp_json):
            flows_candidate = resp_json
        if flows_candidate is None and isinstance(resp_json.get("subquery"), dict) and contains_step_like_keys(resp_json["subquery"]):
            flows_candidate = resp_json["subquery"]

        if flows_candidate is None:
            return None

        combined_workflow = normalize_to_workflow(flows_candidate)
        if not combined_workflow:
            return None
        return combined_workflow
    except requests.RequestException:
        return None


