#!/usr/bin/env python3
"""Score synthetic Chinese prompts and tool-call envelopes without executing tools."""

import json
import unicodedata

from benchmark import MODEL, api

CHINESE_CASES = (
    ("volume", "250 毫升、350 毫升和 400 毫升加起來是多少公升？只輸出兩位小數，不要單位。", "1.00"),
    ("time", "會議在下午 2 時 35 分開始，50 分鐘後結束。只用 24 小時制輸出結束時間，格式 HH:MM。", "15:25"),
    ("extraction", "訂單 A-110 已付清。香港的訂單編號是 HK-027，澳門的編號是 MO-042。只輸出香港的訂單編號。", "HK-027"),
    ("traditional", "請將「这个数据库已经恢复正常」轉成繁體中文，使用香港常見用字，只輸出句子，不要標點。", "這個資料庫已經恢復正常"),
    ("selection", "甲倉有 7 箱，乙倉有 12 箱，丙倉有 9 箱。哪一個倉的箱數最多？只輸出倉名。", "乙倉"),
)
INVENTORY = {
    "type": "function",
    "function": {
        "name": "lookup_inventory",
        "description": "Look up fictional inventory by SKU and warehouse.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "warehouse": {"type": "string"},
                           "include_reserved": {"type": "boolean"}},
            "required": ["sku", "warehouse", "include_reserved"],
            "additionalProperties": False,
        },
    },
}
COMPARISON = {
    "type": "function",
    "function": {
        "name": "compare_warehouses",
        "description": "Compare fictional warehouses in the requested order.",
        "parameters": {
            "type": "object",
            "properties": {"warehouses": {"type": "array", "items": {"type": "string"}},
                           "limit": {"type": "integer"},
                           "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}},
                                             {"type": "null"}]}},
            "required": ["warehouses", "limit", "tags"],
            "additionalProperties": False,
        },
    },
}
TOOL_CASES = (
    ("inventory_en", "Use the tool to look up SKU SPARK-42 in the Kowloon warehouse. Exclude reserved stock.",
     "lookup_inventory", {"sku": "SPARK-42", "warehouse": "Kowloon", "include_reserved": False}),
    ("inventory_zh", "請用工具查詢沙田倉庫（倉庫名稱請用 Sha Tin）的貨品 TAI-12，要包括已預留庫存。",
     "lookup_inventory", {"sku": "TAI-12", "warehouse": "Sha Tin", "include_reserved": True}),
    ("typed_array", "Use the comparison tool with warehouses Kowloon then Sha Tin, limit 3, tags cold and priority.",
     "compare_warehouses", {"warehouses": ["Kowloon", "Sha Tin"], "limit": 3, "tags": ["cold", "priority"]}),
)

def ask(messages, tools=None):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}, "trace": False}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    choice = api("/v1/chat/completions", body, timeout=120)["choices"][0]
    return choice["message"], choice.get("finish_reason")

def normalized(value):
    return unicodedata.normalize("NFKC", value or "").strip().strip("。.!！")

def chinese_case(case):
    name, prompt, expected = case
    message, _ = ask([{"role": "user", "content": prompt}])
    actual = normalized(message.get("content"))
    return {"name": name, "passed": actual == expected, "expected": expected, "actual": actual}

def tool_case(case):
    name, prompt, expected_name, expected_args = case
    messages = [{"role": "user", "content": prompt}]
    message, reason = ask(messages, [INVENTORY, COMPARISON])
    calls = message.get("tool_calls") or []
    call = calls[0] if len(calls) == 1 else {}
    function = call.get("function") or {}
    args = function.get("arguments")
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        parsed = args
    passed = (function.get("name") == expected_name and bool(call.get("id"))
              and reason == "tool_calls" and isinstance(args, str)
              and json.dumps(parsed, sort_keys=True) == json.dumps(expected_args, sort_keys=True))
    return {"name":name, "passed":passed, "expected_tool":expected_name, "expected_args":expected_args,
            "actual_tool":function.get("name"), "actual_args":parsed, "finish_reason":reason,
            "call_count":len(calls), "string_args":isinstance(args, str)}

def tool_result():
    messages = [
        {"role":"user", "content":"Use the inventory tool for SPARK-42 in Kowloon without reserved stock. After receiving the result, reply with the numeric quantity only."},
        {"role":"assistant", "content":None, "tool_calls":[{"id":"call_synthetic_42", "type":"function",
             "function":{"name":"lookup_inventory","arguments":json.dumps(TOOL_CASES[0][3])}}]},
        {"role":"tool","tool_call_id":"call_synthetic_42","content":'{"quantity":23}'},
    ]
    message, _ = ask(messages, [INVENTORY])
    actual = normalized(message.get("content"))
    return {"name":"follow_up", "passed": actual == "23" and not message.get("tool_calls"),
            "actual":actual, "expected":"23", "extra_calls":len(message.get("tool_calls") or [])}

def checked(case_fn, case):
    try:
        return case_fn(case)
    except Exception as exc:
        return {"name":case[0], "passed":False, "error":str(exc)}

if __name__ == "__main__":
    chinese = [checked(chinese_case, case) for case in CHINESE_CASES]
    tools = [checked(tool_case, case) for case in TOOL_CASES]
    tools.append(checked(lambda _:tool_result(), ("follow_up",)))
    print(json.dumps({"model":MODEL,"chinese":chinese,"tools":tools}, ensure_ascii=False, indent=2))
