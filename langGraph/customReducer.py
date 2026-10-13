from typing import TypedDict, Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


def merge_completed_steps(current: list[str], incoming: list[str]) -> list[str]:
    """合并已完成步骤，保留首次出现顺序，并且不修改输入列表。"""
    return list(dict.fromkeys([*current, *incoming]))

# 自定义State，增加额外业务字段
class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    completed_steps: Annotated[list[str], merge_completed_steps]
    task_stage: Literal["pending", "running", "completed"]   # 任务阶段
    summary: str      # 中间摘要

# 模拟节点
def start_node(state: CustomState):
    return {
        "messages": [AIMessage(content="任务启动")],
        "completed_steps": ["start"],
        "task_stage": "running",
        "summary": ""
    }

def summary_node(state: CustomState):
    return {
        "messages": [AIMessage(content="模拟销售数据读取完成")],
        "completed_steps": ["start", "summary"],
        "task_stage": "completed",
        "summary": "模拟销售数据读取完成（此示例未读取实际文件）",
    }

def build_graph():
    builder = StateGraph(CustomState)
    builder.add_node("start", start_node)
    builder.add_node("summary", summary_node)
    builder.add_edge(START, "start")
    builder.add_edge("start", "summary")
    builder.add_edge("summary", END)
    return builder.compile()

if __name__ == "__main__":
    res = build_graph().invoke({
        "messages": [("user", "分析销售数据")],
        "completed_steps": [],
        "task_stage": "pending",
        "summary": ""
    })
    print("任务阶段：", res["task_stage"])
    print("已完成步骤：", res["completed_steps"])
    print("摘要：", res["summary"])
    for message in res["messages"]:
        print(f"{message.type}: {message.content}")
