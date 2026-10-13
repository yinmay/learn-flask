"""自定义除法工具：将计算错误作为观察结果交给模型。"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode


class State(MessagesState):
    rounds: int


@tool
def division(a: float, b: float) -> str:
    """计算 a 除以 b；除数为零或数值无效时返回错误说明。"""
    if not math.isfinite(a) or not math.isfinite(b):
        return "工具调用失败：参数必须是有限数值。"
    if b == 0:
        return "工具调用失败：除数不能为零，10 / 0 等此类运算没有定义。"
    result = a / b
    if not math.isfinite(result):
        return "工具调用失败：计算结果超出有限数值范围。"
    return str(result)


def build_graph(model, max_rounds: int = 5):
    if max_rounds < 1:
        raise ValueError("max_rounds 必须至少为 1")
    tools = [division]
    llm = model.bind_tools(tools)

    def llm_node(state: State):
        rounds = state.get("rounds", 0)
        if rounds >= max_rounds:
            return {"messages": [AIMessage(content="已达到最大循环轮次，任务终止。") ]}
        response = llm.invoke([
            SystemMessage(content=(
                "使用 division 工具完成除法计算。工具返回错误时向用户解释原因，"
                "不要反复调用同一组无效参数，不要编造计算结果。"
            )),
            *state["messages"],
        ])
        return {"messages": [response], "rounds": rounds + 1}

    def should_continue(state: State):
        return "tool_node" if state["messages"][-1].tool_calls else END

    builder = StateGraph(State)
    builder.add_node("llm_node", llm_node)
    # 参数校验错误等也转成 ToolMessage，交由模型处理。
    builder.add_node("tool_node", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "llm_node")
    builder.add_conditional_edges("llm_node", should_continue)
    builder.add_edge("tool_node", "llm_node")
    return builder.compile()


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY")
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    max_rounds = 5
    result = build_graph(model, max_rounds).invoke(
        {"messages": [("user", "请调用除法工具计算 10 / 0")], "rounds": 0},
        config={"recursion_limit": 2 * max_rounds + 2},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
