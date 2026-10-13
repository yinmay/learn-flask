import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END


class State(MessagesState):
    next_agent: Literal["research", "write", "END"]
    research_done: bool
    write_done: bool


def build_graph(llm):
    def research_worker(state: State):
        response = llm.invoke([
            SystemMessage(content="你是调研专员，仅完成销售分析，不写营销方案。基于用户提供的数据输出简短结论。没有数据时说明缺少信息，不编造市场事实。"),
            *state["messages"],
        ])
        return {
            "messages": [AIMessage(content=response.content, name="research")],
            "research_done": True,
        }

    def write_worker(state: State):
        response = llm.invoke([
            SystemMessage(content="你是文案专员，基于调研结论写简短营销方案。区分建议和已知事实，不编造产品规格。"),
            *state["messages"],
        ])
        return {
            "messages": [AIMessage(content=response.content, name="write")],
            "write_done": True,
        }

    def supervisor(state: State):
        # 本示例要求先调研再写文案；完成标记确保每个 worker 只执行一次。
        expected = (
            "research" if not state.get("research_done", False)
            else "write" if not state.get("write_done", False)
            else "END"
        )
        if expected == "END":
            return {"next_agent": "END"}
        decision = llm.invoke([
            SystemMessage(content=(
                "你是主管，只返回 research、write 或 END 中的一个名字。"
                f"调研完成：{state.get('research_done', False)}；"
                f"文案完成：{state.get('write_done', False)}。"
                "必须先调研，再写文案，两项完成后结束。"
            )),
            *state["messages"],
        ])
        choice = decision.content.strip() if isinstance(decision.content, str) else ""
        # 不合法、跳步或重复的决策回退到当前允许的步骤。
        return {"next_agent": choice if choice == expected else expected}

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("research_worker", research_worker)
    builder.add_node("write_worker", write_worker)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {"research": "research_worker", "write": "write_worker", "END": END},
    )
    builder.add_edge("research_worker", "supervisor")
    builder.add_edge("write_worker", "supervisor")
    return builder.compile()


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    result = build_graph(llm).invoke({
        "messages": [("user", "分析销售额：水杯2400、背包1800、雨伞1200，并写简短营销方案。")],
        "research_done": False,
        "write_done": False,
    }, config={"recursion_limit": 10})
    for message in result["messages"]:
        if message.name in ("research", "write"):
            print(f"\n===== {message.name} =====\n{message.content}")


if __name__ == "__main__":
    main()
