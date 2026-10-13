"""LangGraph 工具审批示例；邮件工具仅模拟发送。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


class State(MessagesState):
    rounds: int


@tool
def send_email(receiver: str, content: str) -> str:
    """模拟向指定收件人发送邮件；需人工批准，不发送真实邮件。"""
    approved = interrupt({
        "question": "是否批准这次模拟邮件发送？",
        "receiver": receiver,
        "content": content,
    })
    if type(approved) is not bool:
        raise ValueError("审批结果必须是 True 或 False")
    if not approved:
        return "用户拒绝审批，邮件未发送。"
    return f"模拟邮件发送完成：收件人 {receiver}，内容：{content}。没有发送真实邮件。"


def build_graph(model, max_rounds: int = 5):
    if max_rounds < 1:
        raise ValueError("max_rounds 必须至少为 1")
    llm = model.bind_tools([send_email])

    def llm_node(state: State):
        rounds = state.get("rounds", 0)
        if rounds >= max_rounds:
            return {"messages": [AIMessage(content="已达到最大循环轮次，任务终止。") ]}
        response = llm.invoke(state["messages"])
        return {"messages": [response], "rounds": rounds + 1}

    def should_continue(state: State):
        return "tool_node" if state["messages"][-1].tool_calls else END

    builder = StateGraph(State)
    builder.add_node("llm_node", llm_node)
    builder.add_node("tool_node", ToolNode([send_email]))
    builder.add_edge(START, "llm_node")
    builder.add_conditional_edges("llm_node", should_continue)
    builder.add_edge("tool_node", "llm_node")
    # 必须使用检查点和同一个 thread_id 才能恢复中断。
    return builder.compile(checkpointer=InMemorySaver())


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    graph = build_graph(model)
    config = {"configurable": {"thread_id": "email-review-demo"}, "recursion_limit": 20}
    result = graph.invoke({
        "messages": [("user", "调用模拟邮件工具给 admin@example.com 发送报告摘要：销售分析已完成。")],
        "rounds": 0,
    }, config=config)
    while result.get("__interrupt__"):
        for pending in result["__interrupt__"]:
            print("等待审批：", pending.value, flush=True)
        try:
            answer = input("批准此次模拟操作？输入 yes 批准，其他输入拒绝：")
        except EOFError:
            print("没有收到审批输入，退出示例，工具未执行。内存检查点不会跨进程保留。")
            return
        result = graph.invoke(Command(resume=answer.strip().lower() == "yes"), config=config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
