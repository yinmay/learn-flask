"""SQLite 检查点示例：恢复会话并隔离不同 thread_id 的历史。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END

DATABASE_PATH = Path(__file__).resolve().with_name("agent_session.db")


def build_graph(llm, checkpointer):
    def llm_node(state: MessagesState):
        response = llm.invoke([
            SystemMessage(content="简洁回答用户问题。只能依据当前会话的信息回答个人信息，不知道名字时直接说明。"),
            *state["messages"],
        ])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm_node", llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    return builder.compile(checkpointer=checkpointer)


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
    user_a = {"configurable": {"thread_id": "user_a_001"}}
    user_b = {"configurable": {"thread_id": "user_b_001"}}

    # from_conn_string 返回上下文管理器，需用 with 获取实际 saver。
    with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
        graph = build_graph(llm, saver)
        first = graph.invoke({"messages": [("user", "记住我的名字是 Alice")]}, config=user_a)
        print("用户 A：", first["messages"][-1].content, flush=True)
        second = graph.invoke({"messages": [("user", "我的名字是什么？")]}, config=user_b)
        print("用户 B：", second["messages"][-1].content, flush=True)

    # 关闭连接并重建图，演示从磁盘恢复，而非依赖内存中的旧图。
    with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
        graph = build_graph(llm, saver)
        restored = graph.invoke({"messages": [("user", "我叫什么名字？")]}, config=user_a)
        print("恢复用户 A：", restored["messages"][-1].content)


if __name__ == "__main__":
    main()
