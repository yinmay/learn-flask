import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

# 1. 定义全局状态
class State(MessagesState):
    """使用内置消息 reducer，自动合并消息并按 ID 更新。"""

# 2. 模型
def build_graph(llm):
    # 3. 定义节点函数：只返回新增消息，由 reducer 合并到状态。
    def llm_node(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 4. 构建图：开始 → LLM → 结束。
    builder = StateGraph(State)
    builder.add_node("llm_node", llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    return builder.compile()


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    graph = build_graph(llm)
    result = graph.invoke({"messages": [("user", "简单介绍LangGraph")]})
    print(result["messages"][-1].content)

# 5. 运行
if __name__ == "__main__":
    main()
