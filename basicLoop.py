import ast
import operator
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode


class State(MessagesState):
    rounds: int


OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def evaluate_expression(expression: str):
    if len(expression) > 200:
        raise ValueError("表达式过长")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
            return OPERATIONS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return evaluate(node.operand)
        raise ValueError("仅支持数字、括号和 +、-、*、/、//、% 运算")

    return evaluate(tree.body)


@tool
def calculator(expression: str) -> str:
    """计算基本算术表达式，支持数字、括号和 +、-、*、/、//、%。"""
    try:
        return str(evaluate_expression(expression))
    except (ValueError, SyntaxError, ArithmeticError, RecursionError) as error:
        return f"计算失败：{error}"


def build_graph(model, max_rounds: int = 5):
    if max_rounds < 1:
        raise ValueError("max_rounds 必须至少为 1")
    tools = [calculator]
    llm = model.bind_tools(tools)

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
    builder.add_node("tool_node", ToolNode(tools))
    builder.add_edge(START, "llm_node")
    builder.add_conditional_edges("llm_node", should_continue)
    builder.add_edge("tool_node", "llm_node")
    return builder.compile()


def main():
    load_dotenv(Path(__file__).resolve().with_name(".env"))
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
    max_rounds = 5
    graph = build_graph(model, max_rounds)
    result = graph.invoke(
        {"messages": [("user", "请使用计算器计算 2400 + 1800 + 1200")], "rounds": 0},
        config={"recursion_limit": 2 * max_rounds + 2},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
