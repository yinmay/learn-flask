"""启动：.venv/bin/python -m langGraph.sse；检查点仅在当前进程内保存。"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, StrictBool
from sse_starlette.sse import EventSourceResponse
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

PROJECT_ROOT = Path(__file__).resolve().parent / "my_project"
ALLOW_SUFFIX = {".py", ".js", ".ts", ".md", ".txt"}


class State(MessagesState):
    rounds: int


def validate_file_path(file_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    root = project_root.resolve()
    target = (root / file_path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError("禁止访问项目外目录")
    if target.suffix.lower() not in ALLOW_SUFFIX:
        raise PermissionError("文件类型不在白名单内")
    if not target.is_file():
        raise FileNotFoundError("文件不存在或不是普通文件")
    return target


def make_tools(project_root: Path):
    @tool
    def read_code(file_path: str) -> str:
        """读取工作目录内代码或文本文件，最多返回2000个字符。"""
        try:
            path = validate_file_path(file_path, project_root)
            with path.open(encoding="utf-8") as handle:
                return handle.read(2000)
        except (OSError, UnicodeError) as error:
            return f"读取失败：{error}"

    @tool
    def delete_file(file_path: str) -> str:
        """删除工作目录内文件；执行前必须得到人工审批。"""
        # interrupt 不能被普通异常处理吞掉；删除必须发生在审批之后。
        approved = interrupt({"task": "删除文件", "file_path": file_path})
        if approved is not True:
            return "用户拒绝审批，文件未删除。"
        try:
            path = validate_file_path(file_path, project_root)
            path.unlink()
            return f"已删除文件：{file_path}"
        except OSError as error:
            return f"删除失败：{error}"

    return [read_code, delete_file]


def build_graph(model, project_root: Path = PROJECT_ROOT, max_rounds: int = 5):
    if max_rounds < 1:
        raise ValueError("max_rounds 必须至少为1")
    tools = make_tools(project_root)
    llm = model.bind_tools(tools)

    async def agent_node(state: State):
        rounds = state.get("rounds", 0)
        if rounds >= max_rounds:
            return {"messages": [AIMessage(content="已达到最大循环轮次，任务终止。") ]}
        response = await llm.ainvoke([
            SystemMessage(content=(
                "你是代码助理。只能通过工具访问工作目录中的文件。"
                "删除文件需人工审批，审批被拒后不要重复尝试。工具错误应如实说明。"
            )),
            *state["messages"],
        ])
        return {"messages": [response], "rounds": rounds + 1}

    builder = StateGraph(State)
    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges(
        "agent_node",
        lambda state: "tool_node" if state["messages"][-1].tool_calls else END,
    )
    builder.add_edge("tool_node", "agent_node")
    return builder.compile(checkpointer=InMemorySaver())


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("请在根目录 .env 配置 DEEPSEEK_API_KEY")
    PROJECT_ROOT.mkdir(exist_ok=True)
    async with AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=60) as client:
        model = ChatOpenAI(
            api_key=key, base_url="https://api.deepseek.com", model="deepseek-chat",
            temperature=0, streaming=True, timeout=60, max_retries=2,
            root_async_client=client, async_client=client.chat.completions,
        )
        app.state.graph = build_graph(model)
        app.state.locks = {}
        yield


app = FastAPI(title="LangGraph SSE Agent", lifespan=lifespan)


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    interrupt_id: str = Field(min_length=1)
    approved: StrictBool


def sse(event: str, payload):
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


def pending_interrupts(snapshot):
    return [item for task in snapshot.tasks for item in task.interrupts]


async def event_generator(graph, graph_input, config, lock):
    try:
        async for mode, data in graph.astream(
            graph_input, config=config, stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "agent_node" and chunk.content:
                    yield sse("token", {"content": chunk.content})
            else:
                for pending in data.get("__interrupt__", ()):
                    yield sse("interrupt", {"id": pending.id, "value": pending.value})
                for node, update in data.items():
                    if node == "__interrupt__" or not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                yield sse("tool_call", call)
                        elif isinstance(message, ToolMessage):
                            yield sse("tool_result", {
                                "tool_call_id": message.tool_call_id,
                                "content": message.content,
                            })
        snapshot = await graph.aget_state(config)
        paused = pending_interrupts(snapshot)
        if paused:
            yield sse("done", {"status": "paused"})
        else:
            messages = snapshot.values.get("messages", [])
            reply = messages[-1].content if messages else ""
            yield sse("answer", {"content": reply})
            yield sse("done", {"status": "completed"})
    except Exception:
        logging.getLogger(__name__).exception("SSE graph execution failed")
        yield sse("error", {"message": "执行失败，请检查服务日志"})
    finally:
        lock.release()


async def prepare_stream(thread_id: str, query: str | None = None, resume: ResumeRequest | None = None):
    if not thread_id.strip() or query is not None and not query.strip():
        raise HTTPException(status_code=422, detail="输入不能为空白")
    lock = app.state.locks.setdefault(thread_id, asyncio.Lock())
    if lock.locked():
        raise HTTPException(status_code=409, detail="此会话已有请求正在执行")
    await lock.acquire()
    try:
        graph = app.state.graph
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
        snapshot = await graph.aget_state(config)
        pending = pending_interrupts(snapshot)
        if resume is not None:
            if resume.interrupt_id not in {item.id for item in pending}:
                raise HTTPException(status_code=409, detail="审批不存在或已处理")
            graph_input = Command(resume={resume.interrupt_id: resume.approved})
        else:
            if snapshot.next:
                raise HTTPException(status_code=409, detail="会话尚未完成，请先处理审批或重新使用 thread_id")
            graph_input = {"messages": [("user", query)], "rounds": 0}
        return EventSourceResponse(event_generator(graph, graph_input, config, lock))
    except BaseException:
        lock.release()
        raise


@app.get("/agent/stream")
async def agent_stream(
    query: str = Query(min_length=1, max_length=10000),
    thread_id: str = Query(min_length=1, max_length=128),
):
    return await prepare_stream(thread_id, query=query)


@app.post("/agent/resume")
async def agent_resume(request: ResumeRequest):
    return await prepare_stream(request.thread_id, resume=request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
