"""从项目根目录启动：.venv/bin/python -m langGraph.FastAPI"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DATABASE_PATH = Path(__file__).resolve().with_name("agent_api.db")


def build_graph(model, saver):
    async def llm_node(state: MessagesState):
        response = await model.ainvoke([
            SystemMessage(content="用用户的语言简洁回答问题，不编造未知信息。"),
            *state["messages"],
        ])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm_node", llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    return builder.compile(checkpointer=saver)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY")
    model_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60,
        max_retries=2,
    )
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
        timeout=60,
        max_retries=2,
        root_async_client=model_client,
        async_client=model_client.chat.completions,
    )
    # 锁串行处理请求，避免同时修改同一 thread_id 的检查点。
    app.state.chat_lock = asyncio.Lock()
    try:
        async with AsyncSqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
            await saver.setup()
            app.state.graph = build_graph(model, saver)
            yield
    finally:
        await model_client.close()


app = FastAPI(title="LangGraph Chat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_msg: str = Field(min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.thread_id.strip() or not request.user_msg.strip():
        raise HTTPException(status_code=422, detail="thread_id 和 user_msg 不能为空白")
    config = {"configurable": {"thread_id": request.thread_id}}
    async with app.state.chat_lock:
        try:
            result = await app.state.graph.ainvoke(
                {"messages": [("user", request.user_msg)]}, config=config,
            )
        except Exception as error:
            logging.getLogger(__name__).exception("Chat failed for LangGraph request")
            raise HTTPException(status_code=502, detail="模型调用失败，请稍后重试") from error
    content = result["messages"][-1].content
    reply = content if isinstance(content, str) else str(content)
    return ChatResponse(thread_id=request.thread_id, reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
