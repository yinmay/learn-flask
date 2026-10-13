import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")
    app.state.model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        ),
        model="deepseek-chat",
        stream=False,
    )
    app.state.sessions = {}
    try:
        yield
    finally:
        app.state.sessions.clear()
        await app.state.model.client.close()


app = FastAPI(title="AgentScope Chat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt 不能为空白")

    if request.session_id is None:
        session_id = uuid4().hex
        agent = Agent(
            name="销售助理",
            system_prompt="简短回答用户销售相关问题，不编造未提供的销售数据。",
            model=app.state.model,
            react_config=ReActConfig(max_iters=5),
        )
        app.state.sessions[session_id] = (agent, asyncio.Lock())
    else:
        session_id = request.session_id
        if session_id not in app.state.sessions:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")

    agent, lock = app.state.sessions[session_id]
    # 同一会话顺序处理，避免并发请求交叉修改历史。
    async with lock:
        previous_state = agent.state.model_copy(deep=True)
        try:
            result = await agent.reply(Msg(
                name="user",
                role="user",
                content=[TextBlock(text=request.prompt)],
            ))
        except Exception as error:
            agent.state = previous_state
            raise HTTPException(status_code=502, detail="模型调用失败，请稍后重试") from error

    reply = "\n".join(block.text for block in result.content if isinstance(block, TextBlock))
    return ChatResponse(session_id=session_id, reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
