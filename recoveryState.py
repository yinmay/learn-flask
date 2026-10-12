import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.state import AgentState

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "agent_state.json"
SYSTEM_PROMPT = (
    "你是销售分析师，按用户指定步骤分析。第一步列出产品，第二步总结结论。"
    "没有提供销售数据时，应明确说明数据不足，不得编造销售额、销量或排名。"
)


def user_message(text: str) -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(text=text)])


def text_of(message: Msg) -> str:
    return "\n".join(block.text for block in message.content if isinstance(block, TextBlock))


def save_state(state: AgentState, path: Path) -> None:
    """持久化会话和运行状态，原子替换避免写入中断。"""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def load_state(path: Path) -> AgentState:
    return AgentState.model_validate_json(path.read_text(encoding="utf-8"))


def create_agent(model: OpenAIChatModel, state: AgentState | None = None) -> Agent:
    # 模型与系统提示词由应用重新配置，不包含在 AgentState 中。
    return Agent(
        name="分析师",
        system_prompt=SYSTEM_PROMPT,
        model=model,
        state=state,
        react_config=ReActConfig(max_iters=5),
    )


async def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")
    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        ),
        model="deepseek-chat",
        stream=False,
    )
    agent = create_agent(model)
    first_result = await agent.reply(user_message(
        "分析水杯、背包、雨伞三款产品销售情况。现在仅完成第一步：列出产品，等待下一条消息再总结。"
    ))
    print("第一轮输出：\n" + text_of(first_result), flush=True)
    save_state(agent.state, STATE_PATH)
    print(f"状态已保存：{STATE_PATH.name}", flush=True)

    # 新实例只通过 JSON 文件恢复历史，不依赖原 Agent 实例。
    new_agent = create_agent(model, load_state(STATE_PATH))
    print("\n状态加载完成，继续第二步", flush=True)
    second_result = await new_agent.reply(user_message(
        "继续完成第二步，针对之前列出的三款产品总结；若没有销售数据，请说明缺少哪些数据。"
    ))
    save_state(new_agent.state, STATE_PATH)
    print("继续执行结果：\n" + text_of(second_result))


if __name__ == "__main__":
    asyncio.run(main())
