import asyncio
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg, TextBlock

import os
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv("DEEPSEEK_API_KEY")


# ========== 配置 OpenAI兼容模型（DeepSeek / 通义千问都可以） ==========
model = OpenAIChatModel(
    credential=OpenAICredential(
        api_key=secret_key,
        base_url="https://api.deepseek.com",
    ),
    model="deepseek-chat",
    stream=False,
)

# Agent1：市场调研智能体
research_agent = Agent(
    name="市场调研专员",
    system_prompt="你是市场调研专员，负责分析产品卖点，输出简短市场调研结论。",
    model=model,
    react_config=ReActConfig(max_iters=5),
)

# Agent2：创意策划智能体
creative_agent = Agent(
    name="创意策划",
    system_prompt="你是创意策划，拿到市场调研结果，写一条简短产品宣传文案。",
    model=model,
    react_config=ReActConfig(max_iters=5),
)

async def main():
    user_msg = Msg(name="user", content=[TextBlock(text="给一款便携水杯做市场分析")], role="user")
    # 串行流水线：先调研Agent执行，输出交给创意Agent（串行流水线模式）
    research_msg = await research_agent.reply(user_msg)
    final_msg = await creative_agent.reply(research_msg)
    print("\n===== 最终输出 =====")
    for block in final_msg.content:
        if isinstance(block, TextBlock):
            print(block.text)

if __name__ == "__main__":
    asyncio.run(main())
