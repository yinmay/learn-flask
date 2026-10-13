import asyncio
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import Msg, TextBlock, ToolResultState
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit, ToolChunk

load_dotenv()


def read_sales_data(file_path: str) -> ToolChunk:
    """读取销售 CSV，统计销售额。

    Args:
        file_path: 包含 product 和 sales 列的 CSV 文件路径。
    """
    df = pd.read_csv(file_path)
    if df.empty or not {"product", "sales"}.issubset(df.columns):
        raise ValueError("数据不能为空，且必须包含 product 和 sales 列")
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")
    if df["sales"].isna().any():
        raise ValueError("销售额不能缺失")
    top_product = df.loc[df["sales"].idxmax(), "product"]
    result = (
        f"总销售额：{df['sales'].sum()}, "
        f"单品平均销售额：{df['sales'].mean():.2f}, "
        f"销售额最高产品：{top_product}\n"
        f"各产品销售额：\n{df.to_string(index=False)}"
    )
    return ToolChunk(content=[TextBlock(text=result)], state=ToolResultState.SUCCESS)


def message(text: str) -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(text=text)])


def text_of(msg: Msg) -> str:
    return "\n".join(block.text for block in msg.content if isinstance(block, TextBlock))


async def main():
    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        ),
        model="deepseek-chat",
        stream=False,
    )
    toolkit = Toolkit(tools=[FunctionTool(
        read_sales_data,
        is_read_only=True,
        permission=PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="允许读取销售 CSV 以完成分析",
        ),
    )])
    orchestrator = Agent(
        name="主管",
        system_prompt="你是项目主管。拆解任务并汇总下属结果，不自行计算数据或编写营销文案。不得编造数据。",
        model=model,
        react_config=ReActConfig(max_iters=5),
    )
    data_analyst = Agent(
        name="数据分析员",
        system_prompt="你只负责销售分析。必须调用 read_sales_data 读取实际数据，再给出结论。不要编造数据或写营销文案。",
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=5),
    )
    creative = Agent(
        name="创意策划",
        system_prompt="基于提供的销售结论给出简短营销创意方案，不重新计算数据，不编造产品规格或市场事实。",
        model=model,
        react_config=ReActConfig(max_iters=5),
    )
    sales_path = Path(__file__).resolve().with_name("sales.csv")
    plan = await orchestrator.reply(message(
        f"读取 {sales_path}，分析销售情况，输出营销创意方案。请给数据分析员和创意策划分配任务。"
    ))
    print("\n===== 主管任务拆解 =====\n" + text_of(plan), flush=True)
    analysis = await data_analyst.reply(message(
        f"主管计划：\n{text_of(plan)}\n请调用工具分析文件：{sales_path}"
    ))
    print("\n===== 数据分析 =====\n" + text_of(analysis), flush=True)
    proposal = await creative.reply(message(
        f"主管计划：\n{text_of(plan)}\n数据分析结论：\n{text_of(analysis)}\n请给出营销方案。"
    ))
    print("\n===== 营销创意 =====\n" + text_of(proposal), flush=True)
    report = await orchestrator.reply(message(
        f"下属任务已完成，请汇总以下内容为完整报告。\n数据分析：\n{text_of(analysis)}"
        f"\n营销创意：\n{text_of(proposal)}"
    ))
    print("\n===== 最终报告 =====\n" + text_of(report))


if __name__ == "__main__":
    asyncio.run(main())
