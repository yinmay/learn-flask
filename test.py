import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv("DEEPSEEK_API_KEY")


# ========== 1. 配置模型客户端 ==========
client = OpenAI(
    api_key=secret_key,
    base_url="https://api.deepseek.com" # 换成你要用的模型地址
)

# ========== 2. 定义工具（Agent的手脚） ==========
def calculator(expression: str):
    """计算器工具，简单数学计算"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{str(e)}"

def get_weather(city: str):
    """模拟天气查询工具"""
    weather_map = {
        "北京": "晴天，25℃",
        "上海": "阴天，22℃",
        "蒙特利尔": "小雨，10℃"
    }
    return weather_map.get(city, f"暂无{city}天气数据")

# 工具映射表：工具名字 -> 对应函数
tools = {
    "calculator": calculator,
    "get_weather": get_weather
}

# ========== 3. ReAct系统提示词（强制输出格式，重点！） ==========
system_prompt = """
你是一个ReAct智能体。你可以使用工具来回答用户问题。
可用工具：
1. calculator：用于数学计算，入参是数学表达式字符串，例如 "2+3*5"
2. get_weather：查询城市天气，入参是城市名称

你必须严格按照下面格式输出，不能额外增加其他文字！
Thought: 你的思考过程
Action: 工具名称
Action Input: 工具的参数

当你收集完所有信息，不需要再调用工具时，输出：
Thought: 我已经获取足够信息
Final Answer: 给用户的最终答案
"""

# ========== 4. ReAct主循环 ==========
def react_agent(user_query: str, max_round=5):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    for _ in range(max_round):
        # 调用大模型
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0
        )
        llm_output = resp.choices[0].message.content
        print("\n【模型输出】\n", llm_output)

        # 追加模型输出到对话历史
        messages.append({"role": "assistant", "content": llm_output})

        # 正则匹配是否得到Final Answer
        final_ans_match = re.search(r"Final Answer:(.*)", llm_output, re.DOTALL)
        if final_ans_match:
            return final_ans_match.group(1).strip()

        # 正则匹配 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", llm_output, re.DOTALL)
        if action_match:
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip()

            if action_name in tools:
                # 执行工具
                obs = tools[action_name](action_input)
                observation_text = f"Observation: {obs}"
                print(f"\n【工具返回】{observation_text}")
                # 把观察结果丢回消息列表，继续下一轮思考
                messages.append({"role": "user", "content": observation_text})
            else:
                messages.append({"role": "user", "content": f"Observation: 不存在工具 {action_name}"})
        else:
            messages.append({"role": "user", "content": "Observation: 输出格式错误，请严格遵守格式！"})

    return "达到最大循环轮次，任务终止"

# ========== 5. 启动Agent测试 ==========
if __name__ == "__main__":
    question = "蒙特利尔现在多少度？蒙特利尔当前温度加上10等于多少？"
    result = react_agent(question)
    print("\n✅ 最终答案：", result)
