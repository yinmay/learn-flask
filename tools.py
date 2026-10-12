import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv("DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=secret_key,
    base_url="https://api.deepseek.com" # 兼容OpenAI接口都可以
)

# 1. 定义工具函数
def get_weather(city: str):
    weather_map = {"蒙特利尔": "小雨，10℃", "北京":"晴天，25℃"}
    return weather_map.get(city, f"暂无{city}天气")

def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

# 2. 工具描述 schema（给大模型看的）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"]
            }
        }
    }
]

# 工具映射
tool_map = {"get_weather": get_weather, "calculator": calculator}

def agent_run(user_query):
    messages = [{"role":"user", "content": user_query}]
    max_round = 5
    for _ in range(max_round):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            temperature=0
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # 判断是否调用工具
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = eval(tool_call.function.arguments)
                result = tool_map[func_name](**func_args)
                # 把工具结果加入消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # 没有工具调用，直接返回答案
            return msg.content
    return "超出最大轮次"

if __name__ == "__main__":
    print(agent_run("蒙特利尔温度加10等于多少"))
