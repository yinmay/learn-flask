import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv("DEEPSEEK_API_KEY")


client = AsyncOpenAI(
    api_key=secret_key,
    base_url="https://api.deepseek.com" # 兼容OpenAI接口都可以
)

# 工具定义
def get_weather(city: str):
    weather_map = {"蒙特利尔": "小雨，10℃", "北京":"晴天，25℃"}
    return weather_map.get(city, f"暂无{city}天气")

def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

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
tool_map = {"get_weather": get_weather, "calculator": calculator}

# ========== 会话存储：同一个会话持续复用messages ==========
class SessionAgent:
    def __init__(self):
        self.messages = []
        self.max_round = 5

    async def chat(self, user_input):
        self.messages.append({"role":"user", "content": user_input})
        for _ in range(self.max_round):
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=tools,
                temperature=0
            )
            msg = resp.choices[0].message
            self.messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = eval(tool_call.function.arguments)
                    result = tool_map[func_name](**func_args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                return msg.content
        return "超出最大轮次限制"

if __name__ == "__main__":
    agent = SessionAgent()
    print("Agent启动，输入exit退出")
    while True:
        user_text = input("你：")
        if user_text == "exit":
            break
        res = agent.chat(user_text)
        print(f"Agent：{res}")


