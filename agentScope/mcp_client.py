from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 连接一个MCP服务（独立进程，提供工具）
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"] # 单独的MCP服务文件
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 连接MCP服务，自动发现服务暴露的所有工具
            await session.initialize()
            tools = await session.list_tools()
            print("MCP服务可用工具：", tools)

            # 调用远程MCP工具
            result = await session.call_tool("get_weather", arguments={"city": "蒙特利尔"})
            print(result)

# MCP服务端会自动对外暴露工具Schema，不需要在Agent代码里手写tools定义！
