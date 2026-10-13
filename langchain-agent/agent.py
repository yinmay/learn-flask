"""Minimal weather agent from the LangChain Python quickstart."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def get_weather(city: str) -> str:
    """Get example weather for a city. This is mock data, not a live forecast."""
    return f"It's always sunny in {city}! (Mock quickstart data.)"


def build_agent(model):
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "You are a helpful assistant. Use get_weather for weather questions. "
            "Explain that the tool returns mock weather, not a live forecast."
        ),
    )


def main():
    directory = Path(__file__).resolve().parent
    load_dotenv(directory / ".env")
    # Reuse the surrounding project's provider key when running in this repo.
    if not os.getenv("DEEPSEEK_API_KEY"):
        load_dotenv(directory.parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY in .env.")
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    result = build_agent(model).invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
        config={"recursion_limit": 10},
    )
    tool_messages = [message for message in result["messages"] if message.type == "tool"]
    for message in tool_messages:
        print(f"Tool result: {message.content}")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
