# LangChain Python quickstart

A minimal weather agent using `create_agent`, following the official quickstart:
https://docs.langchain.com/oss/python/langchain/quickstart

Uses DeepSeek `deepseek-chat` through its OpenAI-compatible API. The weather tool
returns mock data; it does not fetch live weather. No LangSmith setup is needed.

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Create .env and set DEEPSEEK_API_KEY=your-deepseek-api-key.
.venv/bin/python agent.py
```

Inside the learn-flask repository, the script can also read the existing provider
key from the parent `.env`. Secrets and the virtual environment are gitignored.

For next steps, use the installed `langchain-fundamentals` skill.
