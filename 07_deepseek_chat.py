# 文件：learn_flask/07_deepseek_chat.py
# Flask + DeepSeek API 流式聊天示例

from flask import Flask, Response, request, render_template_string
from openai import OpenAI
import json
import os

app = Flask(__name__)

# DeepSeek API 配置（兼容 OpenAI SDK）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-api-key"),
    base_url="https://api.deepseek.com",
)


def chat_stream(messages):
    """调用 DeepSeek API 并流式返回"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                data = json.dumps({"content": content, "done": False}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        # 发送完成信号
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


@app.route("/chat", methods=["POST"])
def chat():
    """聊天 API - 流式响应"""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return {"error": "消息不能为空"}, 400

    messages = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": user_message},
    ]

    return Response(
        chat_stream(messages),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/chat/sync", methods=["POST"])
def chat_sync():
    """聊天 API - 非流式响应"""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return {"error": "消息不能为空"}, 400

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
                {"role": "user", "content": user_message},
            ],
            stream=False,
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/")
def index():
    """聊天页面"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DeepSeek Chat</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        #chat-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .message {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .assistant {
            background: #e9ecef;
            color: #333;
        }
        #input-container {
            display: flex;
            gap: 10px;
        }
        #message-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .typing { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>🤖 DeepSeek Chat</h1>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="message-input" placeholder="输入消息..." onkeypress="if(event.key==='Enter')sendMessage()">
        <button onclick="sendMessage()" id="send-btn">发送</button>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');

        function addMessage(content, role) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = content;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // 添加用户消息
            addMessage(message, 'user');
            messageInput.value = '';
            sendBtn.disabled = true;

            // 创建 AI 响应容器
            const assistantDiv = addMessage('', 'assistant');
            assistantDiv.classList.add('typing');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullContent = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const text = decoder.decode(value);
                    const lines = text.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.error) {
                                    assistantDiv.textContent = '错误: ' + data.error;
                                } else if (!data.done) {
                                    fullContent += data.content;
                                    assistantDiv.textContent = fullContent;
                                    assistantDiv.classList.remove('typing');
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (error) {
                assistantDiv.textContent = '请求失败: ' + error.message;
            }

            assistantDiv.classList.remove('typing');
            sendBtn.disabled = false;
            messageInput.focus();
        }
    </script>
</body>
</html>
    ''')


if __name__ == "__main__":
    print("请设置环境变量: export DEEPSEEK_API_KEY=your-api-key")
    print("然后访问: http://127.0.0.1:5001")
    app.run(debug=True, port=5001, threaded=True)
