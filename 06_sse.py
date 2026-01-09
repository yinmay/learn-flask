# 文件：learn_flask/06_sse.py

from flask import Flask, Response, request, render_template_string
import json
import time

app = Flask(__name__)


# SSE 生成器
def event_stream():
    """生成 SSE 事件流"""
    count = 0
    while count < 10:
        count += 1
        # SSE 格式: data: <内容>\n\n
        data = json.dumps({"count": count, "time": time.strftime("%H:%M:%S")})
        yield f"data: {data}\n\n"
        time.sleep(1)

    # 发送结束事件
    yield f"event: done\ndata: {json.dumps({'message': '流结束'})}\n\n"


# 模拟 AI 流式响应
def ai_stream(prompt):
    """模拟 AI 流式输出"""
    response = f"您问的是: {prompt}。这是一个模拟的 AI 回复，"
    words = response + "每个字符会逐个输出，模拟真实的 AI 流式响应效果。"

    for char in words:
        data = json.dumps({"content": char, "done": False})
        yield f"data: {data}\n\n"
        time.sleep(0.05)

    # 发送完成信号
    yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"


# SSE 端点
@app.route("/stream")
def stream():
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# AI 流式响应端点
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "Hello")

    return Response(
        ai_stream(prompt),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# 测试页面
@app.route("/")
def index():
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Flask SSE 示例</title>
    <style>
        body { font-family: monospace; padding: 20px; }
        #output { background: #f0f0f0; padding: 10px; min-height: 200px; }
        button { margin: 10px 0; padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>Flask SSE 示例</h1>

    <h2>1. 基础 SSE 流</h2>
    <button onclick="startStream()">开始接收</button>
    <div id="output"></div>

    <h2>2. AI 流式响应</h2>
    <input id="prompt" value="你好" style="padding: 10px; width: 200px;">
    <button onclick="startChat()">发送</button>
    <div id="chat-output"></div>

    <script>
        function startStream() {
            const output = document.getElementById('output');
            output.innerHTML = '';

            const eventSource = new EventSource('/stream');

            eventSource.onmessage = (e) => {
                const data = JSON.parse(e.data);
                output.innerHTML += `计数: ${data.count}, 时间: ${data.time}<br>`;
            };

            eventSource.addEventListener('done', (e) => {
                output.innerHTML += '<br><b>流结束</b>';
                eventSource.close();
            });

            eventSource.onerror = () => {
                eventSource.close();
            };
        }

        async function startChat() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('chat-output');
            output.innerHTML = '';

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        if (!data.done) {
                            output.innerHTML += data.content;
                        }
                    }
                }
            }
        }
    </script>
</body>
</html>
    """
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
