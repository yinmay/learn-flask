# 文件：learn_flask/app.py
# 3.1 Blueprint 基础 - 主入口

from flask import Flask

# 导入 Blueprint
from routes import app_bp, user_bp

app = Flask(__name__)

# 注册 Blueprint
app.register_blueprint(app_bp)   # /app/*
app.register_blueprint(user_bp)  # /users/*


@app.route("/")
def index():
    return {
        "message": "Flask Blueprint 示例",
        "endpoints": {
            "app": "/app/, /app/about, /app/health",
            "users": "/users/, /users/<id>, /users/search?role=xxx",
        },
    }


if __name__ == "__main__":
    app.run(debug=True, port=5001)
