# 文件：learn_flask/02_routes.py

from flask import Flask, request, jsonify

app = Flask(__name__)


# 基本路由
@app.route("/")
def index():
    return "首页"


# 带参数的路由
@app.route("/user/<username>")
def show_user(username):
    return f"用户: {username}"


# 带类型转换的参数
@app.route("/post/<int:post_id>")
def show_post(post_id):
    return f"文章 ID: {post_id}"


# 多种 HTTP 方法
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return "处理登录请求"
    return "显示登录页面"


# 返回 JSON
@app.route("/api/data")
def get_data():
    return jsonify({"name": "Flask", "version": "3.x", "status": "ok"})


# 获取查询参数
@app.route("/search")
def search():
    keyword = request.args.get("q", "")
    page = request.args.get("page", 1, type=int)
    return jsonify({"keyword": keyword, "page": page})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
