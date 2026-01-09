# 文件：learn_flask/routes/app_routes.py

from flask import Blueprint, jsonify

# 创建 Blueprint，url_prefix 为所有路由添加前缀
app_bp = Blueprint("app", __name__, url_prefix="/app")


@app_bp.route("/")
def index():
    return jsonify({"message": "应用首页"})


@app_bp.route("/about")
def about():
    return jsonify({"message": "关于我们", "version": "1.0.0"})


@app_bp.route("/health")
def health():
    return jsonify({"status": "healthy"})
