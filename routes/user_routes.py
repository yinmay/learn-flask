# 文件：learn_flask/routes/user_routes.py

from flask import Blueprint, request, jsonify

user_bp = Blueprint("user", __name__, url_prefix="/users")

# 模拟数据
users_db = {
    1: {"id": 1, "name": "张三", "role": "admin"},
    2: {"id": 2, "name": "李四", "role": "user"},
}


@user_bp.route("/")
def list_users():
    return jsonify(list(users_db.values()))


@user_bp.route("/<int:user_id>")
def get_user(user_id):
    user = users_db.get(user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify(user)


@user_bp.route("/search")
def search_users():
    role = request.args.get("role")
    if role:
        result = [u for u in users_db.values() if u["role"] == role]
        return jsonify(result)
    return jsonify(list(users_db.values()))
