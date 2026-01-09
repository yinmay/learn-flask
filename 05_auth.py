# 文件：learn_flask/05_auth.py

from flask import Flask, request, jsonify, g
from functools import wraps
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
SECRET_KEY = "your-secret-key"

# 模拟用户数据库
users_db = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}


# 生成 JWT Token
def generate_token(username, role):
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


# 验证 Token 的装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({"error": "缺少 Token"}), 401

        # 移除 "Bearer " 前缀
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            g.current_user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token 已过期"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "无效的 Token"}), 401

        return f(*args, **kwargs)

    return decorated


# 角色验证装饰器
def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if g.current_user.get("role") != role:
                return jsonify({"error": "权限不足"}), 403
            return f(*args, **kwargs)

        return decorated

    return decorator


# 登录接口
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = users_db.get(username)
    if not user or user["password"] != password:
        return jsonify({"error": "用户名或密码错误"}), 401

    token = generate_token(username, user["role"])
    return jsonify({"token": token, "username": username, "role": user["role"]})


# 需要登录的接口
@app.route("/profile")
@token_required
def profile():
    return jsonify(
        {"message": "个人信息", "user": g.current_user["username"], "role": g.current_user["role"]}
    )


# 需要管理员权限的接口
@app.route("/admin")
@token_required
@role_required("admin")
def admin_only():
    return jsonify({"message": "管理员专属页面"})


# 公开接口
@app.route("/public")
def public():
    return jsonify({"message": "公开接口，无需认证"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
