# 文件：learn_flask/03_crud.py

from flask import Flask, request, jsonify

app = Flask(__name__)

# 模拟数据库
users = {
    1: {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
    2: {"id": 2, "name": "李四", "email": "lisi@example.com"},
}
next_id = 3


# Create - 创建用户
@app.route("/users", methods=["POST"])
def create_user():
    global next_id
    data = request.get_json()

    if not data or "name" not in data:
        return jsonify({"error": "name 是必填字段"}), 400

    user = {
        "id": next_id,
        "name": data["name"],
        "email": data.get("email", ""),
    }
    users[next_id] = user
    next_id += 1

    return jsonify(user), 201


# Read - 获取所有用户
@app.route("/users", methods=["GET"])
def get_users():
    return jsonify(list(users.values()))


# Read - 获取单个用户
@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = users.get(user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify(user)


# Update - 更新用户
@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    if user_id not in users:
        return jsonify({"error": "用户不存在"}), 404

    data = request.get_json()
    user = users[user_id]

    if "name" in data:
        user["name"] = data["name"]
    if "email" in data:
        user["email"] = data["email"]

    return jsonify(user)


# Delete - 删除用户
@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"error": "用户不存在"}), 404

    del users[user_id]
    return jsonify({"message": "删除成功"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
