# 文件：learn_flask/04_database.py

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"  # 简单起见用 SQLite
db = SQLAlchemy(app)


# 定义模型
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关联 - 一对多
    posts = db.relationship("Post", backref="author", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class Post(db.Model):
    __tablename__ = "posts"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
        }


# 创建表
with app.app_context():
    db.create_all()


# CRUD 路由
@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    user = User(username=data["username"], email=data["email"])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201


@app.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])


@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())


@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    if "username" in data:
        user.username = data["username"]
    if "email" in data:
        user.email = data["email"]
    db.session.commit()
    return jsonify(user.to_dict())


@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "删除成功"})


# 常用查询示例
@app.route("/users/search", methods=["GET"])
def search_users():
    # 过滤
    username = request.args.get("username")
    if username:
        users = User.query.filter(User.username.like(f"%{username}%")).all()
    else:
        users = User.query.all()

    # 排序
    # User.query.order_by(User.created_at.desc()).all()

    # 分页
    # User.query.paginate(page=1, per_page=10)

    return jsonify([u.to_dict() for u in users])


if __name__ == "__main__":
    app.run(debug=True, port=5001)
