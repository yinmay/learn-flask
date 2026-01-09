# 文件：learn_flask/01_hello.py

from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, Flask!"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
