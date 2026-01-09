# 文件：learn_flask/routes/__init__.py

from .app_routes import app_bp
from .user_routes import user_bp

__all__ = ["app_bp", "user_bp"]
