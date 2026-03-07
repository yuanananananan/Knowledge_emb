from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object("app.config.Config")
    # 用于 session 加密
    app.secret_key = 'embed'  # 一定要保密，建议用环境变量
    db.init_app(app)
    # 允许前端携带 cookie 的跨域设置
    CORS(app, supports_credentials=True)


    from app.routes import register_routes
    register_routes(app)

    return app