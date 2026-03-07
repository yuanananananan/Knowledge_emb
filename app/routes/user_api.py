from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from app.models.user import User
from functools import wraps

bp = Blueprint("user", __name__, url_prefix="/api/user")

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user_id = session.get("user_id")
            if not user_id:
                return jsonify({"message": "Unauthorized: not logged in"}), 401

            # 可选角色校验
            if role:
                user = User.query.get(user_id)
                user_role = user.userRole if user else None
                if isinstance(role, list):
                    if user_role not in role:
                        return jsonify({"message": "Forbidden: insufficient permissions"}), 403
                elif user_role != role:
                    return jsonify({"message": "Forbidden: insufficient permissions"}), 403

            return f(*args, **kwargs)
        return wrapper
    return decorator
@bp.route("/", methods=["GET"])
def list_users():
    users = User.query.filter_by(isDelete=False).all()
    return jsonify([{
        "id": u.id,
        "userName": u.userName,
        "userAccount": u.userAccount,
        "userRole": u.userRole
    } for u in users])

@bp.route("/", methods=["POST"])
def create_user():
    data = request.json
    user = User(
        userName=data["userName"],
        userAccount=data["userAccount"],
        userPassword=data["userPassword"],
        userRole=data.get("userRole", "user")
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created", "id": user.id})

@bp.route("/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.json
    user.userName = data.get("userName", user.userName)
    user.userPassword = data.get("userPassword", user.userPassword)
    user.userRole = data.get("userRole", user.userRole)
    db.session.commit()
    return jsonify({"message": "User updated"})

@bp.route("/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    user.isDelete = True
    db.session.commit()
    return jsonify({"message": "User deleted"})



@bp.route("/register", methods=["POST"])
def register():
    data = request.json
    userAccount = data.get("userAccount")
    userName = data.get("userName")
    userPassword = data.get("userPassword")
    userRole = data.get("userRole", "user")

    # 检查账户是否已存在
    existing_user = User.query.filter_by(userAccount=userAccount, isDelete=False).first()
    if existing_user:
        return jsonify({"message": "Account already exists"}), 400

    # 使用哈希加密存储密码
    hashed_password = generate_password_hash(userPassword)

    new_user = User(
        userName=userName,
        userAccount=userAccount,
        userPassword=hashed_password,
        userRole=userRole
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "id": new_user.id}), 201


@bp.route("/login", methods=["POST"])
def login():
    data = request.json
    userAccount = data.get("userAccount")
    userPassword = data.get("userPassword")

    user = User.query.filter_by(userAccount=userAccount, isDelete=False).first()
    if user and check_password_hash(user.userPassword, userPassword):
        # 登录成功，设置 session
        session["user_id"] = user.id
        session["user_role"] = user.userRole

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user.id,
                "userName": user.userName,
                "userAccount": user.userAccount,
                "userRole": user.userRole
            }
        })
    else:
        return jsonify({"message": "Invalid credentials"}), 401

@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})