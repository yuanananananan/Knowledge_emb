from flask import Blueprint, request, jsonify, Response
from app import db
from utils.utils import load_config
from app.utils.utils import yaml_to_schema, generate_schema

bp = Blueprint("config", __name__, url_prefix="/api/config")

@bp.route("/schema", methods=["GET"])
def get_config_schema():
    config_name = request.args.get("name", "sar_config.yaml")
    cfg = load_config(f'D:/Project/knowledge_emb/configs/{config_name}')
    # schema = yaml_to_schema(cfg)
    schema = generate_schema(f'D:/Project/knowledge_emb/configs/{config_name}')
    return jsonify({"extra":schema})