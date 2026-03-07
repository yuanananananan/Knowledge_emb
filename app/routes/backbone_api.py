from flask import Blueprint, request, jsonify
from app import db
from app.models.backbone import Backbone

bp = Blueprint("backbone", __name__, url_prefix="/api/backbone")

@bp.route("/", methods=["GET"])
def list_backbones():
    data = Backbone.query.filter_by(isDelete=False).all()
    return jsonify([{"id": d.id, "name": d.name, "params_schema": d.params_schema, "description": d.description} for d in data])

@bp.route("/", methods=["POST"])
def create_backbone():
    data = request.json
    item = Backbone(name=data["name"], modality=data["modality"], params_schema=data["params_schema"], description=data["description"])
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Backbone created", "id": item.id})

@bp.route("/<int:id>", methods=["PUT"])
def update_backbone(id):
    item = Backbone.query.get_or_404(id)
    data = request.json
    item.name = data.get("name", item.name)
    item.modality = data.get("modality", item.modality)
    item.params_schema = data.get("params_schema", item.params_schema)
    item.description = data.get("description", item.description)
    db.session.commit()
    return jsonify({"message": "Backbone updated"})

@bp.route("/<int:id>", methods=["DELETE"])
def delete_backbone(id):
    item = Backbone.query.get_or_404(id)
    item.isDelete = True
    db.session.commit()
    return jsonify({"message": "Backbone deleted"})
