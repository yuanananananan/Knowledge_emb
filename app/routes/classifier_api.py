from flask import Blueprint, request, jsonify
from app import db
from app.models.classifier import Classifier

bp = Blueprint("classifier", __name__, url_prefix="/api/classifier")

@bp.route("/", methods=["GET"])
def list_classifiers():
    data = Classifier.query.filter_by(isDelete=False).all()
    return jsonify([{"id": d.id, "name": d.name, "params_schema": d.params_schema, "description": d.description} for d in data])

@bp.route("/", methods=["POST"])
def create_classifier():
    data = request.json
    item = Classifier(name=data["name"], modality=data["modality"], params_schema=data["params_schema"], description=data["description"])
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Classifier created", "id": item.id})

@bp.route("/<int:id>", methods=["PUT"])
def update_classifier(id):
    item = Classifier.query.get_or_404(id)
    data = request.json
    item.name = data.get("name", item.name)
    item.modality = data.get("modality", item.modality)
    item.params_schema = data.get("params_schema", item.params_schema)
    item.description = data.get("description", item.description)
    db.session.commit()
    return jsonify({"message": "Classifier updated"})

@bp.route("/<int:id>", methods=["DELETE"])
def delete_classifier(id):
    item = Classifier.query.get_or_404(id)
    item.isDelete = True
    db.session.commit()
    return jsonify({"message": "Classifier deleted"})
