from flask import Blueprint, request, jsonify, send_file, abort
from app import db
from app.models.image import Image
import os
from PIL import Image as PILImage
import io
import mimetypes
from app.utils.utils import rd_to_image
import numpy as np

bp = Blueprint("image", __name__, url_prefix="/api/images")

@bp.route("/<int:id>", methods=["GET"])
def list_images(id):
    data = Image.query.filter_by(isDelete=False, dataset_id=id).all()
    return jsonify([{"id": d.id, "dataset_id": d.dataset_id, "name": d.name, "modality": d.modality, "path": d.path, "label": d.label} for d in data])

@bp.route("/", methods=["POST"])
def create_images():
    data = request.json
    item = Image(dataset_id=data["dataset_id"], name=data["name"], modality=data["modality"], path=data["path"], label=data["label"])
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Image created", "id": item.id})

@bp.route("/<int:id>", methods=["PUT"])
def update_images(id):
    item = Image.query.get_or_404(id)
    data = request.json
    item.dataset_id = data.get("dataset_id", item.dataset_id)
    item.name = data.get("name", item.name)
    item.modality = data.get("modality", item.modality)
    item.path = data.get("path", item.path)
    item.label = data.get("label", item.label)
    db.session.commit()
    return jsonify({"message": "Image updated"})

@bp.route("/<int:id>", methods=["DELETE"])
def delete_images(id):
    item = Image.query.get_or_404(id)
    item.isDelete = True
    db.session.commit()
    return jsonify({"message": "Image deleted"})

SUPPORTED_FORMATS = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp"
]

# 获取图片文件（根据文件路径）
@bp.route("/file", methods=["GET"])
def get_image_file():
    image_path = request.args.get("image_path")
    if not image_path:
        abort(400, description="Missing 'image_path' parameter")

    if not os.path.isfile(image_path):
        abort(404, description="Image not found")

    mime_type, _ = mimetypes.guess_type(image_path)

    # 如果是 .npy 文件
    if image_path.endswith(".npy"):
        try:
            data = np.load(image_path)
            buffer = rd_to_image(data)
            return send_file(
                buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name=f"{os.path.splitext(os.path.basename(image_path))[0]}.png",
            )
        except Exception as e:
            abort(500, description=f"Image conversion failed: {str(e)}")

    # 其他情况：尝试原格式返回，否则转为 PNG
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type in SUPPORTED_FORMATS:
        return send_file(
            image_path,
            mimetype=mime_type,
            as_attachment=False,
            download_name=os.path.basename(image_path),
        )

    try:
        image = PILImage.open(image_path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="image/png",
            as_attachment=False,
            download_name=f"{os.path.splitext(os.path.basename(image_path))[0]}.png"
        )
    except Exception as e:
        abort(500, description=f"Image conversion failed: {str(e)}")