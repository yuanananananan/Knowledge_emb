from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app import db
from app.models.dataset import Dataset
from app.models.image import Image
import zipfile
import tempfile
import os
import json
from app.utils.utils import insert_images_per_class

bp = Blueprint("dataset", __name__, url_prefix="/api/datasets")
# 允许的压缩包扩展名
ALLOWED_EXTENSIONS = {"zip"}
UPLOAD_BASE = "data/"  # 数据集解压根目录

@bp.route("/", methods=["GET"])
def list_datasets():
    data = Dataset.query.filter_by(isDelete=False).all()
    return jsonify([{"id": d.id, "name": d.name, "modality": d.modality, "path": d.path} for d in data])

@bp.route("/", methods=["POST"])
def create_dataset():
    data = request.json
    item = Dataset(name=data["name"], modality=data["modality"], path=data["path"])
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "Dataset created", "id": item.id})

@bp.route("/<int:id>", methods=["PUT"])
def update_dataset(id):
    item = Dataset.query.get_or_404(id)
    data = request.json
    item.name = data.get("name", item.name)
    item.modality = data.get("modality", item.modality)
    item.path = data.get("path", item.path)
    db.session.commit()
    return jsonify({"message": "Dataset updated"})

@bp.route("/<int:id>", methods=["DELETE"])
def delete_dataset(id):
    item = Dataset.query.get_or_404(id)
    item.isDelete = True
    Image.query.filter_by(id=item.id).delete()
    db.session.commit()
    return jsonify({"message": "Dataset deleted"})

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/upload", methods=["POST"])
def upload_dataset():
    modality = request.form.get("modality")
    if not modality:
        return jsonify({"error": "Missing modality parameter"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # 临时保存上传文件
    filename = secure_filename(file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)

    # 计算解压路径（相对路径）
    # 数据集名从文件名推断
    dataset_name = os.path.splitext(filename)[0]
    project_root = os.path.dirname(current_app.root_path)
    extract_dir = os.path.join(project_root, UPLOAD_BASE)
    dataset_dir = os.path.join(extract_dir, dataset_name)
    relate_dir = os.path.relpath(dataset_dir, project_root)
    os.makedirs(extract_dir, exist_ok=True)

    # 解压 zip 文件到 UPLOAD_BASE
    try:
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        os.remove(temp_path)
        return jsonify({"error": "Invalid ZIP file"}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 自动扫描一级子目录作为标签
    labels = []
    extract_dir = os.path.join(extract_dir, dataset_name)
    for entry in os.listdir(extract_dir):
        sub_path = os.path.join(extract_dir, entry)
        if os.path.isdir(sub_path):
            labels.append(entry)

    if not labels:
        return jsonify({"error": "No label folders found in dataset"}), 400


    # 写入数据库
    dataset = Dataset(
        name=dataset_name,
        modality=modality,
        path=relate_dir,
        labels=json.dumps(labels, ensure_ascii=False),  # 存为 JSON 字符串
    )
    db.session.add(dataset)
    db.session.commit()

    # 开始写入实际图像数据到数据库
    with current_app.app_context():
        base_dir = os.path.join("D:/Project/knowledge_emb/data/", dataset_name)
        insert_images_per_class(
            db=db,
            base_dir=base_dir,
            dataset_id=dataset.id,
            dataset_name=dataset_name,
            modality=dataset_name,
            per_class_limit=10  # 每个类别取 10 张
        )

    return jsonify({
        "message": "Dataset uploaded successfully",
        "id": dataset.id,
        "name": dataset.name,
        "path": dataset.path,
        "labels": labels
    })