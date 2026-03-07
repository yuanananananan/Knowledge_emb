from flask import Blueprint, request, jsonify
from app import db
from app.models.feature_extractor import FeatureExtractor, FeatureGroup, FeatureGroupItem
from app.routes.user_api import login_required
import os
import ast
bp = Blueprint("feature_extractor", __name__, url_prefix="/api/feature_extractor")

@bp.route("/", methods=["GET"])
@login_required(role=["admin", "user"])
def list_feature_extractors():
    data = FeatureExtractor.query.filter_by(isDelete=False).all()
    return jsonify([{"id": d.id, "name": d.name, "modality": d.modality, "params_schema": d.params_schema, "description": d.description} for d in data])

@bp.route("/", methods=["POST"])
@login_required(role=["admin", "user"])
def list_feature_names_by_ids():
    data = request.get_json()
    ids = data.get("ids", [])

    if not isinstance(ids, list):
        return jsonify({"error": "ids must be a list"}), 400

    # 查询所有匹配 id 且未删除的对象
    rows = FeatureExtractor.query.filter(
        FeatureExtractor.id.in_(ids),
        FeatureExtractor.isDelete == False
    ).all()

    # 返回 name 列表
    names = [r.name for r in rows]

    return jsonify(names)


@bp.route("/", methods=["POST"])
@login_required(role="admin")
def create_feature_extractors():
    data = request.json
    item = FeatureExtractor(name=data["name"], modality=data["modality"], description=data["description"]) #params_schema=data["params_schema"]
    db.session.add(item)
    db.session.commit()
    return jsonify({"message": "FeatureExtractor created", "id": item.id})

@bp.route("/<int:id>", methods=["PUT"])
@login_required(role="admin")
def update_feature_extractors(id):
    item = FeatureExtractor.query.get_or_404(id)
    data = request.json
    item.name = data.get("name", item.name)
    item.modality = data.get("modality", item.modality)
    item.params_schema = data.get("params_schema", item.params_schema)
    item.description = data.get("description", item.description)
    db.session.commit()
    return jsonify({"message": "FeatureExtractor updated"})

@bp.route("/<int:id>", methods=["DELETE"])
@login_required(role="admin")
def delete_feature_extractors(id):
    item = FeatureExtractor.query.get_or_404(id)
    item.isDelete = True
    db.session.commit()
    return jsonify({"message": "FeatureExtractor deleted"})

# 根据组合ID查询该组合下所有特征
@bp.route("/group/<int:group_id>", methods=["GET"])
@login_required(role=["admin", "user"])
def get_features_by_group(group_id):
    group = FeatureGroup.query.filter_by(id=group_id, isDelete=False).first_or_404()
    features = [
        {
            "id": item.feature_extractor.id,
            "name": item.feature_extractor.name,
            "modality": item.feature_extractor.modality,
            "params_schema": item.feature_extractor.params_schema,
            "description": item.feature_extractor.description
        }
        for item in group.items
    ]
    return jsonify({
        "group_id": group.id,
        "group_name": group.name,
        "features": features
    })


# 修改组合名称或描述
@bp.route("/group/<int:group_id>", methods=["PUT"])
@login_required(role="admin")
def update_feature_group(group_id):
    group = FeatureGroup.query.filter_by(id=group_id, isDelete=False).first_or_404()
    data = request.json
    group.name = data.get("name", group.name)
    group.description = data.get("description", group.description)
    db.session.commit()
    return jsonify({"message": "FeatureGroup updated"})

# 删除组合
@bp.route("/group/<int:group_id>", methods=["DELETE"])
@login_required(role="admin")
def delete_feature_group(group_id):
    group = FeatureGroup.query.filter_by(id=group_id, isDelete=False).first_or_404()
    group.isDelete = True
    db.session.commit()
    return jsonify({"message": "FeatureGroup deleted"})

# 查询所有特征组合
@bp.route("/group/", methods=["GET"])
@login_required(role=["admin", "user"])
def list_feature_groups():
    groups = FeatureGroup.query.filter_by(isDelete=False).all()
    result = [
        {
            "id": g.id,
            "name": g.name,
            "description": g.description,
            "feature_count": len(g.items),
            "createTime": g.createTime
        } for g in groups
    ]
    return jsonify(result)

# 通过传入特征ID列表直接创建组合
# json body示例
# {
#   "name": "基础特征组合",
#   "features": [1, 2, 3, 4]
# }
@bp.route("/group/", methods=["POST"])
@login_required(role="admin")
def create_group_from_feature_list():
    data = request.json
    name = data.get("name")
    description = data.get("description", "")
    feature_ids = data.get("features", [])

    if not name or not feature_ids:
        return jsonify({"error": "Missing name or features"}), 400

    group = FeatureGroup(name=name, description=description)
    db.session.add(group)
    db.session.flush()

    for fid in feature_ids:
        db.session.add(FeatureGroupItem(group_id=group.id, feature_extractor_id=fid))

    db.session.commit()
    return jsonify({"message": "FeatureGroup created", "group_id": group.id})

# 特征提取算子代码存放的Python文件路径
FEATURE_EXTRACTOR_FILE = "D:/Project/knowledge_emb/utils/get_feature.py"

def append_feature_code_to_file(code):
    """将特征提取函数代码追加到Python文件中"""
    # 确保文件存在，不存在则创建
    if not os.path.exists(FEATURE_EXTRACTOR_FILE):
        with open(FEATURE_EXTRACTOR_FILE, 'w', encoding='utf-8') as f:
            f.write("# 自动生成的特征提取算子代码\n")
            f.write("import cv2\nimport numpy as np\n\n")  # 预导入常用库

    # 追加新函数（假设用户保证函数名唯一）
    with open(FEATURE_EXTRACTOR_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{code}")

# 通过上传 JSON 文件添加组合及特征
@bp.route("/upload_json", methods=["POST"])
@login_required(role="admin")
def upload_group_json():
    # todo :需要知道一个标准json格式示例
    data = request.get_json()
    group_name = data.get("name")
    group_description = data.get("description", "")
    features = data.get("features", [])  # 包含详细信息的特征列表

    # 基础校验
    if not group_name or not features:
        return jsonify({"error": "Missing group name or features list"}), 400

    # 创建特征组合
    feature_group = FeatureGroup(
        name=group_name,
        description=group_description
    )
    db.session.add(feature_group)
    db.session.flush()  # 刷新以获取group的id

    # 处理每个特征
    for feat in features:
        # 校验单个特征的必要字段
        required_fields = ["name", "description", "modality", "code"]
        if not all(field in feat for field in required_fields):
            db.session.rollback()  # 回滚之前的操作
            return jsonify({"error": f"Feature missing required fields: {required_fields}"}), 400

        # 从提供的代码中提取函数名，确保代码是合法的函数定义
        code_first_line = feat["code"].strip().splitlines()[0] if feat["code"].strip() else ""
        if not code_first_line.startswith("def "):
            db.session.rollback()
            return jsonify({"error": f"Feature code must start with 'def ': {feat['name']}"}), 400

        # 提取函数名称，例如从"def get_spatial_feat(data):"中提取"get_spatial_feat"
        func_name = code_first_line.split("def ")[1].split("(")[0].strip()
        if not func_name:
            db.session.rollback()
            return jsonify({"error": f"Invalid function name in feature: {feat['name']}"}), 400

        # 创建特征提取器记录（存入数据库，直接使用用户定义的函数名）
        feature_extractor = FeatureExtractor(
            name=feat["name"], #这里需要和func_name完全保持一致，否则后续无法调用（那我这里要不要直接强制赋值）
            description=feat["description"],
            modality=feat["modality"]  # 例如："spatial", "spectral", "temporal"
        )
        db.session.add(feature_extractor)
        db.session.flush()  # 刷新以获取feature_extractor的id

        # 建立特征组合与特征的关联
        group_item = FeatureGroupItem(
            group_id=feature_group.id,
            feature_extractor_id=feature_extractor.id
        )
        db.session.add(group_item)

        # 将特征提取函数代码写入Python文件
        try:
            # 校验 Python 语法
            code_str = feat["code"]
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                return jsonify({"error": f"Invalid function code: {e}"}), 400
            append_feature_code_to_file(feat["code"])  # 直接使用用户提供的代码
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to write feature code: {str(e)}"}), 500

    # # 二次验证（尝试导入执行）
    # try:
    #     # 重新加载模块检查可执行性
    #     import importlib.util
    #     spec = importlib.util.spec_from_file_location("get_feature", FEATURE_EXTRACTOR_FILE)
    #     module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(module)
    # except Exception as e:
    #     # 回滚文件
    #     with open(FEATURE_EXTRACTOR_FILE, "w", encoding="utf-8") as f:
    #         f.write(original_content)
    #     return jsonify({"error": f"Failed to import code: {str(e)}"}), 400

    # 提交所有数据库操作
    db.session.commit()

    return jsonify({
        "message": "FeatureGroup created successfully",
        "group_id": feature_group.id,
        "feature_count": len(features),
        "extractor_file": FEATURE_EXTRACTOR_FILE
    })

