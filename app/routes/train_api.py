from flask import Blueprint, request, jsonify, Response
from app import db
from app.models.train_job import TrainJob
from app.models.dataset import Dataset
from app.models.backbone import Backbone
from app.models.feature_extractor import FeatureExtractor
from app.utils.tasks import run_training_task
from app.utils.requestConfig2json import json_extra_to_json
import os
import time
import json
from datetime import datetime
from celery.result import AsyncResult
from app.utils.celery_worker import celery_app
from app.utils.utils import config_to_json
from utils.utils import load_config
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# from app.utils.id_generator import generate_id

bp = Blueprint("train", __name__, url_prefix="/api/train")

@bp.route("/", methods=["GET"])
def list_train_jobs():
    data = TrainJob.query.filter_by(isDelete=False).all()
    return jsonify([
        {
            "id": d.id,
            "job_id": d.job_id,
            "user_id": d.user_id,
            "dataset_id": d.dataset_id,
            "nodes": d.nodes,
            "config": d.config,
            "description": d.description,
            "backbone_id": d.backbone_id,
            "classifier_id": d.classifier_id,
            "feature_ids": d.feature_ids,
            "status": d.status,
            "progress": d.progress,
            "result": d.result,
            "model_path": d.model_path,
            "log_path": d.log_path,
            "createTime": d.createTime.isoformat() if d.createTime else None,
            "updateTime": d.updateTime.isoformat() if d.updateTime else None
        }
        for d in data
    ])

@bp.route("/<string:celery_task_id>", methods=["DELETE"])
def delete_train_job(celery_task_id):
    item = TrainJob.query.filter_by(job_id=celery_task_id).first()
    item.isDelete = True
    db.session.commit()
    return jsonify({"message": "Train job deleted"})

@bp.route("/start", methods=["POST"])
def start_train_jobs():
    data = request.get_json()
    # todo:这里临时先改成使用默认的config文件，前端先传进来空
    # cfg = load_config('D:/Project/knowledge_emb/configs/sar_config.yaml')
    # static_config = config_to_json(cfg)
    config_flatten_json = data.get("extra", [])
    # 查询数据库所有匹配 id 且未删除的对象
    feature_ext_ids = data.get("feature_ids", [])
    rows = FeatureExtractor.query.filter(
        FeatureExtractor.id.in_(feature_ext_ids),
        FeatureExtractor.isDelete == False
    ).all()
    # 返回 name 列表
    external_feature_list = [r.name for r in rows]
    # 配置文件不再以本地yaml文件形式出现，而是直接存储到数据库中
    config = json_extra_to_json(config_flatten_json, external_feature_list)
    # print(type(config))
    # config = json.loads(config)
    # 处理description字段
    description = data.get("description", "")
    if not description:  # 如果description为空
        # 1. 查询数据集名称
        dataset = Dataset.query.get(data["dataset_id"])
        if not dataset:
            return jsonify({"error": f"数据集ID {data['dataset_id']} 不存在"}), 400
        dataset_name = dataset.name  # 假设数据集模型有name字段

        # 2. 查询主干模型名称
        backbone = Backbone.query.get(data["backbone_id"])
        if not backbone:
            return jsonify({"error": f"主干模型ID {data['backbone_id']} 不存在"}), 400
        backbone_name = backbone.name  # 假设主干模型有name字段

        # 3. 获取当前创建时间（格式化为YYYYMMDDHHMMSS，避免特殊字符）
        create_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # 4. 生成唯一编号（查询同类任务数量+1）
        # todo: 目前统计相同数据集和主干模型的任务数量作为基础编号
        task_count = TrainJob.query.filter(
            TrainJob.dataset_id == data["dataset_id"],
            TrainJob.backbone_id == data["backbone_id"]
        ).count()
        unique_id = task_count + 1  # 编号从1开始

        # 5. 组合生成description
        description = f"{dataset_name}_{backbone_name}_{create_time}_{unique_id}"

    # 解析参数
    job = TrainJob(
        user_id=data["user_id"],
        description=description,
        dataset_id=data["dataset_id"],
        nodes=data["nodes"],
        backbone_id=data["backbone_id"],
        classifier_id=data["classifier_id"],
        config = config, # data["config"],
        feature_ids=data["feature_ids"],
        adaptive=data["adaptive"],
        status="pending",
    )
    db.session.add(job)
    db.session.commit()

    # 异步任务启动
    task = run_training_task.delay(job.id)

    # 把 Celery 生成的 task_id 更新回 job.job_id
    job.job_id = task.id
    db.session.commit()

    return jsonify({
        "message": "训练任务已启动",
        "id": job.id,
        "job_id": task.id,
        "celery_task_id": task.id
    })

@bp.route("/status/<string:celery_task_id>", methods=["GET"])
def get_job_status(celery_task_id):
    # task = AsyncResult(celery_task_id, app=celery_app)
    # return jsonify({
    #     "state": task.state,
    #     "info": str(task.info) if task.info else None
    # })
    print(celery_task_id)
    job = TrainJob.query.filter_by(job_id=celery_task_id).first()
    return jsonify({"status": job.status, "progress": job.progress})

@bp.route("logs/<job_id>", methods=["GET"])
def stream_log(job_id):
    # job = TrainJob.query.filter_by(job_id=job_id).first()
    # job_id = job.id
    def generate():
        log_path = f"D:/Project/knowledge_emb/logs/{job_id}.log"
        last_size = 0

        while True:
            if os.path.exists(log_path):
                current_size = os.path.getsize(log_path)
                if current_size > last_size:
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(last_size)
                        new_lines = f.read()
                        for line in new_lines.strip().splitlines():
                            yield f"data: {line}\n\n"
                    last_size = current_size
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")

@bp.route("/logs_file/<string:job_id>", methods=["GET"])
def get_log_by_job(job_id):
    path = f"D:/Project/knowledge_emb/logs/{job_id}.log"
    if not os.path.exists(path):
        return jsonify({"message": "Log not found"}, status=404)

    with open(path, 'r', encoding='utf-8') as f:
        return jsonify({"log": f.read()})


@bp.route("/result/<string:job_id>", methods=["GET"])
def get_result_by_job_id(job_id):
    job = TrainJob.query.filter_by(job_id=job_id).first()
    return jsonify({"result": job.result})

@bp.route("/config/<job_id>", methods=["GET"])
def get_config_by_job_id(job_id):
    job = TrainJob.query.filter_by(job_id=job_id, isDelete=0).first()
    if not job:
        return jsonify({"error": "job_id not found"}), 404
    return jsonify(job.config or {})