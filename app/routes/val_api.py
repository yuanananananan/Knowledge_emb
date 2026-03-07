from flask import Blueprint, request, jsonify, Response
from app import db
from app.models.test_record import TestRecord
from app.models.train_job import TrainJob
from app.utils.tasks import run_valuating_task
import os
import json
import time
from celery.result import AsyncResult
from app.utils.celery_worker import celery_app
# from app.utils.id_generator import generate_id

bp = Blueprint("val", __name__, url_prefix="/api/val")

@bp.route("/", methods=["GET"])
def list_val_jobs():
    data = TestRecord.query.filter_by(isDelete=False).all()
    return jsonify([
        {
            "id": d.id,
            "val_job_id": d.val_job_id,
            "train_job_id": d.train_job_id,
            "user_id": d.user_id,
            "config": d.config,
            "status": d.status,
            "result": d.result,
            "createTime": d.createTime.isoformat() if d.createTime else None,
            "updateTime": d.updateTime.isoformat() if d.updateTime else None
        }
        for d in data
    ])


@bp.route("/start", methods=["POST"])
def start_val_jobs():
    data = request.get_json()
    train_job_id = data["train_job_id"]
    train_data = TrainJob.query.filter_by(job_id=train_job_id).first()
    # 解析参数
    job = TestRecord(
        train_job_id=data["train_job_id"],
        user_id=data["user_id"],
        config=train_data.config,
        status="pending",
    )
    db.session.add(job)
    db.session.commit()

    # 异步任务启动
    task = run_valuating_task.delay(job.id)

    # 把 Celery 生成的 task_id 更新回 job.job_id
    job.val_job_id = task.id
    db.session.commit()

    return jsonify({
        "message": "测试任务已启动",
        "id": job.id,
        "val_job_id": task.id,
        "celery_task_id": task.id
    })

@bp.route("/status/<string:celery_task_id>", methods=["GET"])
def get_job_status(celery_task_id):
    print(celery_task_id)
    job = TestRecord.query.filter_by(val_job_id=celery_task_id).first()
    return jsonify({"status": job.status, "progress": job.progress})

@bp.route("/logs_file/<string:job_id>", methods=["GET"])
def get_log_by_job(job_id):
    path = f"D:/Project/knowledge_emb/logs/{job_id}.log"
    if not os.path.exists(path):
        return jsonify({"message": "Log not found"}, status=404)

    with open(path, 'r', encoding='utf-8') as f:
        return jsonify({"log": f.read()})


@bp.route("/result/<string:job_id>", methods=["GET"])
def get_result_by_job_id(job_id):
    job = TestRecord.query.filter_by(val_job_id=job_id).first()
    return jsonify({"result": job.result})

@bp.route("/validate", methods=["POST"])
def get_metrics_by_ids():
    data = request.get_json()
    if not data or "job_ids" not in data:
        return jsonify({"error": "Missing job_ids"}), 400

    job_ids = data["job_ids"]
    if not isinstance(job_ids, list) or len(job_ids) == 0:
        return jsonify({"error": "job_ids must be a non-empty list"}), 400

    # 查询数据库
    jobs = TestRecord.query.filter(TestRecord.val_job_id.in_(job_ids)).all()
    # todo: 加个校验，所有勾选的任务需要先独立测试完成

    if not jobs:
        return jsonify({"error": "No matching jobs found"}), 404

    results = []
    for job in jobs:
        try:
            result_data = job.result
            metrics = result_data.get("Metrics", {})
        except Exception:
            metrics = {}

        results.append({
            "id": job.val_job_id,
            "Metrics": metrics
        })

    return jsonify({"results": results})