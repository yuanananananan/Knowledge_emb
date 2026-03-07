from torch_geometric.graphgym.register import metric_dict

from train import train   # 根目录下 train.py 中的函数
from val import val
from dataset.sar import get_sar_dataset_full
from model.ked import KED
from utils.utils import get_model, get_dataloader
from app.utils.celery_worker import celery_app
from app import db
from app.models.train_job import TrainJob
from app.models.test_record import TestRecord
from app.utils.log_streamer import append_log
from app.utils.utils import json_to_config
from torch.utils.data import DataLoader
from datetime import datetime
from celery import Celery
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


@celery_app.task(bind=True)
def run_training_task(self, job_id):
    job = None
    try:
        job = TrainJob.query.get(job_id)
        if not job:
            return {"error": "Invalid job_id"}

        job.status = "running"
        db.session.commit()
        config = job.config
        config_dict = json_to_config(config)
        print(config_dict)
        # 训练
        epoches = config_dict['training']['epochs']
        # 数据加载
        train_loader, test_loader = get_dataloader(config_dict)
        # 模型加载
        model = get_model(config_dict)

        # 定义回调函数用于进度更新
        def update_progress(epoch):
            job.progress = int((epoch + 1) / epoches * 100)
            job.updateTime = datetime.now()
            db.session.commit()

        job_id = job.job_id
        append_log(job_id, "开始训练")
        # 启动训练
        # train(epoches, model, train_loader, test_loader, job_id, update_fn=update_progress)
        _, result, model_path, log_path = train(config_dict, model, train_loader, test_loader, job_id, update_fn=update_progress)
        append_log(job_id, "训练结束")
        job.status = "completed"
        job.result = result
        job.model_path = model_path
        job.log_path = log_path
        job.updateTime = datetime.now()
        db.session.commit()


    except Exception as e:
        job.status = "failed"
        job.result = {"error": str(e)}
        job.updateTime = datetime.now()
        db.session.commit()
        raise

@celery_app.task(bind=True)
def run_valuating_task(self, job_id):
    job = None
    try:
        job = TestRecord.query.get(job_id)
        if not job:
            return {"error": "Invalid job_id"}

        job.status = "running"
        db.session.commit()
        config = job.config
        config_dict = json_to_config(config)
        print(config_dict)
        # 数据加载
        train_loader, test_loader = get_dataloader(config_dict)
        # 模型加载
        model = get_model(config_dict)

        job_id = job.val_job_id
        append_log(job_id, "开始测试")
        train_job_id = job.train_job_id
        model_path = f"model_ckp/{train_job_id}/model_best.pth"
        _, result, = val(config_dict, model, test_loader, model_path, job_id)
        append_log(job_id, "测试结束")
        job.status = "completed"
        job.result = result
        job.updateTime = datetime.now()
        db.session.commit()


    except Exception as e:
        job.status = "failed"
        job.result = {"error": str(e)}
        job.updateTime = datetime.now()
        db.session.commit()
        raise

