from app import db
from datetime import datetime

class TrainJob(db.Model):
    __tablename__ = 'train_jobs'

    id = db.Column(db.BigInteger, primary_key=True)
    job_id = db.Column(db.String(128), nullable=True)
    description = db.Column(db.String(256), nullable=True)
    user_id = db.Column(db.BigInteger, nullable=False)
    dataset_id = db.Column(db.BigInteger, nullable=False)
    nodes = db.Column(db.JSON, nullable=False)
    backbone_id = db.Column(db.BigInteger, nullable=False)
    classifier_id = db.Column(db.BigInteger, nullable=False)
    config = db.Column(db.JSON, nullable=False)
    feature_ids = db.Column(db.JSON, nullable=False)
    adaptive = db.Column(db.Boolean, nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False)
    progress = db.Column(db.Integer, default=0)
    result = db.Column(db.JSON)
    model_path = db.Column(db.String(128))
    log_path = db.Column(db.String(128))
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)

