from app import db
from datetime import datetime

class TestRecord(db.Model):
    __tablename__ = 'test_records'

    id = db.Column(db.BigInteger, primary_key=True)
    val_job_id = db.Column(db.String(64), nullable=False)
    train_job_id = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.BigInteger, nullable=False)
    config = db.Column(db.JSON, nullable=False)
    result = db.Column(db.JSON)
    status = db.Column(db.String(32), default='pending', nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
