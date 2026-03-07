from app import db
from datetime import datetime

class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    modality = db.Column(db.String(128), nullable=False)
    path = db.Column(db.String(256), nullable=False)
    labels = db.Column(db.String(128), nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
