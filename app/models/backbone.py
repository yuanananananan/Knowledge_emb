from app import db
from datetime import datetime

class Backbone(db.Model):
    __tablename__ = 'backbones'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    params_schema = db.Column(db.JSON, nullable=False)
    description = db.Column(db.Text)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
