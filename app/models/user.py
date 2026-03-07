from app import db
from datetime import datetime

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    userName = db.Column(db.String(256))
    userAccount = db.Column(db.String(256))
    userPassword = db.Column(db.String(512), nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
    userRole = db.Column(db.String(256), default='user', nullable=False)
