from app import db
from datetime import datetime

class FeatureExtractor(db.Model):
    __tablename__ = 'feature_extractors'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    modality = db.Column(db.String(128), nullable=False)
    params_schema = db.Column(db.JSON)
    description = db.Column(db.Text)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)

class FeatureGroup(db.Model):
    __tablename__ = "feature_groups"

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False, comment="组合名称")
    description = db.Column(db.Text, comment="组合描述")
    createTime = db.Column(db.DateTime, server_default=db.func.now())
    updateTime = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
    isDelete = db.Column(db.Boolean, default=False)

    items = db.relationship("FeatureGroupItem", backref="group", cascade="all, delete-orphan")


class FeatureGroupItem(db.Model):
    __tablename__ = "feature_group_items"

    id = db.Column(db.BigInteger, primary_key=True)
    group_id = db.Column(db.BigInteger, db.ForeignKey("feature_groups.id", ondelete="CASCADE"))
    feature_extractor_id = db.Column(db.BigInteger, db.ForeignKey("feature_extractors.id", ondelete="CASCADE"))

    feature_extractor = db.relationship("FeatureExtractor", backref="group_links")