import os
import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from collections import defaultdict

app = Flask(__name__)

# ===== 数据库连接配置 =====
# 根据你的实际情况修改用户名、密码、库名
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@localhost:3306/embedded_software"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ===== 数据表模型 =====
class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True, comment='id')
    dataset_id = db.Column(db.BigInteger, nullable=False, comment='所属数据集ID')
    name = db.Column(db.String(128), nullable=False, comment='所属数据集名称')
    modality = db.Column(db.String(128), nullable=False, comment='模态类型(SAR, RD, 1D)')
    path = db.Column(db.String(256), nullable=False, comment='数据文件路径（绝对路径）')
    label = db.Column(db.String(128), nullable=False, comment='标签')
    createTime = db.Column(db.DateTime, default=datetime.datetime.now, comment='创建时间')
    updateTime = db.Column(db.DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    isDelete = db.Column(db.Integer, default=0, nullable=False, comment='是否删除')


def insert_images_per_class(base_dir, dataset_id, dataset_name, modality, per_class_limit=10):
    """
    每个类别取固定数量的图片插入数据库
    """
    inserted_count = defaultdict(int)
    total_inserted = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')):
                abs_path = os.path.abspath(os.path.join(root, file))

                # 标签：倒数第二级目录作为类别
                parts = abs_path.split(os.sep)
                label = parts[-2]

                # 如果这个类别已经达到限制，就跳过
                if inserted_count[label] >= per_class_limit:
                    continue

                image = Image(
                    dataset_id=dataset_id,
                    name=dataset_name,
                    modality=modality,
                    path=abs_path,
                    label=label
                )
                db.session.add(image)
                inserted_count[label] += 1
                total_inserted += 1

    db.session.commit()
    print(f"完成！共插入 {total_inserted} 条记录，每类最多 {per_class_limit} 条")
    print(dict(inserted_count))


if __name__ == '__main__':
    with app.app_context():
        base_dir = "D:/Project/knowledge_emb/data/RD"
        insert_images_per_class(
            base_dir=base_dir,
            dataset_id=3,
            dataset_name='RD',
            modality='RD',
            per_class_limit=10  # 每个类别取 10 张
        )