from app import create_app, db
from app.utils.celery_worker import make_celery

flask_app = create_app()
celery_app = make_celery(flask_app)
