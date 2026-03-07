from .user_api import bp as user_bp
from .dataset_api import bp as dataset_bp
from .train_api import bp as train_job_bp
# from .test_record_api import bp as test_record_bp
from .feature_extractor_api import bp as feature_bp
from .backbone_api import bp as backbone_bp
from .image_api import bp as image_bp
from .val_api import bp as val_bp
from .classifier_api import bp as classifier_bp
from .config_api import bp as config_bp
from django.urls import path

def register_routes(app):
    app.register_blueprint(user_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(train_job_bp)
    # app.register_blueprint(test_record_bp)
    app.register_blueprint(feature_bp)
    app.register_blueprint(backbone_bp)
    app.register_blueprint(classifier_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(val_bp)
    app.register_blueprint(config_bp)

