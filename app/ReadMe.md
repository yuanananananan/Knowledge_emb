
[//]: # (celery -A pipeline.celery_worker.celery_app worker --loglevel=info)


[//]: # (启动 FastAPI 主服务)

[//]: # (uvicorn main:app --reload)


安装
pip install timm==0.6.11    
pip install celery[redis] redis


sudo apt install redis-server
redis-server


[//]: # (celery -A app.utils.tasks.celery_app worker --pool=threads --loglevel=info)
启动 Celery Worker(切换到项目根目录)
celery -A app.utils.run_celery_worker.celery_app worker --pool=threads --loglevel=info
