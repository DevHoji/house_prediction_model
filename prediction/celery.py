import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'prediction.settings')

app = Celery('prediction')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

app.conf.beat_schedule = {
    'retrain-ml-model-every-24h': {
        'task': 'predict.tasks.retrain_model',
        'schedule': crontab(minute=0, hour=0),  # every day at midnight
    },
}
