import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prediction.settings")

app = Celery("prediction")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Celery Beat schedule for retraining every 24 hours
app.conf.beat_schedule = {
    'retrain-daily-midnight': {
        'task': 'predict.tasks.retrain_model',
        'schedule': crontab(minute=0, hour=0),
        'options': {'queue': 'celery'},
    },
}

# Keep backward-compatible connect handler
def setup_periodic_tasks(sender, **kwargs):
    from predict.tasks import retrain_model
    sender.add_periodic_task(
        crontab(minute=0, hour=0),  # Every day at midnight
        retrain_model.s(),
        name="Retrain ML model every 24 hours"
    )

app.on_after_finalize.connect(setup_periodic_tasks)

# To run the Flower monitoring tool for Celery, use the following command:
# celery -A prediction flower
