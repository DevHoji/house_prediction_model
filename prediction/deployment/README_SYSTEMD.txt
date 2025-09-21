To install systemd units for the Celery worker and beat (requires sudo):

sudo cp deployment/prediction-celery-worker.service /etc/systemd/system/
sudo cp deployment/prediction-celery-beat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prediction-celery-worker
sudo systemctl enable --now prediction-celery-beat

Check status:
sudo systemctl status prediction-celery-worker
sudo systemctl status prediction-celery-beat
