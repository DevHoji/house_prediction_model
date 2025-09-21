This folder contains example systemd unit files to run Celery worker and Celery beat as services.

To install and enable (requires root privileges):

sudo cp prediction-celery-worker.service /etc/systemd/system/
sudo cp prediction-celery-beat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable prediction-celery-worker
sudo systemctl enable prediction-celery-beat
sudo systemctl start prediction-celery-worker
sudo systemctl start prediction-celery-beat

Logs are written to the project's logs/ directory as configured in the unit files.
