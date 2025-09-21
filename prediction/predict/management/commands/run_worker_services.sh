#!/bin/bash
set -e
# Helper script to start worker, beat and flower for development
export PYTHONPATH=$(pwd)
VENV_BIN="/home/hojiwaq/Desktop/SAT/myenv/bin"
mkdir -p logs
# start worker
${VENV_BIN}/celery -A prediction.celery:app worker --loglevel=info --logfile=$(pwd)/logs/celery_worker.log &
# start beat
${VENV_BIN}/celery -A prediction.celery:app beat --loglevel=info --logfile=$(pwd)/logs/celery_beat.log &
# start flower
${VENV_BIN}/celery -A prediction.celery:app flower --port=5555 --address=127.0.0.1 --basic_auth=admin:admin --broker=redis://localhost:6379/0 &

echo "Started worker, beat, flower. Logs in ./logs. Flower: http://127.0.0.1:5555 (admin/admin)"
