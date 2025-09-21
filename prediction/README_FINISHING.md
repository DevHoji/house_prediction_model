Finish checklist for Task 2 - Automation & Scheduling

Completed:
- Celery app and Celery Beat (prediction/celery.py) with daily schedule at midnight UTC
- retrain_model task in predict/tasks.py with:
  - simulated incremental data
  - lock file to prevent overlap
  - absolute logs at ./logs/retrain.log
  - optional remote fetch via DATA_FETCH_URL (CSV/JSON), with retry fallback
  - model and scaler persisted to predict/ml_model.joblib and predict/scaler.joblib
- UI route /pr/ wired to predict app and template present
- Example systemd unit files in deployment/

Remaining optional tasks (can be completed for production):
- If you want remote data ingestion in production, set DATA_FETCH_URL to a URL that returns CSV or JSON rows with columns: size, bedrooms, age, price
- Enable systemd units (requires sudo): see deployment/*.service
- Switch to DEBUG=False and move secrets to env vars
- Add testing/CI and model validation / rollback strategy

How to verify locally (commands):
1) Start Redis
   sudo systemctl start redis
2) Start worker/beat/flower
   cd prediction
   ./predict/management/commands/run_worker_services.sh
3) Dispatch a retrain
   source myenv/bin/activate
   export PYTHONPATH=$(pwd)
   python manage.py shell -c "from predict.tasks import retrain_model; print(retrain_model.delay())"
4) Check logs and artifacts
   tail -n 200 logs/retrain.log
   ls -l predict/ml_model.joblib predict/scaler.joblib
5) Start devserver and verify UI
   python manage.py runserver
   visit: http://127.0.0.1:8000/pr/

If you want, I can enable systemd unit files (requires sudo) to run worker+beat persistently. Alternatively, I can implement unit tests and CI pipeline if you prefer.
