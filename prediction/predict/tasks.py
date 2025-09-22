from celery import shared_task
import numpy as np
import pandas as pd
from .ml_model import MODEL_PATH, SCALER_PATH
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
import time
import traceback
import requests

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOCK_PATH = os.path.join(os.path.dirname(__file__), '..', 'retrain.lock')

os.makedirs(LOG_DIR, exist_ok=True)

def _write_log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(LOG_DIR, 'retrain.log'), 'a') as f:
        f.write(f"[{ts}] {msg}\n")

@shared_task
def retrain_model():
    """
    Retrain the ML model using simulated new data (or real data if you integrate a fetcher).
    This function uses a simple lock file to prevent overlapping retrains.
    """
    # Acquire lock (simple exclusive creation)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        _write_log('Retrain skipped: another retrain is already running.')
        return 'Skipped - already running'

    start_ts = time.time()
    try:
        _write_log('Retrain started')
        # Load existing data if available
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            # Simulate new data by generating more synthetic data
            np.random.seed()
            new_data = pd.DataFrame({
                'size': np.random.uniform(1000, 5000, 200),
                'bedrooms': np.random.randint(1, 6, 200),
                'age': np.random.uniform(0, 50, 200),
            })
            new_data['price'] = (
                new_data['size'] * 0.2 +
                new_data['bedrooms'] * 50000 -
                new_data['age'] * 1000 +
                np.random.normal(0, 50000, 200)
            )
            # For demo, just use new_data as the new dataset
            data = new_data
        else:
            # If no model exists, create initial data
            np.random.seed(42)
            data = pd.DataFrame({
                'size': np.random.uniform(1000, 5000, 1000),
                'bedrooms': np.random.randint(1, 6, 1000),
                'age': np.random.uniform(0, 50, 1000),
            })
            data['price'] = (
                data['size'] * 0.2 +
                data['bedrooms'] * 50000 -
                data['age'] * 1000 +
                np.random.normal(0, 50000, 1000)
            )

        DATA_FETCH_URL = os.environ.get('DATA_FETCH_URL')
        if DATA_FETCH_URL:
            try:
                _write_log(f'Fetching online data from {DATA_FETCH_URL}')
                resp = requests.get(DATA_FETCH_URL, timeout=20)
                resp.raise_for_status()
                fetched = resp.json()
                # expect fetched to be list of dicts with keys size, bedrooms, age, price
                fetched_df = pd.DataFrame(fetched)
                if {'size','bedrooms','age','price'}.issubset(fetched_df.columns):
                    data = fetched_df
                    _write_log(f'Using fetched data, {len(data)} rows')
                else:
                    _write_log('Fetched data missing required columns, falling back to simulated')
            except Exception as e:
                _write_log('Failed fetching online data: ' + repr(e))

        X = data[['size', 'bedrooms', 'age']]
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        # Ensure target directory exists
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(model, MODEL_PATH)
        elapsed = time.time() - start_ts
        _write_log(f'Retrain finished in {elapsed:.2f}s')
        return 'Model retrained and saved.'
    except Exception as e:
        _write_log('Retrain failed: ' + repr(e))
        _write_log(traceback.format_exc())
        raise
    finally:
        try:
            os.remove(LOCK_PATH)
        except Exception:
            pass
