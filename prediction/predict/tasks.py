from celery import shared_task
import numpy as np
import pandas as pd
from .ml_model import MODEL_PATH, SCALER_PATH
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import time
from pathlib import Path
import json
from dotenv import load_dotenv

# load environment variables from project .env if present
load_dotenv()

# Paths
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOG_FILE = LOG_DIR / 'retrain.log'
LOCK_PATH = Path(__file__).resolve().parent.parent / 'retrain.lock'

# Temp paths for safe swap
MODEL_TMP = Path(str(MODEL_PATH) + '.tmp')
SCALER_TMP = Path(str(SCALER_PATH) + '.tmp')
BACKUP_MODEL = Path(str(MODEL_PATH) + '.bak')
BACKUP_SCALER = Path(str(SCALER_PATH) + '.bak')

# Ensure log directory and file exist with safe permissions
LOG_DIR.mkdir(parents=True, exist_ok=True)
try:
    LOG_FILE.touch(exist_ok=True)
    LOG_FILE.chmod(0o644)
except Exception:
    # best-effort: ignore permission errors here
    pass


def _write_log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp} - {msg}\n")
    except Exception:
        # avoid raising from logging to not break the task
        pass


# Try to import requests, otherwise use urllib
try:
    import requests
except Exception:
    requests = None
    import urllib.request
    import urllib.error


def _fetch_remote_data(url: str, timeout: int = 10, retries: int = 2):
    """Attempt to fetch CSV or JSON data from URL. Returns DataFrame or raises.
    Accepts CSV (content-type text/csv) or JSON array of objects or wrapper {"data": [...]}.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            if requests:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                text = resp.text
            else:
                with urllib.request.urlopen(url, timeout=timeout) as resp_obj:
                    info = resp_obj.info()
                    content_type = info.get_content_type() if hasattr(info, 'get_content_type') else ''
                    text = resp_obj.read().decode('utf-8')

            # Try JSON body (either array or wrapper {"data": [...]})
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and 'data' in parsed and isinstance(parsed['data'], list):
                    return pd.DataFrame(parsed['data'])
                if isinstance(parsed, list):
                    return pd.DataFrame(parsed)
                if isinstance(parsed, dict):
                    try:
                        return pd.json_normalize(parsed)
                    except Exception:
                        pass
            except Exception:
                pass

            # Try pandas read_json (handles some JSON shapes)
            try:
                data_json = pd.read_json(text)
                if isinstance(data_json, (pd.DataFrame, pd.Series)):
                    return pd.DataFrame(data_json)
            except Exception:
                pass

            # Try CSV
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                return df
            except Exception:
                pass

            raise ValueError('Unsupported remote data format')
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
            continue
    raise last_exc


@shared_task
def retrain_model():
    # Prevent overlapping retrains using a simple lock file
    if LOCK_PATH.exists():
        _write_log('Retrain requested but lock present; another retrain is running.')
        return 'Retrain skipped; already running.'

    try:
        # create lock
        with open(LOCK_PATH, 'w') as lf:
            lf.write(str(os.getpid()))

        _write_log('Retrain started.')

        # Load existing data if available
        data = None
        data_source = 'simulated'
        # If a DATA_FETCH_URL env var is present, attempt to fetch remote data
        fetch_url = os.environ.get('DATA_FETCH_URL')
        if fetch_url:
            try:
                _write_log(f'Attempting to fetch remote data from {fetch_url}')
                remote_df = _fetch_remote_data(fetch_url, timeout=10, retries=2)
                # Ensure required columns exist
                if {'size', 'bedrooms', 'age', 'price'}.issubset(set(remote_df.columns)):
                    data = remote_df[['size', 'bedrooms', 'age', 'price']].dropna()
                    data_source = 'remote'
                    _write_log(f'Remote data fetched with {len(data)} rows')
                else:
                    _write_log('Remote data missing required columns; falling back to simulation')
            except Exception as e:
                _write_log(f'Remote data fetch failed: {repr(e)}; falling back to simulation')

        if data is None:
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
                data_source = 'simulated_incremental'
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
                data_source = 'simulated_initial'

        _write_log(f'Using data source: {data_source}, rows={len(data)}')

        X = data[['size', 'bedrooms', 'age']]
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        # Write to temp files first
        joblib.dump(scaler, SCALER_TMP)
        joblib.dump(model, MODEL_TMP)

        # Validate new model against existing (if exists)
        swap_ok = True
        new_mse = None
        old_mse = None
        try:
            X_test_scaled = scaler.transform(X_test)
            preds_new = model.predict(X_test_scaled)
            new_mse = float(mean_squared_error(y_test, preds_new))
            _write_log(f'New model MSE: {new_mse}')

            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                # load old model
                old_scaler = joblib.load(SCALER_PATH)
                old_model = joblib.load(MODEL_PATH)
                old_preds = old_model.predict(old_scaler.transform(X_test))
                old_mse = float(mean_squared_error(y_test, old_preds))
                _write_log(f'Old model MSE: {old_mse}')
                # accept new model only if mse is not worse by more than 1% (tolerance)
                if new_mse > old_mse * 1.01:
                    swap_ok = False
                    _write_log('New model is worse than existing model; will rollback to old model.')
        except Exception as e:
            # If validation fails, be conservative and keep old model
            swap_ok = False
            _write_log(f'Validation failed: {repr(e)}; will keep existing model if present')

        if swap_ok:
            # backup old model
            try:
                if os.path.exists(MODEL_PATH):
                    os.replace(MODEL_PATH, BACKUP_MODEL)
                if os.path.exists(SCALER_PATH):
                    os.replace(SCALER_PATH, BACKUP_SCALER)
            except Exception:
                pass
            # move tmp to final
            os.replace(MODEL_TMP, MODEL_PATH)
            os.replace(SCALER_TMP, SCALER_PATH)
            _write_log('Retrain finished successfully. Model and scaler saved (swapped).')
            return 'Model retrained and saved.'
        else:
            # remove tmp files
            try:
                if MODEL_TMP.exists():
                    MODEL_TMP.unlink()
                if SCALER_TMP.exists():
                    SCALER_TMP.unlink()
            except Exception:
                pass
            _write_log('Retrain finished but new model was not accepted; existing model retained.')
            return 'Retrain completed; no swap performed.'
    except Exception as e:
        _write_log(f'Retrain failed: {repr(e)}')
        raise
    finally:
        try:
            if LOCK_PATH.exists():
                LOCK_PATH.unlink()
        except Exception:
            pass
