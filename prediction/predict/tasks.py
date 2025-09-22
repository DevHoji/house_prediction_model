from celery import shared_task
import numpy as np
import pandas as pd
from .ml_model import MODEL_PATH, SCALER_PATH
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

@shared_task
def retrain_model():
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
        # Load previous data (simulate by retraining on new + old)
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
    X = data[['size', 'bedrooms', 'age']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    return 'Model retrained and saved.'
