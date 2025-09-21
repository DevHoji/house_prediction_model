import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self._train_and_save()

    def _train_and_save(self):
        # Generate synthetic data (replace with real data for production)
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
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.model, MODEL_PATH)

    def predict(self, size, bedrooms, age):
        if not self.model or not self.scaler:
            self._load()
        input_data = np.array([[size, bedrooms, age]])
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)[0]
        return round(prediction, 2)
