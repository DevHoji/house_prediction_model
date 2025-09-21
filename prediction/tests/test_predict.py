import os
import tempfile
from predict.ml_model import ModelManager

def test_model_prediction_runs():
    mm = ModelManager()
    pred = mm.predict(2500, 3, 10)
    assert isinstance(pred, float) or isinstance(pred, int)
