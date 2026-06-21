import sys
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_load")

def custom_asymmetric_objective(y_true, y_pred):
    import numpy as np
    p = 1.0 / (1.0 + np.exp(-y_pred))
    w = 10.0
    grad = p * (1.0 + y_true * (w - 1.0)) - w * y_true
    hess = (1.0 + y_true * (w - 1.0)) * p * (1.0 - p)
    return grad, hess

# Register it in __main__ and also astra_ml.models.m1_closure_classifier in case joblib looks for it there
sys.modules["__main__"].custom_asymmetric_objective = custom_asymmetric_objective

# Mock the module import to prevent actually importing the full classifier file if it drags slow packages
import types
mock_module = types.ModuleType("astra_ml.models.m1_closure_classifier")
mock_module.custom_asymmetric_objective = custom_asymmetric_objective
sys.modules["astra_ml.models.m1_closure_classifier"] = mock_module

logger.info("Starting load...")
try:
    model = joblib.load("models/lgbm_champion.pkl")
    logger.info("Model loaded successfully!")
    print(model)
except Exception as e:
    logger.error(f"Error loading model: {e}")
