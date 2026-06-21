import sys
import numpy as np

def custom_asymmetric_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = 1.0 / (1.0 + np.exp(-y_pred))
    w = 10.0
    grad = p * (1.0 + y_true * (w - 1.0)) - w * y_true
    hess = (1.0 + y_true * (w - 1.0)) * p * (1.0 - p)
    return grad, hess

sys.modules["__main__"].custom_asymmetric_objective = custom_asymmetric_objective

import types
mock_m1_module = types.ModuleType("astra_ml.models.m1_closure_classifier")
mock_m1_module.custom_asymmetric_objective = custom_asymmetric_objective
sys.modules["astra_ml.models.m1_closure_classifier"] = mock_m1_module

import joblib
import polars as pl
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import IsotonicRegression
from lightgbm import LGBMClassifier

def load_configs():
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.to_container(OmegaConf.load("configs/data.yaml"), resolve=True)
    m1_cfg = OmegaConf.to_container(OmegaConf.load("configs/m1_closure_classifier.yaml"), resolve=True)
    return data_cfg, m1_cfg

def get_cause_group(cause):
    cause = cause.lower() if cause else ""
    if cause in ["vip_movement", "public_event", "protest", "procession"]:
        return "high_closure"
    elif cause in ["tree_fall", "construction", "road_conditions"]:
        return "medium_closure"
    elif cause in ["vehicle_breakdown", "accident", "water_logging", "others"]:
        return "low_closure"
    elif cause in ["debris", "congestion"]:
        return "very_low_closure"
    else:
        return "global_fallback"

def main():
    data_cfg, m1_cfg = load_configs()
    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)
    
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == "test")
    
    val_causes = val_df["event_cause"].to_list()
    test_causes = test_df["event_cause"].to_list()
    
    all_cols = df.columns
    text_emb_cols = [c for c in all_cols if c.startswith("text_emb_")]
    
    base_challenger_feats = m1_cfg["challenger_features"]
    base_challenger_feats = [f for f in base_challenger_feats if f not in ["cause_corridor", "cause_hour"] and not f.startswith("text_emb_")]
    variant_b_feats = base_challenger_feats + text_emb_cols
    
    target = m1_cfg["target"]
    
    # Load model and calibration assets
    models_dir = Path("models")
    champion_model = joblib.load(models_dir / "lgbm_champion.pkl")
    calibrator = joblib.load(models_dir / "isotonic_calibrator.pkl")
    thresholds = joblib.load(models_dir / "cause_thresholds.pkl")
    m1_label_encoders = joblib.load(models_dir / "m1_label_encoders.pkl")
    
    def encode_split(split_df):
        arrays = []
        for f in variant_b_feats:
            if f in m1_label_encoders:
                vals = split_df[f].cast(pl.Utf8).fill_null("__MISSING__").to_list()
                encoded = m1_label_encoders[f].transform(vals)
                arrays.append(encoded.reshape(-1, 1))
            elif f in split_df.columns:
                arr = split_df[f].to_numpy().astype(np.float64)
                arr = np.nan_to_num(arr, nan=0.0)
                arrays.append(arr.reshape(-1, 1))
        return np.hstack(arrays)
        
    X_val = encode_split(val_df)
    X_test = encode_split(test_df)
    
    y_val = val_df[target].cast(pl.Int32).to_numpy()
    y_test = test_df[target].cast(pl.Int32).to_numpy()
    
    # Predict raw scores
    raw_val = champion_model.predict(X_val, raw_score=True)
    y_prob_val = 1.0 / (1.0 + np.exp(-raw_val))
    raw_test = champion_model.predict(X_test, raw_score=True)
    y_prob_test = 1.0 / (1.0 + np.exp(-raw_test))
    
    # Calibrate
    y_prob_val_cal = calibrator.predict(y_prob_val)
    y_prob_test_cal = calibrator.predict(y_prob_test)
    
    # Evaluate with saved thresholds
    y_pred_test = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        cause = test_causes[i]
        group = get_cause_group(cause)
        thresh = thresholds.get(group, 0.5)
        y_pred_test[i] = 1 if y_prob_test_cal[i] >= thresh else 0
        
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    print("SAVED MODEL CM ON TEST SPLIT:")
    print(cm)

if __name__ == "__main__":
    main()
