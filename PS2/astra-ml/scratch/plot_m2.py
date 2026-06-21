import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from astra_ml.models.m2_duration_acute import _prepare_duration_data, load_configs
import os

def main():
    data_cfg, m2_cfg = load_configs()
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoders, features = _prepare_duration_data(
        data_cfg, m2_cfg, regime_filter="acute"
    )
    
    # Train quick CatBoost
    model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, loss_function='MAE', verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    
    y_pred = model.predict(X_test)
    
    # Log scale is used by default in m2_duration_acute
    actual_durations = np.exp(y_test)
    predicted_durations = np.exp(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.style.use('dark_background')
    
    plt.scatter(actual_durations, predicted_durations, alpha=0.5, color='#3b82f6', s=40, edgecolors='none')
    
    # Diagonal line
    max_val = max(actual_durations.max(), predicted_durations.max())
    min_val = min(actual_durations.min(), predicted_durations.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Duration (hours)')
    plt.ylabel('Predicted Duration (hours)')
    plt.title('M2 Duration Estimator: Predicted vs. Actual (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.2, ls='--')
    
    # Needs absolute path for assets folder
    out_dir = Path("/Users/prathameshnawale/Desktop/Flipkart Grid 2.0/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m2_duration_scatter.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#1e1e1e', bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
