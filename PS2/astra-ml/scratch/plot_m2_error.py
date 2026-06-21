import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from astra_ml.models.m2_duration_acute import _prepare_duration_data, load_configs

def main():
    data_cfg, m2_cfg = load_configs()
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoders, features = _prepare_duration_data(
        data_cfg, m2_cfg, regime_filter="acute"
    )
    
    # Train quick CatBoost
    model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, loss_function='MAE', verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    
    y_pred = model.predict(X_test)
    
    # Calculate actual durations (inverse log)
    actual_durations = np.exp(y_test)
    predicted_durations = np.exp(y_pred)
    
    # Calculate Error (Residuals)
    errors = predicted_durations - actual_durations
    
    # Calculate RMSE
    rmse = root_mean_squared_error(actual_durations, predicted_durations)
    
    plt.figure(figsize=(8, 6))
    plt.style.use('dark_background')
    
    # Plot histogram of errors
    plt.hist(errors, bins=50, color='#ef4444', alpha=0.7, edgecolor='black')
    
    # Add vertical line at 0 (perfect prediction)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1.5, label='Zero Error')
    
    plt.xlabel('Prediction Error (hours)')
    plt.ylabel('Frequency')
    plt.title(f'M2 Error Distribution\nRMSE: {rmse:.2f} hours')
    plt.legend()
    plt.grid(True, alpha=0.2, ls='--')
    
    out_dir = Path("/Users/prathameshnawale/Desktop/Flipkart Grid 2.0/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m2_error_dist.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#1e1e1e', bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
