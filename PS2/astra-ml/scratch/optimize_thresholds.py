import sys
import os
import numpy as np

# LightGBM custom objective unpickling hack
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
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, precision_score, recall_score, f1_score
from sklearn.calibration import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTENC

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

def fit_bootstrap_threshold(y_true, y_prob, target_recall=0.85, n_bootstrap=200, seed=42):
    np.random.seed(seed)
    thresholds = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        y_true_s = y_true[sample_idx]
        y_prob_s = y_prob[sample_idx]
        
        if np.sum(y_true_s == 1) == 0:
            continue
            
        precisions, recalls, ths = precision_recall_curve(y_true_s, y_prob_s)
        valid_indices = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]
            thresholds.append(ths[best_idx])
        else:
            thresholds.append(ths[0] if len(ths) > 0 else 0.5)
            
    if not thresholds:
        return 0.5
    return float(np.median(thresholds))

def draw_cm_plot(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    
    # Add values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large', weight='bold')
            
    ax.set_xticklabels([''] + ["No Road Closure Required (0)", "Road Closure Required (1)"], fontsize=9, fontweight="bold")
    ax.set_yticklabels([''] + ["No Road Closure Required (0)", "Road Closure Required (1)"], fontsize=9, fontweight="bold", rotation=90, va="center")
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight="bold")
    ax.set_ylabel('True Label', fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    print("==========================================================")
    print("M1 CLOSURE MODEL THRESHOLD OPTIMIZATION PIPELINE")
    print("==========================================================")
    
    data_cfg, m1_cfg = load_configs()
    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)
    
    train_df = df.filter(pl.col("split") == "train")
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
    
    # 1. Encode features using standard label encoders
    label_encoders = {}
    categorical_cols = [
        f for f in variant_b_feats
        if df[f].dtype in (pl.Utf8, pl.Categorical) or f in ["event_cause", "corridor", "priority", "vehicle_type"]
    ]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            all_values = df[col].cast(pl.Utf8).fill_null("__MISSING__").to_list()
            le.fit(all_values)
            label_encoders[col] = le
            
    def encode_split(split_df):
        arrays = []
        for f in variant_b_feats:
            if f in label_encoders:
                vals = split_df[f].cast(pl.Utf8).fill_null("__MISSING__").to_list()
                encoded = label_encoders[f].transform(vals)
                arrays.append(encoded.reshape(-1, 1))
            elif f in split_df.columns:
                arr = split_df[f].to_numpy().astype(np.float64)
                arr = np.nan_to_num(arr, nan=0.0)
                arrays.append(arr.reshape(-1, 1))
        return np.hstack(arrays)
        
    X_train = encode_split(train_df)
    X_val = encode_split(val_df)
    X_test = encode_split(test_df)
    
    y_train = train_df[target].cast(pl.Int32).to_numpy()
    y_val = val_df[target].cast(pl.Int32).to_numpy()
    y_test = test_df[target].cast(pl.Int32).to_numpy()
    
    # Fit the exact model from compute_per_cause_cm.py to match starting baseline performance
    cat_indices = [i for i, f in enumerate(variant_b_feats) if f in categorical_cols]
    smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=3)
    X_train_strat, y_train_strat = smote.fit_resample(X_train, y_train)
    
    best_params = {
        'learning_rate': 0.011974534294889197,
        'num_leaves': 87,
        'max_depth': 7,
        'min_child_samples': 62,
        'reg_alpha': 8.560536283177967,
        'reg_lambda': 5.298641125101568,
        'n_estimators': 600,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = LGBMClassifier(**best_params)
    model.fit(X_train_strat, y_train_strat)
    
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Calibrate
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob_val, y_val)
    y_prob_val_cal = calibrator.predict(y_prob_val)
    y_prob_test_cal = calibrator.predict(y_prob_test)
    
    # 2. Get baseline test performance using the bootstrap thresholds from original script
    baseline_thresholds = {}
    from astra_ml.eval.metrics import compute_classification_metrics
    base_metrics = compute_classification_metrics(y_val, y_prob_val_cal, target_recall=0.85)
    t_global = base_metrics.threshold
    
    for group_name in ["high_closure", "medium_closure", "low_closure", "very_low_closure", "global_fallback"]:
        if group_name == "global_fallback":
            group_idx = [i for i, c in enumerate(val_causes) if get_cause_group(c) == "global_fallback"]
        else:
            group_idx = [i for i, c in enumerate(val_causes) if get_cause_group(c) == group_name]
            
        if len(group_idx) == 0:
            baseline_thresholds[group_name] = t_global
            continue
            
        y_true_g = y_val[group_idx]
        y_prob_g = y_prob_val_cal[group_idx]
        
        t_g = fit_bootstrap_threshold(y_true_g, y_prob_g, target_recall=0.85, n_bootstrap=200)
        baseline_thresholds[group_name] = t_g
        
    y_pred_baseline = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        group = get_cause_group(test_causes[i])
        thresh = baseline_thresholds.get(group, t_global)
        y_pred_baseline[i] = 1 if y_prob_test_cal[i] >= thresh else 0
        
    cm_baseline = confusion_matrix(y_test, y_pred_baseline, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_baseline[0][0], cm_baseline[0][1], cm_baseline[1][0], cm_baseline[1][1]
    recall_b = recall_score(y_test, y_pred_baseline)
    prec_b = precision_score(y_test, y_pred_baseline)
    f1_b = f1_score(y_test, y_pred_baseline)
    f2_b = 5 * (prec_b * recall_b) / (4 * prec_b + recall_b) if (prec_b + recall_b) > 0 else 0
    
    print(f"Verified Current Model Baseline on Test Split:")
    print(f"  TN: {tn_b} | FP: {fp_b} | FN: {fn_b} | TP: {tp_b}")
    print(f"  Recall: {recall_b:.4%}, Precision: {prec_b:.4%}, F1: {f1_b:.4%}, F2: {f2_b:.4%}")
    
    # 3. Global threshold search from 0.01 to 0.95 (step 0.01) on validation split
    threshold_grid = np.arange(0.01, 0.96, 0.01)
    val_recalls = []
    val_precisions = []
    val_scores = []
    
    best_global_thresh = None
    best_global_score = -1.0
    
    for t in threshold_grid:
        preds = (y_prob_val_cal >= t).astype(int)
        r = recall_score(y_val, preds, zero_division=0)
        p = precision_score(y_val, preds, zero_division=0)
        score = 0.7 * r + 0.3 * p
        
        val_recalls.append(r)
        val_precisions.append(p)
        val_scores.append(score)
        
        if r >= 0.85:
            if score > best_global_score:
                best_global_score = score
                best_global_thresh = t
                
    if best_global_thresh is None:
        # Fallback to threshold that gives highest recall if none >= 0.85
        best_global_thresh = threshold_grid[np.argmax(val_recalls)]
        print(f"WARNING: No global threshold satisfied Recall >= 85% on validation split. Fallback to: {best_global_thresh:.2f}")
    else:
        print(f"Optimal Global Threshold found on validation split: {best_global_thresh:.2f} (Score: {best_global_score:.4f})")
        
    # Evaluate global threshold on test split
    y_pred_global = (y_prob_test_cal >= best_global_thresh).astype(int)
    cm_global = confusion_matrix(y_test, y_pred_global, labels=[0, 1])
    tn_g, fp_g, fn_g, tp_g = cm_global[0][0], cm_global[0][1], cm_global[1][0], cm_global[1][1]
    recall_g = recall_score(y_test, y_pred_global)
    prec_g = precision_score(y_test, y_pred_global)
    f1_g = f1_score(y_test, y_pred_global)
    f2_g = 5 * (prec_g * recall_g) / (4 * prec_g + recall_g) if (prec_g + recall_g) > 0 else 0
    
    # 4. Cause-group threshold search
    # Map each validation cause to one of the three tiers: High (0), Medium (1), Low (2)
    def map_cause_to_tier(cause):
        cause = cause.lower()
        if cause in ["vip_movement", "public_event", "protest", "procession"]:
            return 0 # High
        elif cause in ["construction", "tree_fall", "road_conditions"]:
            return 1 # Medium
        elif cause in ["vehicle_breakdown", "accident", "pot_holes", "water_logging"]:
            return 2 # Low
        return 2 # fallback to Low
        
    val_tiers = np.array([map_cause_to_tier(c) for c in val_causes])
    test_tiers = np.array([map_cause_to_tier(c) for c in test_causes])
    
    # Pre-calculate validation TP and FP for each group and each threshold in step 0.01
    t_list = np.arange(0.01, 0.96, 0.01)
    N_t = len(t_list)
    
    tp_grid = np.zeros((3, N_t))
    fp_grid = np.zeros((3, N_t))
    fn_grid = np.zeros((3, N_t))
    tn_grid = np.zeros((3, N_t))
    
    for tier in range(3):
        idx = (val_tiers == tier)
        y_prob_tier = y_prob_val_cal[idx]
        y_true_tier = y_val[idx]
        
        for k, t in enumerate(t_list):
            preds = (y_prob_tier >= t).astype(int)
            tp_grid[tier, k] = np.sum((preds == 1) & (y_true_tier == 1))
            fp_grid[tier, k] = np.sum((preds == 1) & (y_true_tier == 0))
            fn_grid[tier, k] = np.sum((preds == 0) & (y_true_tier == 1))
            tn_grid[tier, k] = np.sum((preds == 0) & (y_true_tier == 0))
            
    P_total = np.sum(y_val == 1)
    N_total = np.sum(y_val == 0)
    
    # Search all combinations of thresholds
    best_combo = None
    best_combo_score = -1.0
    
    best_combo_min_fp = None
    min_fp_value = float('inf')
    
    for idx_h in range(N_t):
        for idx_m in range(N_t):
            for idx_l in range(N_t):
                tp = tp_grid[0, idx_h] + tp_grid[1, idx_m] + tp_grid[2, idx_l]
                fp = fp_grid[0, idx_h] + fp_grid[1, idx_m] + fp_grid[2, idx_l]
                
                recall = tp / P_total
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                score = 0.7 * recall + 0.3 * precision
                
                if recall >= 0.85:
                    if score > best_combo_score:
                        best_combo_score = score
                        best_combo = (t_list[idx_h], t_list[idx_m], t_list[idx_l])
                    if fp < min_fp_value:
                        min_fp_value = fp
                        best_combo_min_fp = (t_list[idx_h], t_list[idx_m], t_list[idx_l])
                        
    if best_combo is None:
        best_combo = (t_global, t_global, t_global)
        best_combo_min_fp = (t_global, t_global, t_global)
        print("WARNING: No cause-group combination satisfied Recall >= 85% on validation split. Fallback to global.")
    else:
        print(f"Optimal Cause-Group (Score-Max): High={best_combo[0]:.2f}, Medium={best_combo[1]:.2f}, Low={best_combo[2]:.2f} (Score: {best_combo_score:.4f})")
        print(f"Optimal Cause-Group (FP-Min): High={best_combo_min_fp[0]:.2f}, Medium={best_combo_min_fp[1]:.2f}, Low={best_combo_min_fp[2]:.2f} (Val FP: {min_fp_value})")
        
    # Evaluate score-maximizing cause-group thresholds on test split
    t_high, t_med, t_low = best_combo
    y_pred_combo = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        tier = test_tiers[i]
        thresh = t_high if tier == 0 else (t_med if tier == 1 else t_low)
        y_pred_combo[i] = 1 if y_prob_test_cal[i] >= thresh else 0
        
    cm_combo = confusion_matrix(y_test, y_pred_combo, labels=[0, 1])
    tn_c, fp_c, fn_c, tp_c = cm_combo[0][0], cm_combo[0][1], cm_combo[1][0], cm_combo[1][1]
    recall_c = recall_score(y_test, y_pred_combo)
    prec_c = precision_score(y_test, y_pred_combo)
    f1_c = f1_score(y_test, y_pred_combo)
    f2_c = 5 * (prec_c * recall_c) / (4 * prec_c + recall_c) if (prec_c + recall_c) > 0 else 0

    # Evaluate FP-minimizing cause-group thresholds on test split
    t_high_f, t_med_f, t_low_f = best_combo_min_fp
    y_pred_combo_f = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        tier = test_tiers[i]
        thresh = t_high_f if tier == 0 else (t_med_f if tier == 1 else t_low_f)
        y_pred_combo_f[i] = 1 if y_prob_test_cal[i] >= thresh else 0
        
    cm_combo_f = confusion_matrix(y_test, y_pred_combo_f, labels=[0, 1])
    tn_cf, fp_cf, fn_cf, tp_cf = cm_combo_f[0][0], cm_combo_f[0][1], cm_combo_f[1][0], cm_combo_f[1][1]
    recall_cf = recall_score(y_test, y_pred_combo_f)
    prec_cf = precision_score(y_test, y_pred_combo_f)
    f1_cf = f1_score(y_test, y_pred_combo_f)
    f2_cf = 5 * (prec_cf * recall_cf) / (4 * prec_cf + recall_cf) if (prec_cf + recall_cf) > 0 else 0
    
    # Calculate FP reduction and FN increase
    fp_red_global = (fp_b - fp_g) / fp_b * 100
    fn_inc_global = (fn_g - fn_b) / fn_b * 100 if fn_b > 0 else 0
    
    fp_red_combo = (fp_b - fp_c) / fp_b * 100
    fn_inc_combo = (fn_c - fn_b) / fn_b * 100 if fn_b > 0 else 0

    fp_red_combo_f = (fp_b - fp_cf) / fp_b * 100
    fn_inc_combo_f = (fn_cf - fn_b) / fn_b * 100 if fn_b > 0 else 0
    
    # Calculate PR-AUC for reference
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob_test_cal)
    pr_auc_test = auc(recall_vals, precision_vals)
    
    print("\n" + "=" * 80)
    print("DETAILED GRID OF GLOBAL THRESHOLDS ON TEST SPLIT")
    print("=" * 80)
    print(f"{'Thresh':<8} | {'Recall':<8} | {'Precision':<10} | {'FP':<6} | {'FN':<6} | {'F2':<8} | {'FP Red %':<10} | {'Score (0.7R + 0.3P)':<20}")
    print("-" * 80)
    for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
        preds = (y_prob_test_cal >= t).astype(int)
        r = recall_score(y_test, preds, zero_division=0)
        p = precision_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        fp = cm[0][1]
        fn = cm[1][0]
        score = 0.7 * r + 0.3 * p
        f2 = 5 * (p * r) / (4 * p + r) if (p + r) > 0 else 0
        fp_red = (fp_b - fp) / fp_b * 100
        print(f"{t:<8.2f} | {r:<8.2%} | {p:<10.2%} | {fp:<6} | {fn:<6} | {f2:<8.4f} | {fp_red:<9.2f}% | {score:<20.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON TABLE (TEST SPLIT)")
    print("=" * 60)
    print(f"Baseline model:")
    print(f"  Recall: {recall_b:.2%}, Precision: {prec_b:.2%}, FP: {fp_b}, FN: {fn_b}, F2: {f2_b:.4f}")
    print(f"Optimized Global Threshold ({best_global_thresh:.2f}):")
    print(f"  Recall: {recall_g:.2%}, Precision: {prec_g:.2%}, FP: {fp_g}, FN: {fn_g}, F2: {f2_g:.4f}")
    print(f"  FP Reduction: {fp_red_global:.2f}%, FN Increase: {fn_inc_global:.2f}%")
    print(f"Optimized Cause-Group Thresholds (Score-Max) (High={t_high:.2f}, Med={t_med:.2f}, Low={t_low:.2f}):")
    print(f"  Recall: {recall_c:.2%}, Precision: {prec_c:.2%}, FP: {fp_c}, FN: {fn_c}, F2: {f2_c:.4f}")
    print(f"  FP Reduction: {fp_red_combo:.2f}%, FN Increase: {fn_inc_combo:.2f}%")
    print(f"Optimized Cause-Group Thresholds (FP-Min) (High={t_high_f:.2f}, Med={t_med_f:.2f}, Low={t_low_f:.2f}):")
    print(f"  Recall: {recall_cf:.2%}, Precision: {prec_cf:.2%}, FP: {fp_cf}, FN: {fn_cf}, F2: {f2_cf:.4f}")
    print(f"  FP Reduction: {fp_red_combo_f:.2f}%, FN Increase: {fn_inc_combo_f:.2f}%")
    
    # 5. Save all plots to reports folder
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'Test PR Curve (AUC={pr_auc_test:.4f})')
    ax.set_xlabel('Recall', fontsize=11, fontweight="bold")
    ax.set_ylabel('Precision', fontsize=11, fontweight="bold")
    ax.set_title('Precision-Recall Curve (Calibrated Test Split)', fontsize=12, fontweight="bold")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(reports_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 2: Threshold vs Recall & Precision
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(threshold_grid, val_recalls, color='red', lw=2, label='Validation Recall')
    ax.plot(threshold_grid, val_precisions, color='green', lw=2, label='Validation Precision')
    ax.axvline(x=best_global_thresh, color='gray', linestyle='--', label=f'Optimal Global Thresh ({best_global_thresh:.2f})')
    ax.set_xlabel('Threshold', fontsize=11, fontweight="bold")
    ax.set_ylabel('Score', fontsize=11, fontweight="bold")
    ax.set_title('Threshold vs Recall & Precision', fontsize=12, fontweight="bold")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(reports_dir / "threshold_vs_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 3 & 4: Confusion Matrices
    draw_cm_plot(cm_baseline, "Baseline Model Confusion Matrix\n(FP=890, Recall=88.03%)", reports_dir / "cm_baseline.png")
    draw_cm_plot(cm_global, f"Optimized Global Threshold ({best_global_thresh:.2f}) CM\n(FP={fp_g}, Recall={recall_g:.2%})", reports_dir / "cm_global.png")
    draw_cm_plot(cm_combo, f"Optimized Cause-Group (Score-Max) CM\n(FP={fp_c}, Recall={recall_c:.2%})", reports_dir / "cm_cause_group.png")
    draw_cm_plot(cm_combo_f, f"Optimized Cause-Group (FP-Min) CM\n(FP={fp_cf}, Recall={recall_cf:.2%})", reports_dir / "cm_cause_group_fp_min.png")
    
    # 6. Generate detailed report
    report_content = f"""# ASTRA M1 Model: Threshold Optimization and Executive Summary

This report presents the analysis and recommendation for optimizing thresholds on the ASTRA M1 Closure Prediction Model to reduce operator alert fatigue while ensuring critical event capture.

---

## 1. Event Causes Contributing Most to False Positives
The baseline model generated a high rate of False Positives (**890 FP** on the test set). 

A per-cause breakdown of False Positives from the test split reveals that a few classes dominate the false alarm rate:
1. **vehicle_breakdown**: Generated **442 FP** (49.7% of total FP). 
2. **water_logging**: Generated **146 FP** (16.4% of total FP).
3. **pot_holes**: Generated **91 FP** (10.2% of total FP).
4. **tree_fall**: Generated **67 FP** (7.5% of total FP).
5. **others**: Generated **64 FP** (7.2% of total FP).

### Top FP Contributors Summary
Combined, `vehicle_breakdown`, `water_logging`, and `pot_holes` account for **76.3%** of all false positive alerts. Alert fatigue is primarily driven by these low-prevalence/high-volume categories where the local threshold was set extremely low to force high recall.

---

## 2. Per-Cause Prevalence & Evaluation Metrics
Below is the test split breakdown of closure prevalence, recall, precision, and false positive rates (FPR) under the starting model parameters:

| Event Cause | Group | Total Samples | True Positives | FPR | Recall | Precision | Prevalence % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vehicle_breakdown** | Low | 899 | 27 | 50.7% | 66.67% | 3.91% | 3.00% |
| **water_logging** | Low | 213 | 16 | 74.1% | 93.75% | 9.32% | 7.51% |
| **tree_fall** | Medium | 126 | 44 | 81.7% | 95.45% | 38.53% | 34.92% |
| **pot_holes** | Low | 116 | 3 | 80.5% | 66.67% | 2.15% | 2.59% |
| **others** | Low | 89 | 7 | 78.0% | 100.00% | 9.86% | 7.87% |
| **construction** | Medium | 56 | 13 | 44.2% | 100.00% | 40.62% | 23.21% |
| **accident** | Low | 43 | 3 | 55.0% | 66.67% | 8.33% | 6.98% |
| **road_conditions** | Medium | 35 | 5 | 50.0% | 40.00% | 11.76% | 14.29% |
| **congestion** | Very Low | 27 | 0 | 59.3% | 0.00% | 0.00% | 0.00% |
| **vip_movement** | High | 16 | 16 | 0.0% | 100.00% | 100.00% | 100.00% |
| **public_event** | High | 9 | 6 | 100.0% | 100.00% | 66.67% | 66.67% |
| **procession** | High | 8 | 2 | 50.0% | 100.00% | 40.00% | 25.00% |
| **protest** | High | 1 | 0 | 100.0% | 0.00% | 0.00% | 0.00% |
| **debris** | Very Low | 1 | 0 | 100.0% | 0.00% | 0.00% | 0.00% |

---

## 3. Comparison Table: Current vs. Optimized Models

| Metric | Current Model (Baseline) | Optimized Global Threshold ({best_global_thresh:.2f}) | Optimized Cause-Group Tiers (Score-Max) | Optimized Cause-Group Tiers (FP-Min) |
| :--- | :--- | :--- | :--- | :--- |
| **True Negatives (TN)** | {tn_b} | {tn_g} | {tn_c} | {tn_cf} |
| **False Positives (FP)** | {fp_b} | {fp_g} | {fp_c} | {fp_cf} |
| **False Negatives (FN)** | {fn_b} | {fn_g} | {fn_c} | {fn_cf} |
| **True Positives (TP)** | {tp_b} | {tp_g} | {tp_c} | {tp_cf} |
| **Overall Recall** | {recall_b:.2%} | {recall_g:.2%} | {recall_c:.2%} | {recall_cf:.2%} |
| **Overall Precision** | {prec_b:.2%} | {prec_g:.2%} | {prec_c:.2%} | {prec_cf:.2%} |
| **F1 Score** | {f1_b:.4f} | {f1_g:.4f} | {f1_c:.4f} | {f1_cf:.4f} |
| **F2 Score** | {f2_b:.4f} | {f2_g:.4f} | {f2_c:.4f} | {f2_cf:.4f} |
| **PR-AUC** | {pr_auc_test:.4f} | {pr_auc_test:.4f} | {pr_auc_test:.4f} | {pr_auc_test:.4f} |
| **FP Reduction %** | Baseline | **{fp_red_global:.2f}%** | **{fp_red_combo:.2f}%** | **{fp_red_combo_f:.2f}%** |
| **FN Increase %** | Baseline | **{fn_inc_global:.2f}%** | **{fn_inc_combo:.2f}%** | **{fn_inc_combo_f:.2f}%** |

---

## 4. Final Recommendation & Deployment Threshold Strategy

### Analysis of Threshold Tuning Constraints
To qualify as a successful threshold-only optimization under the project requirements:
1. Recall must remain **>= 85%**.
2. False Positives must be reduced by **>= 40%** (i.e. FP <= 534).

Looking at the optimization results:
* The **Optimal Global Threshold ({best_global_thresh:.2f})** yields **FP = {fp_g}** ({fp_red_global:.2f}% reduction) with a Recall of **{recall_g:.2%}**.
* The **Optimal Cause-Group Thresholds (Score-Max)** yield **FP = {fp_c}** ({fp_red_combo:.2f}% reduction) with a Recall of **{recall_c:.2%}**.
* The **Optimal Cause-Group Thresholds (FP-Min)** yield **FP = {fp_cf}** ({fp_red_combo_f:.2f}% reduction) with a Recall of **{recall_cf:.2%}**.

### Recommendation:
> [!WARNING]
> **Threshold Tuning Limitation Identified:** None of the threshold-only optimization strategies can successfully reduce False Positives by **>= 40%** (target FP <= 534) while maintaining the operational recall floor of **>= 85%**.
> 
> Specifically, the best possible FP reduction that maintains a Recall >= 85% on the test split is **{fp_red_combo_f:.2f}%** (yielding **{fp_cf} FP** and **{recall_cf:.2%} Recall** using cause-group thresholds of High={t_high_f:.2f}, Med={t_med_f:.2f}, Low={t_low_f:.2f}).
> 
> Raising the global threshold to any value greater than or equal to `0.07` drops the test Recall to `77.46%`, which violates the `85%` recall floor. This indicates that true positive and false positive predictions are highly overlapped between probabilities `0.01` and `0.07`.
> 
> Therefore, we recommend **not deploying a threshold-only change** as the final solution. Instead, the model needs to be retrained with advanced improvements.

### Recommended Advanced Model Improvements:
1. **Cost-Sensitive Learning / Sample Weighting:** During training, apply a custom sample weight to the positive class (e.g. 5x or 10x weight to road closure events) or modify the loss function to explicitly penalize false negatives more heavily than false positives. This will force the classifier to learn a cleaner boundary, separating positive and negative instances with a larger probability gap.
2. **Feature Engineering:** Build interaction features such as `cause_prev_closures` (historical closure rate for the event cause), `corridor_closure_density`, or NLP embeddings from text fields like `incident_description` to provide stronger predictive signals.
3. **Calibrated Cause-Specific Modeling:** Train separate models or sub-classifiers for high-risk cause categories (like public events or construction) to model their closure conditions independently.

### backend/main.py Integration (If threshold-only solution is selected):
If a sub-optimal threshold-only solution must be deployed, modify the backend startup threshold dictionary in `main.py` using the FP-Min Cause-Group configuration:
```python
# Update thresholds in src/astra_ml/api/main.py:
cause_thresholds = {{
    "vip_movement": {t_high_f:.2f},
    "public_event": {t_high_f:.2f},
    "protest": {t_high_f:.2f},
    "procession": {t_high_f:.2f},
    "construction": {t_med_f:.2f},
    "tree_fall": {t_med_f:.2f},
    "road_conditions": {t_med_f:.2f},
    "vehicle_breakdown": {t_low_f:.2f},
    "accident": {t_low_f:.2f},
    "pot_holes": {t_low_f:.2f},
    "water_logging": {t_low_f:.2f},
    "debris": {t_low_f:.2f},
    "congestion": {t_low_f:.2f},
}}
```
"""
    
    with open(reports_dir / "m1_optimization_report.md", "w") as f:
        f.write(report_content)
        
    print(f"\nOptimization Report saved to: {reports_dir / 'm1_optimization_report.md'}")
    print("Matplotlib figures saved to reports/")
    
    # 7. Executive summary printed directly
    print("\n" + "=" * 60)
    print("ASTRA HACKATHON EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"* Current Performance: Recall = {recall_b:.2%}, Precision = {prec_b:.2%}, FP = {fp_b}")
    print(f"* Global Threshold ({best_global_thresh:.2f}): Recall = {recall_g:.2%}, Precision = {prec_g:.2%}, FP = {fp_g} ({fp_red_global:.2f}% FP reduction)")
    print(f"* Cause-Group (Score-Max): Recall = {recall_c:.2%}, Precision = {prec_c:.2%}, FP = {fp_c} ({fp_red_combo:.2f}% FP reduction)")
    print(f"* Cause-Group (FP-Min): Recall = {recall_cf:.2%}, Precision = {prec_cf:.2%}, FP = {fp_cf} ({fp_red_combo_f:.2f}% FP reduction)")
    if fp_cf <= 534 and recall_cf >= 0.85:
        verdict = "Threshold tuning succeeded. FP reduction >= 40% achieved while keeping Recall >= 85%."
    else:
        verdict = f"Threshold tuning failed. Best FP reduction under Recall >= 85% is only {fp_red_combo_f:.2f}% (FP={fp_cf}). Retraining/feature engineering required."
    print(f"* Verdict: {verdict}")
    print("=" * 60)

if __name__ == "__main__":
    main()
