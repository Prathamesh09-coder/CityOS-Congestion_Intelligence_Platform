import mlflow
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.calibration import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTENC
from astra_ml.models.m1_closure_classifier import load_configs

# Define cause groups
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
    from sklearn.metrics import precision_recall_curve
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

def main():
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
    imputed_flags = ["zone_imputed", "junction_imputed", "zone_missing", "junction_missing"]
    
    base_challenger_feats = m1_cfg["challenger_features"]
    base_challenger_feats = [f for f in base_challenger_feats if f not in ["cause_corridor", "cause_hour"] and not f.startswith("text_emb_")]
    
    # Variant B: base + embeddings (imputed flags are already in base)
    variant_b_feats = base_challenger_feats + text_emb_cols
    
    target = m1_cfg["target"]
    
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
    
    # Fit SMOTE-NC on train
    cat_indices = [i for i, f in enumerate(variant_b_feats) if f in categorical_cols]
    smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=3)
    X_train_strat, y_train_strat = smote.fit_resample(X_train, y_train)
    
    # Best params from Variant B search
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
    
    # Thresholds
    thresholds = {}
    from astra_ml.eval.metrics import compute_classification_metrics
    base_metrics = compute_classification_metrics(y_val, y_prob_val_cal, target_recall=0.85)
    t_global = base_metrics.threshold
    
    for group_name in ["high_closure", "medium_closure", "low_closure", "very_low_closure", "global_fallback"]:
        if group_name == "global_fallback":
            group_idx = [i for i, c in enumerate(val_causes) if get_cause_group(c) == "global_fallback"]
        else:
            group_idx = [i for i, c in enumerate(val_causes) if get_cause_group(c) == group_name]
            
        if len(group_idx) == 0:
            thresholds[group_name] = t_global
            continue
            
        y_true_g = y_val[group_idx]
        y_prob_g = y_prob_val_cal[group_idx]
        
        t_g = fit_bootstrap_threshold(y_true_g, y_prob_g, target_recall=0.85, n_bootstrap=200)
        thresholds[group_name] = t_g
        
    # Evaluate and compute per-cause confusion matrix on test
    y_pred_test = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        group = get_cause_group(test_causes[i])
        thresh = thresholds.get(group, t_global)
        y_pred_test[i] = 1 if y_prob_test_cal[i] >= thresh else 0
        
    print("=" * 60)
    print("PER-CAUSE CONFUSION MATRICES ON TEST SPLIT")
    print("=" * 60)
    
    all_causes = sorted(list(set(test_causes)))
    for cause in all_causes:
        idx = [i for i, c in enumerate(test_causes) if c == cause]
        y_true_c = y_test[idx]
        y_pred_c = y_pred_test[idx]
        
        # If no positive or negative exists in test set for this cause:
        cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        prec = precision_score(y_true_c, y_pred_c, zero_division=0)
        rec = recall_score(y_true_c, y_pred_c, zero_division=0)
        
        print(f"\nCause: {cause} (Group: {get_cause_group(cause)})")
        print(f"  TN: {tn:<6} | FP: {fp:<6}")
        print(f"  FN: {fn:<6} | TP: {tp:<6}")
        print(f"  Recall: {rec:.4f} | Precision: {prec:.4f} | Total: {len(y_true_c)} | Pos: {np.sum(y_true_c)}")

    print("\n" + "=" * 60)
    print("OVERALL CONFUSION MATRIX & RECALL ON TEST SPLIT")
    print("=" * 60)
    
    cm_all = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    tn_all, fp_all, fn_all, tp_all = cm_all[0][0], cm_all[0][1], cm_all[1][0], cm_all[1][1]
    
    prec_all = precision_score(y_test, y_pred_test, zero_division=0)
    rec_all = recall_score(y_test, y_pred_test, zero_division=0)
    
    print(f"  TN: {tn_all:<6} | FP: {fp_all:<6}")
    print(f"  FN: {fn_all:<6} | TP: {tp_all:<6}")
    print(f"  Overall Recall: {rec_all:.4f} | Overall Precision: {prec_all:.4f} | Total: {len(y_test)} | Pos: {np.sum(y_test)}\n")

    # Save visual confusion matrix plot
    save_path = str(Path(__file__).parent.parent / "reports" / "overall_confusion_matrix.png")
    plot_overall_confusion_matrix(cm_all, save_path=save_path)

def plot_overall_confusion_matrix(cm, save_path=None, title="Overall Model Confusion Matrix"):
    """Plot confusion matrix in the GNN-style format with proportional cell sizing."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract values
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    values = np.array([[tn, fp], [fn, tp]])

    # Normalize for color mapping (0-1)
    norm_values = values / total

    # Custom colormap: dark navy for high, light blue for low
    colors_list = ["#e8f4f8", "#1a3a5c"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_navy", colors_list)

    # Plot heatmap with proportional cell heights
    row_totals = [tn + fp, fn + tp]
    row_heights = [r / total for r in row_totals]

    # Minimum height so small rows are still visible
    min_height = 0.12
    if row_heights[1] < min_height:
        row_heights[1] = min_height
        row_heights[0] = 1.0 - min_height

    y_positions = [row_heights[1], 0]  # bottom-up: row 1 (Fraud) at bottom, row 0 (Safe) on top

    for i in range(2):
        for j in range(2):
            color_val = norm_values[i][j]
            cell_color = cmap(color_val)
            rect = plt.Rectangle(
                (j * 0.5, y_positions[i]),
                0.5,
                row_heights[i],
                facecolor=cell_color,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Text color: white on dark cells, black on light cells
            luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            text_color = "white" if luminance < 0.5 else "black"

            ax.text(
                j * 0.5 + 0.25,
                y_positions[i] + row_heights[i] / 2,
                str(values[i][j]),
                ha="center",
                va="center",
                fontsize=28,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, row_heights[0] + row_heights[1])
    ax.set_aspect("auto")

    # Axis labels
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["No Road Closure Required (0)", "Road Closure Required (1)"], fontsize=12, fontweight="bold")
    ax.xaxis.set_tick_params(length=6, width=2)

    ax.set_yticks([
        y_positions[1] + row_heights[1] / 2,
        y_positions[0] + row_heights[0] / 2,
    ])
    ax.set_yticklabels(["Road Closure Required (1)", "No Road Closure Required (0)"], fontsize=12, fontweight="bold")
    ax.yaxis.set_tick_params(length=6, width=2)

    ax.set_xlabel("Predicted Label (0 = No Road Closure Required, 1 = Road Closure Required)", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label (0 = No Road Closure Required, 1 = Road Closure Required)", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"\nConfusion matrix plot saved to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
