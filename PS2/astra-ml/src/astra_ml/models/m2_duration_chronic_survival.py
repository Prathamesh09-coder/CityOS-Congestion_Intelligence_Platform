"""M2 — Chronic-Regime Duration Estimator (Survival Analysis).

Uses survival analysis for chronic events (pot_holes, road_conditions, water_logging,
construction, debris) because:
  - Duration distribution is heavy-tailed (~4.5 day median / ~15 day mean)
  - Many events are right-censored (closed_datetime and resolved_datetime both null)
  - Dropping censored rows biases toward shorter durations

Models:
  1. Gradient Boosted Survival Trees (sksurv.ensemble)
  2. Cox PH with elastic-net (sksurv.linear_model.CoxnetSurvivalAnalysis)

Reports C-index and log-MAE on uncensored events separately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from astra_ml.eval.metrics import compute_survival_metrics
from astra_ml.eval.reports import write_regression_report
from astra_ml.utils.mlflow_utils import (
    log_dict_as_params,
    log_markdown_report,
    log_model_artifact,
    setup_experiment,
)
from astra_ml.utils.seeding import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_configs() -> tuple[dict, dict]:
    """Load data and M2 chronic configs."""
    data_cfg = OmegaConf.to_container(OmegaConf.load("configs/data.yaml"), resolve=True)
    m2_cfg = OmegaConf.to_container(OmegaConf.load("configs/m2_duration_chronic.yaml"), resolve=True)
    return data_cfg, m2_cfg  # type: ignore[return-value]


def _prepare_survival_data(
    data_cfg: dict,
    m2_cfg: dict,
) -> tuple:
    """Prepare data for survival analysis.

    Critical censoring logic:
    - event_observed=True → event resolved, duration_days is actual duration
    - event_observed=False → right-censored, duration_days is time from start to data cutoff

    Returns:
        Tuple of (X_train, y_train_struct, X_test, y_test_struct, features,
                  test_uncensored_true, test_uncensored_indices).
    """
    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)

    # Filter to chronic-regime events
    chronic_causes = m2_cfg.get("chronic_causes", [])
    df = df.filter(pl.col("event_cause").is_in(chronic_causes))
    logger.info("Chronic regime: %d records", df.height)

    # Must have valid duration_days (either observed or censored-to-cutoff)
    df = df.filter(
        pl.col("duration_days").is_not_null()
        & (pl.col("duration_days") > 0)
    )
    logger.info("After duration filter: %d records", df.height)

    # Log censoring statistics
    n_observed = df.filter(pl.col("event_observed")).height
    n_censored = df.filter(pl.col("event_observed").not_()).height
    logger.info("Observed: %d, Censored: %d (%.1f%%)", n_observed, n_censored, n_censored / df.height * 100)

    features = m2_cfg["features"]
    available_features = [f for f in features if f in df.columns]

    # Encode categoricals
    label_encoders: dict[str, LabelEncoder] = {}
    cat_cols = [f for f in available_features if f in ["event_cause", "corridor", "priority", "vehicle_type"]]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].cast(pl.Utf8).fill_null("__MISSING__").to_list())
            label_encoders[col] = le

    def encode(split_df: pl.DataFrame) -> np.ndarray:
        arrays = []
        for f in available_features:
            if f in label_encoders:
                vals = split_df[f].cast(pl.Utf8).fill_null("__MISSING__").to_list()
                arrays.append(label_encoders[f].transform(vals).reshape(-1, 1).astype(np.float64))
            elif f in split_df.columns:
                arr = split_df[f].to_numpy().astype(np.float64)
                arrays.append(np.nan_to_num(arr, nan=0.0).reshape(-1, 1))
        return np.hstack(arrays) if arrays else np.empty((split_df.height, 0))

    def make_survival_target(split_df: pl.DataFrame) -> np.ndarray:
        """Create structured array for sksurv: (event: bool, time: float)."""
        events = split_df["event_observed"].to_numpy().astype(bool)
        times = split_df["duration_days"].to_numpy().astype(np.float64)
        return np.array(
            list(zip(events, times)),
            dtype=[("event", bool), ("time", float)],
        )

    # Split
    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == "test")

    # Combine train + val for survival (small dataset)
    train_val_df = pl.concat([train_df, val_df])

    logger.info("Train+Val: %d, Test: %d", train_val_df.height, test_df.height)

    X_train = encode(train_val_df)
    y_train_struct = make_survival_target(train_val_df)

    X_test = encode(test_df)
    y_test_struct = make_survival_target(test_df)

    # Extract uncensored test durations for log-MAE
    test_observed_mask = test_df["event_observed"].to_numpy().astype(bool)
    test_uncensored_true = test_df.filter(pl.col("event_observed"))["duration_days"].to_numpy()

    return (
        X_train, y_train_struct,
        X_test, y_test_struct,
        label_encoders, available_features,
        test_uncensored_true, test_observed_mask,
    )


def train_gbst(data_cfg: dict, m2_cfg: dict) -> dict:
    """Train Gradient Boosted Survival Trees."""
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    seed = m2_cfg.get("seed", 42)
    set_global_seed(seed)

    gbst_config = m2_cfg.get("gbst", {})
    (
        X_train, y_train_struct,
        X_test, y_test_struct,
        label_encoders, features,
        test_uncensored_true, test_observed_mask,
    ) = _prepare_survival_data(data_cfg, m2_cfg)

    artifacts_dir = Path("models")
    artifacts_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(label_encoders, artifacts_dir / "m2_chronic_label_encoders.pkl")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.warning("Insufficient data for GBST — skipping")
        return {"name": "GBST", "c_index": 0.0}

    with mlflow.start_run(run_name="gbst_chronic"):
        mlflow.log_param("model_type", "GradientBoostingSurvivalAnalysis")
        mlflow.log_param("regime", "chronic")
        log_dict_as_params(gbst_config, prefix="gbst")

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=gbst_config.get("n_estimators", 200),
            learning_rate=gbst_config.get("learning_rate", 0.1),
            max_depth=gbst_config.get("max_depth", 4),
            min_samples_split=gbst_config.get("min_samples_split", 10),
            min_samples_leaf=gbst_config.get("min_samples_leaf", 5),
            subsample=gbst_config.get("subsample", 0.8),
            random_state=seed,
        )

        logger.info("Training GBST on chronic regime...")
        model.fit(X_train, y_train_struct)

        # Predict risk scores for C-index
        risk_scores = model.predict(X_test)

        # For uncensored test events, get predicted survival times for log-MAE
        test_uncensored_pred = None
        if test_observed_mask.sum() > 0:
            # Use risk scores for uncensored events as proxy
            # (higher risk = shorter duration in survival analysis)
            test_uncensored_pred = risk_scores[test_observed_mask]

        metrics = compute_survival_metrics(
            y_test_struct,
            risk_scores,
            y_true_uncensored=test_uncensored_true if len(test_uncensored_true) > 0 else None,
            y_pred_uncensored=test_uncensored_pred,
        )

        logger.info("GBST Chronic — C-index: %.4f", metrics.c_index)
        if metrics.log_mae_uncensored is not None:
            logger.info("GBST Chronic — log-MAE (uncensored): %.4f", metrics.log_mae_uncensored)

        mlflow.log_metrics(metrics.to_dict())
        log_model_artifact(model, "gbst_chronic")
        mlflow.log_artifact(str(artifacts_dir / "m2_chronic_label_encoders.pkl"), "models")
        
        # Save model locally in models/
        import joblib
        joblib.dump(model, artifacts_dir / "gbst_chronic.pkl")

        return {"name": "GBST", **metrics.to_dict()}


def train_coxnet(data_cfg: dict, m2_cfg: dict) -> dict:
    """Train Cox PH with elastic-net regularization."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    seed = m2_cfg.get("seed", 42)
    set_global_seed(seed)

    coxnet_config = m2_cfg.get("coxnet", {})
    (
        X_train, y_train_struct,
        X_test, y_test_struct,
        label_encoders, features,
        test_uncensored_true, test_observed_mask,
    ) = _prepare_survival_data(data_cfg, m2_cfg)

    artifacts_dir = Path("models")
    artifacts_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(label_encoders, artifacts_dir / "m2_chronic_label_encoders.pkl")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.warning("Insufficient data for CoxNet — skipping")
        return {"name": "CoxNet", "c_index": 0.0}

    with mlflow.start_run(run_name="coxnet_chronic"):
        mlflow.log_param("model_type", "CoxnetSurvivalAnalysis")
        mlflow.log_param("regime", "chronic")
        log_dict_as_params(coxnet_config, prefix="coxnet")

        model = CoxnetSurvivalAnalysis(
            n_alphas=coxnet_config.get("n_alphas", 100),
            l1_ratio=coxnet_config.get("l1_ratio", 0.5),
            max_iter=coxnet_config.get("max_iter", 1000),
            fit_baseline_model=coxnet_config.get("fit_baseline_model", True),
        )

        logger.info("Training CoxNet on chronic regime...")
        try:
            model.fit(X_train, y_train_struct)

            risk_scores = model.predict(X_test)

            test_uncensored_pred = None
            if test_observed_mask.sum() > 0:
                test_uncensored_pred = risk_scores[test_observed_mask]

            metrics = compute_survival_metrics(
                y_test_struct,
                risk_scores,
                y_true_uncensored=test_uncensored_true if len(test_uncensored_true) > 0 else None,
                y_pred_uncensored=test_uncensored_pred,
            )

            logger.info("CoxNet Chronic — C-index: %.4f", metrics.c_index)
            mlflow.log_metrics(metrics.to_dict())
            log_model_artifact(model, "coxnet_chronic")
            mlflow.log_artifact(str(artifacts_dir / "m2_chronic_label_encoders.pkl"), "models")
            
            # Save model locally in models/
            import joblib
            joblib.dump(model, artifacts_dir / "coxnet_chronic.pkl")

            return {"name": "CoxNet", **metrics.to_dict()}

        except Exception as e:
            logger.warning("CoxNet training failed: %s — this can happen with small/sparse chronic datasets", e)
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e)[:200])
            return {"name": "CoxNet", "c_index": 0.0, "status": "failed"}


def run_m2_chronic() -> None:
    """Run M2 chronic-regime survival pipeline."""
    data_cfg, m2_cfg = load_configs()
    seed = m2_cfg.get("seed", 42)
    set_global_seed(seed)

    setup_experiment(m2_cfg.get("experiment_name", "m2_duration_chronic"))

    logger.info("=" * 60)
    logger.info("M2 — Chronic-Regime Duration Estimator (Survival Analysis)")
    logger.info("=" * 60)

    # GBST
    gbst_metrics = train_gbst(data_cfg, m2_cfg)

    # CoxNet
    coxnet_metrics = train_coxnet(data_cfg, m2_cfg)

    # Pick winner
    winner = max([gbst_metrics, coxnet_metrics], key=lambda x: x.get("c_index", 0))

    report_path = write_regression_report(
        report_path="reports/m2_chronic_survival.md",
        model_name="M2 — Duration Estimator (Survival)",
        regime="chronic",
        metrics_dict=winner,
        pooled_baseline=None,
        split_strategy=(
            "Time-based split: Train+Val (< Mar 15 2024) combined due to small chronic sample, "
            "Test (> Mar 15 2024). Right-censored events included in training."
        ),
        known_limitations=(
            "- Chronic regime has fewer labeled events than acute.\n"
            "- Right-censored events are included via survival analysis, preventing "
            "the bias toward shorter durations that dropping them would cause.\n"
            "- C-index measures ranking accuracy, not calibration — a high C-index "
            "doesn't guarantee well-calibrated survival curves.\n"
            "- CoxNet may fail on very sparse data; GBST is more robust.\n"
            "- log-MAE is computed only on uncensored test events, which may not "
            "be representative of all chronic events."
        ),
    )

    with mlflow.start_run(run_name="m2_chronic_final_report"):
        log_markdown_report(report_path)

    logger.info("✅ M2 chronic complete. Report: %s", report_path)


if __name__ == "__main__":
    run_m2_chronic()
