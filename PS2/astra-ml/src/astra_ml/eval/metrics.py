"""Evaluation metrics for all ASTRA models.

- Classification: ROC-AUC, PR-AUC, calibration, precision/recall at threshold
- Regression: log-MAE, RMSE, MAE
- Survival: Concordance index (C-index)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for binary classification evaluation metrics."""

    roc_auc: float
    pr_auc: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    confusion_matrix: list[list[int]]
    calibration_prob_true: list[float] = field(default_factory=list)
    calibration_prob_pred: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for MLflow logging."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision_at_threshold": self.precision_at_threshold,
            "recall_at_threshold": self.recall_at_threshold,
            "threshold": self.threshold,
        }


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics."""

    log_mae: float
    mae: float
    rmse: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for MLflow logging."""
        return {
            "log_mae": self.log_mae,
            "mae": self.mae,
            "rmse": self.rmse,
        }


@dataclass
class SurvivalMetrics:
    """Container for survival analysis metrics."""

    c_index: float
    log_mae_uncensored: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for MLflow logging."""
        d: dict[str, Any] = {"c_index": self.c_index}
        if self.log_mae_uncensored is not None:
            d["log_mae_uncensored"] = self.log_mae_uncensored
        return d


def compute_classification_metrics(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    target_recall: float = 0.85,
) -> ClassificationMetrics:
    """Compute full classification metrics for imbalanced binary classification.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        target_recall: Minimum recall to achieve when selecting operational threshold.
            The threshold is set to the highest value that still achieves this recall.

    Returns:
        ClassificationMetrics with all computed values.
    """
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Find threshold that achieves target recall (favor recall over precision)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # recalls is sorted descending; find the last index where recall >= target
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_indices) > 0:
        # Pick the threshold that gives highest precision while meeting recall target
        best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]
        threshold = float(thresholds[best_idx])
    else:
        # If no threshold achieves target recall, use the one with max recall
        threshold = float(thresholds[0]) if len(thresholds) > 0 else 0.5

    y_pred = (y_prob >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        cal_true = prob_true.tolist()
        cal_pred = prob_pred.tolist()
    except ValueError:
        cal_true, cal_pred = [], []

    return ClassificationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision_at_threshold=prec,
        recall_at_threshold=rec,
        threshold=threshold,
        confusion_matrix=cm,
        calibration_prob_true=cal_true,
        calibration_prob_pred=cal_pred,
    )


def compute_regression_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    is_log_scale: bool = True,
) -> RegressionMetrics:
    """Compute regression metrics, expecting log-transformed targets.

    Args:
        y_true: Ground truth values (in log-space if is_log_scale).
        y_pred: Predicted values (in log-space if is_log_scale).
        is_log_scale: Whether the inputs are already log-transformed.

    Returns:
        RegressionMetrics with log-MAE, MAE, and RMSE.
    """
    log_mae = float(mean_absolute_error(y_true, y_pred))

    if is_log_scale:
        # Also compute MAE in original scale
        y_true_orig = np.exp(y_true)
        y_pred_orig = np.exp(y_pred)
        mae = float(mean_absolute_error(y_true_orig, y_pred_orig))
        rmse = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    else:
        mae = log_mae
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return RegressionMetrics(log_mae=log_mae, mae=mae, rmse=rmse)


def compute_survival_metrics(
    y_true_structured: NDArray[Any],
    risk_scores: NDArray[np.float64],
    y_true_uncensored: NDArray[np.float64] | None = None,
    y_pred_uncensored: NDArray[np.float64] | None = None,
) -> SurvivalMetrics:
    """Compute survival analysis metrics.

    Args:
        y_true_structured: Structured array with fields (event, time) for sksurv.
        risk_scores: Predicted risk scores from the survival model.
        y_true_uncensored: Optional true durations for uncensored events only.
        y_pred_uncensored: Optional predicted durations for uncensored events only.

    Returns:
        SurvivalMetrics with C-index and optional log-MAE on uncensored events.
    """
    from sksurv.metrics import concordance_index_censored

    event_indicator = y_true_structured["event"]
    event_time = y_true_structured["time"]

    c_index = float(
        concordance_index_censored(event_indicator, event_time, risk_scores)[0]
    )

    log_mae_unc = None
    if y_true_uncensored is not None and y_pred_uncensored is not None:
        log_mae_unc = float(
            mean_absolute_error(np.log1p(y_true_uncensored), np.log1p(y_pred_uncensored))
        )

    return SurvivalMetrics(c_index=c_index, log_mae_uncensored=log_mae_unc)
