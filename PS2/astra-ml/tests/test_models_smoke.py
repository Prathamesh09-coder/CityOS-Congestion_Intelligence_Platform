"""Smoke tests for model training — verify each model can train on a tiny dataset without crashing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def _make_tiny_dataset(n: int = 50) -> pl.DataFrame:
    """Create a tiny synthetic dataset with all required columns."""
    rng = np.random.RandomState(42)

    causes = ["accident", "pot_holes", "congestion", "construction", "vehicle_breakdown"]
    corridors = ["Tumkur Road", "ORR East 1", "Non-corridor"]

    return pl.DataFrame({
        "event_id": [f"E{i:04d}" for i in range(n)],
        "event_cause": rng.choice(causes, n).tolist(),
        "corridor": rng.choice(corridors, n).tolist(),
        "priority": rng.choice(["High", "Low"], n).tolist(),
        "vehicle_type": rng.choice(["lcv", "heavy_vehicle", "bmtc_bus", None], n).tolist(),
        "requires_road_closure": rng.choice([True, False], n, p=[0.1, 0.9]).tolist(),
        "hour": rng.randint(0, 24, n).tolist(),
        "day_of_week": rng.randint(1, 8, n).tolist(),
        "hour_sin": np.sin(rng.randint(0, 24, n) * 2 * np.pi / 24).tolist(),
        "hour_cos": np.cos(rng.randint(0, 24, n) * 2 * np.pi / 24).tolist(),
        "dow_sin": np.sin(rng.randint(1, 8, n) * 2 * np.pi / 7).tolist(),
        "dow_cos": np.cos(rng.randint(1, 8, n) * 2 * np.pi / 7).tolist(),
        "cause_closure_rate": rng.uniform(0, 1, n).tolist(),
        "corridor_closure_rate": rng.uniform(0, 1, n).tolist(),
        "junction_missing": rng.choice([True, False], n).tolist(),
        "zone_missing": rng.choice([True, False], n).tolist(),
        "closed_datetime_missing": rng.choice([True, False], n).tolist(),
        "vehicle_type_missing": rng.choice([True, False], n).tolist(),
        "zone_imputed": rng.choice([True, False], n).tolist(),
        "junction_imputed": rng.choice([True, False], n).tolist(),
        "description_len": rng.randint(0, 200, n).tolist(),
        "comment_len": rng.randint(0, 100, n).tolist(),
        "is_weekend": rng.choice([True, False], n).tolist(),
        "month": rng.randint(1, 13, n).tolist(),
        "duration_minutes": (rng.exponential(60, n) + 1).tolist(),
        "log_duration_minutes": np.log(rng.exponential(60, n) + 1).tolist(),
        "duration_regime": rng.choice(["acute", "chronic"], n).tolist(),
        "event_observed": rng.choice([True, False], n, p=[0.6, 0.4]).tolist(),
        "duration_days": (rng.exponential(5, n) + 0.1).tolist(),
        "split": (["train"] * 30 + ["val"] * 10 + ["test"] * 10),
    })


class TestM1Smoke:
    """Smoke test: M1 RandomForest should train without errors."""

    def test_rf_trains_on_tiny_data(self) -> None:
        """RandomForest should fit on 50 rows without crashing."""
        from sklearn.ensemble import RandomForestClassifier

        df = _make_tiny_dataset()
        train = df.filter(pl.col("split") == "train")

        features = ["hour", "day_of_week"]
        X = train.select(features).to_numpy().astype(np.float64)
        y = train["requires_road_closure"].cast(pl.Int32).to_numpy()

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        y_prob = model.predict_proba(X)
        assert y_prob.shape[0] == X.shape[0]
        assert y_prob.shape[1] == 2


class TestM2AcuteSmoke:
    """Smoke test: M2 acute regressor should train without errors."""

    def test_catboost_regressor_trains(self) -> None:
        """CatBoost regressor should fit on tiny data."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("catboost not installed")

        df = _make_tiny_dataset()
        train = df.filter(pl.col("split") == "train")

        X = train.select(["hour_sin", "hour_cos", "cause_closure_rate"]).to_numpy()
        y = train["log_duration_minutes"].to_numpy()

        model = CatBoostRegressor(iterations=10, verbose=0, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(y)


class TestM2ChronicSmoke:
    """Smoke test: M2 chronic survival model should train without errors."""

    def test_gbst_trains(self) -> None:
        """Gradient Boosted Survival Trees should fit on structured data."""
        try:
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        except ImportError:
            pytest.skip("scikit-survival not installed")

        n = 50
        rng = np.random.RandomState(42)
        X = rng.randn(n, 3)

        events = rng.choice([True, False], n, p=[0.6, 0.4])
        times = rng.exponential(5, n) + 0.1

        y = np.array(
            list(zip(events, times)),
            dtype=[("event", bool), ("time", float)],
        )

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=10, max_depth=2, random_state=42
        )
        model.fit(X, y)

        risk_scores = model.predict(X)
        assert len(risk_scores) == n
