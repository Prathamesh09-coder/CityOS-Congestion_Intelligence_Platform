.PHONY: setup data train-m1 train-m2 train-m3 train-m4 eval test lint typecheck clean help

PYTHON := uv run python
UV := uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and create venv
	$(UV) sync
	$(UV) sync --extra deep --extra dev
	@echo "✅ Environment ready. Run 'uv run python ...' to execute scripts."

data: ## Ingest, clean, feature-engineer, and geo-impute → data/processed/
	$(PYTHON) -m astra_ml.data.ingest
	$(PYTHON) -m astra_ml.data.clean
	$(PYTHON) -m astra_ml.data.geo_impute
	$(PYTHON) -m astra_ml.data.features
	$(PYTHON) -m astra_ml.data.splits
	@echo "✅ Data pipeline complete. Output: data/processed/events_featured.parquet"

train-m1: ## Train M1: Closure-Necessity Classifier (RF baseline + CatBoost/LightGBM challengers)
	$(PYTHON) -m astra_ml.models.m1_closure_classifier
	@echo "✅ M1 training complete. Check MLflow and reports/m1_closure_classifier.md"

train-m2: ## Train M2: Dual-Regime Duration Estimator (acute + chronic/survival)
	$(PYTHON) -m astra_ml.models.m2_duration_acute
	$(PYTHON) -m astra_ml.models.m2_duration_chronic_survival
	@echo "✅ M2 training complete. Check MLflow and reports/m2_duration_estimator.md"

train-m3: ## Train M3: Multimodal Sparse-Event Forecaster (LoRA + MuRIL)
	$(PYTHON) -m astra_ml.models.m3_multimodal_fusion
	@echo "✅ M3 training complete. Check MLflow and reports/m3_multimodal_sparse.md"

train-m4: ## Train M4: Graph Spatio-Temporal Backbone (STUB — architecture validation only)
	$(PYTHON) -m astra_ml.models.m4_gnn_backbone
	@echo "✅ M4 stub complete. Check MLflow and reports/m4_gnn_backbone.md"

api: ## Run the FastAPI backend server
	$(PYTHON) -m astra_ml.api.main

eval: ## Generate all evaluation reports
	$(PYTHON) -m astra_ml.eval.reports
	@echo "✅ Reports generated in reports/"

test: ## Run pytest with coverage
	$(UV) run pytest

lint: ## Run ruff linter and formatter check
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

typecheck: ## Run mypy strict type checking
	$(UV) run mypy src/

clean: ## Remove generated artifacts (keeps raw data)
	rm -rf data/interim/* data/processed/* mlruns/ reports/*.md
	@echo "🧹 Cleaned intermediate data, MLflow runs, and reports."
