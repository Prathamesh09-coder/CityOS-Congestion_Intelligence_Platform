"""M3 — Sparse/Novel Event Multimodal Forecaster.

Architecture:
  - Frozen MuRIL/IndicBERT text encoder + LoRA adapters on top layers
  - Structured feature embeddings (event_cause, priority, corridor, time features)
  - Fusion MLP head with dual output: closure probability + log-duration
  - Multi-task training (BCE + MSE, weighted)

Evaluation:
  - Leave-one-cause-out CV on sparse classes (vip_movement, protest, procession, public_event)
  - Simulates true cold-start: model has zero examples of the held-out event type

Requires: torch, transformers, peft (install via `uv sync --extra deep`)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from astra_ml.eval.metrics import compute_classification_metrics
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
    """Load data and M3 configs."""
    data_cfg = OmegaConf.to_container(OmegaConf.load("configs/data.yaml"), resolve=True)
    m3_cfg = OmegaConf.to_container(OmegaConf.load("configs/m3_multimodal_sparse.yaml"), resolve=True)
    return data_cfg, m3_cfg  # type: ignore[return-value]


def _check_torch_available() -> bool:
    """Check if PyTorch and transformers are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model: dynamic MuRIL text encoder + LoRA + structured features -> dual-head output."""

    def __init__(self, m3_cfg: dict, tokenizer: Any, base_model: Any) -> None:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        super().__init__()
        self.cfg = m3_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.tokenizer = tokenizer

        # Wrap base_model in LoRA
        lora_cfg = m3_cfg.get("lora", {})
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            target_modules=list(lora_cfg.get("target_modules", ["query", "value"])),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias="none",
        )
        self.text_encoder = get_peft_model(base_model, peft_config)

        text_dim = base_model.config.hidden_size

        # Text projection (trainable projection/bottleneck layer)
        self.text_projection = nn.Linear(text_dim, text_dim)

        # Structured feature embedding
        struct_cfg = m3_cfg["structured_features"]
        self.embedding_dim = struct_cfg.get("embedding_dim", 32)

        # Fusion head
        fusion_cfg = m3_cfg["fusion_head"]
        struct_dim = 0
        if struct_cfg.get("categorical"):
            struct_dim += self.embedding_dim * len(struct_cfg["categorical"])
        if struct_cfg.get("numerical"):
            struct_dim += len(struct_cfg["numerical"])
        if struct_dim == 0:
            struct_dim = 1

        hidden_dims = fusion_cfg.get("hidden_dims", [256, 128])
        dropout = fusion_cfg.get("dropout", 0.3)

        layers: list[nn.Module] = []
        in_dim = text_dim + struct_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.fusion_mlp = nn.Sequential(*layers)

        # Dual output heads
        self.closure_head = nn.Linear(hidden_dims[-1], 1)  # Binary
        self.duration_head = nn.Linear(hidden_dims[-1], 1)  # Regression

        # Category embeddings
        self.cat_embeddings = nn.ModuleDict()
        self.cat_vocab: dict[str, dict[str, int]] = {}

    def _build_cat_embeddings(self, df: pl.DataFrame) -> None:
        """Build embedding layers for categorical features from training data."""
        import torch.nn as nn

        struct_cfg = self.cfg["structured_features"]
        for cat_col in struct_cfg.get("categorical", []):
            if cat_col in df.columns:
                unique_vals = df[cat_col].cast(pl.Utf8).fill_null("__MISSING__").unique().to_list()
                vocab = {v: i for i, v in enumerate(unique_vals)}
                vocab["__UNK__"] = len(vocab)
                self.cat_vocab[cat_col] = vocab
                self.cat_embeddings[cat_col] = nn.Embedding(
                    len(vocab), self.embedding_dim
                )

    def _encode_text(self, texts: list[str]) -> Any:
        """Dynamically tokenize and encode text strings using the LoRA-wrapped text encoder."""
        import torch

        max_length = self.cfg["text_encoder"].get("max_length", 128)
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        outputs = self.text_encoder(**encoding)
        # Use CLS token representation
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.text_projection(cls_emb)

    def _encode_structured(self, batch_df: pl.DataFrame) -> Any:
        """Encode structured features for a batch."""
        import torch

        struct_cfg = self.cfg["structured_features"]
        parts: list[Any] = []

        # Categorical embeddings
        for cat_col in struct_cfg.get("categorical", []):
            if cat_col in self.cat_embeddings:
                vocab = self.cat_vocab[cat_col]
                vals = batch_df[cat_col].cast(pl.Utf8).fill_null("__MISSING__").to_list()
                indices = [vocab.get(v, vocab["__UNK__"]) for v in vals]
                idx_tensor = torch.tensor(indices, device=self.device)
                emb = self.cat_embeddings[cat_col](idx_tensor)
                parts.append(emb)

        # Numerical features
        for num_col in struct_cfg.get("numerical", []):
            if num_col in batch_df.columns:
                vals = batch_df[num_col].to_numpy().astype(np.float32)
                vals = np.nan_to_num(vals, nan=0.0)
                parts.append(torch.tensor(vals, device=self.device).unsqueeze(1))

        if parts:
            return torch.cat(parts, dim=1).float()
        return torch.zeros(batch_df.height, 1, device=self.device)

    def train_epoch(
        self,
        df: pl.DataFrame,
        optimizer: Any,
        batch_size: int = 16,
    ) -> float:
        """Train one epoch, return average loss."""
        import torch
        import torch.nn.functional as F

        self.text_encoder.train()
        self.text_projection.train()
        self.fusion_mlp.train()
        self.closure_head.train()
        self.duration_head.train()

        task_heads = self.cfg.get("task_heads", {})
        closure_weight = task_heads.get("closure", {}).get("weight", 1.0)
        duration_weight = task_heads.get("duration", {}).get("weight", 0.5)

        total_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(df.height)

        for start in range(0, df.height, batch_size):
            end = min(start + batch_size, df.height)
            batch_idx = indices[start:end].tolist()
            batch_df = df[batch_idx]

            # Forward pass using dynamic text encoding
            text_fields = self.cfg.get("text_fields", ["description", "comment", "address"])
            batch_texts = []
            for i in range(batch_df.height):
                parts = []
                for field in text_fields:
                    if field in batch_df.columns:
                        val = batch_df[field][i]
                        if val is not None:
                            parts.append(str(val))
                batch_texts.append(" [SEP] ".join(parts) if parts else "")

            text_emb = self._encode_text(batch_texts)
            struct_emb = self._encode_structured(batch_df)
            fused = torch.cat([text_emb, struct_emb], dim=1)
            hidden = self.fusion_mlp(fused)

            # Closure head
            closure_logits = self.closure_head(hidden).squeeze(-1)
            if "requires_road_closure" in batch_df.columns:
                closure_target = torch.tensor(
                    batch_df["requires_road_closure"].cast(pl.Float32).to_numpy(),
                    device=self.device,
                )
                closure_loss = F.binary_cross_entropy_with_logits(closure_logits, closure_target)
            else:
                closure_loss = torch.tensor(0.0, device=self.device)

            # Duration head
            duration_pred = self.duration_head(hidden).squeeze(-1)
            if "log_duration_minutes" in batch_df.columns:
                dur_vals = batch_df["log_duration_minutes"].to_numpy().astype(np.float32)
                dur_mask = ~np.isnan(dur_vals)
                if dur_mask.sum() > 0:
                    dur_target = torch.tensor(dur_vals[dur_mask], device=self.device)
                    dur_pred_masked = duration_pred[torch.tensor(dur_mask, device=self.device)]
                    duration_loss = F.mse_loss(dur_pred_masked, dur_target)
                else:
                    duration_loss = torch.tensor(0.0, device=self.device)
            else:
                duration_loss = torch.tensor(0.0, device=self.device)

            loss = closure_weight * closure_loss + duration_weight * duration_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict_closure(self, df: pl.DataFrame, batch_size: int = 32) -> np.ndarray:
        """Predict closure probabilities."""
        import torch

        self.text_encoder.eval()
        self.text_projection.eval()
        self.fusion_mlp.eval()
        self.closure_head.eval()

        all_probs: list[float] = []

        with torch.no_grad():
            for start in range(0, df.height, batch_size):
                end = min(start + batch_size, df.height)
                batch_df = df[start:end]

                text_fields = self.cfg.get("text_fields", ["description", "comment", "address"])
                batch_texts = []
                for i in range(batch_df.height):
                    parts = []
                    for field in text_fields:
                        if field in batch_df.columns:
                            val = batch_df[field][i]
                            if val is not None:
                                parts.append(str(val))
                    batch_texts.append(" [SEP] ".join(parts) if parts else "")

                text_emb = self._encode_text(batch_texts)
                struct_emb = self._encode_structured(batch_df)
                fused = torch.cat([text_emb, struct_emb], dim=1)
                hidden = self.fusion_mlp(fused)

                logits = self.closure_head(hidden).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())

        return np.array(all_probs)


def run_loo_cv(data_cfg: dict, m3_cfg: dict) -> dict[str, dict]:
    """Run leave-one-cause-out cross-validation on sparse classes with dynamic fine-tuning.

    For each sparse cause: train on all OTHER causes, predict on the held-out cause.
    This simulates true cold-start performance.

    Returns:
        Dict mapping cause name → metrics dict.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)

    # Ensure df has row_idx
    try:
        df = df.with_row_index("row_idx")
    except AttributeError:
        df = df.with_row_count("row_idx")

    model_name = m3_cfg["text_encoder"]["model_name"]
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    loo_config = m3_cfg["loo_eval"]
    sparse_causes = loo_config["sparse_causes"]
    training_cfg = m3_cfg["training"]
    seed = m3_cfg.get("seed", 42)

    results: dict[str, dict] = {}

    for held_out_cause in sparse_causes:
        logger.info("=" * 40)
        logger.info("LOO-CV: holding out '%s'", held_out_cause)

        # Split: train on everything except the held-out cause
        train_df = df.filter(pl.col("event_cause") != held_out_cause)
        test_df = df.filter(pl.col("event_cause") == held_out_cause)

        if test_df.height == 0:
            logger.warning("No records for cause '%s' — skipping", held_out_cause)
            results[held_out_cause] = {"n_test": 0, "status": "no_data"}
            continue

        # Subsample training data to 800 samples to keep training extremely fast on CPU/MPS
        if train_df.height > 800:
            train_df = train_df.sample(n=800, seed=seed)

        logger.info("Train (subsampled): %d records, Test (%s): %d records", train_df.height, held_out_cause, test_df.height)

        # Load a fresh base model for this fold to prevent any weight leakage
        base_model = AutoModel.from_pretrained(model_name)
        model = MultimodalFusionModel(m3_cfg, tokenizer, base_model)
        model._build_cat_embeddings(train_df)
        model.to(model.device)

        # Optimizer (only trainable params)
        trainable_params = []
        trainable_params.extend([p for p in model.text_encoder.parameters() if p.requires_grad])
        trainable_params.extend(model.text_projection.parameters())
        trainable_params.extend(model.fusion_mlp.parameters())
        trainable_params.extend(model.closure_head.parameters())
        trainable_params.extend(model.duration_head.parameters())
        for emb in model.cat_embeddings.values():
            trainable_params.extend(emb.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_cfg.get("learning_rate", 2e-4),
            weight_decay=training_cfg.get("weight_decay", 0.01),
        )

        # Train
        n_epochs = training_cfg.get("epochs", 30)
        best_loss = float("inf")
        patience = training_cfg.get("early_stopping_patience", 5)
        patience_counter = 0

        for epoch in range(n_epochs):
            loss = model.train_epoch(
                train_df,
                optimizer,
                batch_size=training_cfg.get("batch_size", 16),
            )

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (loss: %.4f)", epoch, loss)
                break

            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, n_epochs, loss)

        # Evaluate on held-out cause
        y_prob = model.predict_closure(test_df)
        y_true = test_df["requires_road_closure"].cast(pl.Int32).to_numpy()

        # Handle edge cases (all same class)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        cause_result: dict[str, Any] = {
            "n_test": test_df.height,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "mean_pred_prob": float(y_prob.mean()),
        }

        if n_pos > 0 and n_neg > 0:
            metrics = compute_classification_metrics(y_true, y_prob)
            cause_result.update(metrics.to_dict())
            logger.info(
                "%s — ROC-AUC: %.4f, PR-AUC: %.4f (n=%d, pos=%d)",
                held_out_cause, metrics.roc_auc, metrics.pr_auc, test_df.height, n_pos,
            )
        else:
            cause_result["status"] = "single_class"
            logger.warning(
                "%s — single class in test (pos=%d, neg=%d), cannot compute AUC",
                held_out_cause, n_pos, n_neg,
            )

        results[held_out_cause] = cause_result

        # Clean up memory completely for the next fold
        del model
        del base_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results



def train_and_save_m3(data_cfg: dict, m3_cfg: dict) -> None:
    """Train the final M3 model on the full dataset and save to disk."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    import os

    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)

    try:
        df = df.with_row_index("row_idx")
    except AttributeError:
        df = df.with_row_count("row_idx")

    model_name = m3_cfg["text_encoder"]["model_name"]
    logger.info("Loading tokenizer and base model for final training: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    model = MultimodalFusionModel(m3_cfg, tokenizer, base_model)
    model._build_cat_embeddings(df)
    model.to(model.device)

    training_cfg = m3_cfg["training"]
    seed = m3_cfg.get("seed", 42)

    trainable_params = []
    trainable_params.extend([p for p in model.text_encoder.parameters() if p.requires_grad])
    trainable_params.extend(model.text_projection.parameters())
    trainable_params.extend(model.fusion_mlp.parameters())
    trainable_params.extend(model.closure_head.parameters())
    trainable_params.extend(model.duration_head.parameters())
    for emb in model.cat_embeddings.values():
        trainable_params.extend(emb.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_cfg.get("learning_rate", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    n_epochs = training_cfg.get("epochs", 30)
    logger.info("Training final M3 model on full dataset (%d records) for %d epochs...", df.height, n_epochs)

    for epoch in range(n_epochs):
        loss = model.train_epoch(
            df,
            optimizer,
            batch_size=training_cfg.get("batch_size", 16),
        )
        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, n_epochs, loss)

    os.makedirs("models", exist_ok=True)
    save_path = "models/m3_model.pth"
    
    logger.info("Saving final model to %s", save_path)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "cat_vocab": model.cat_vocab,
        "embedding_dim": model.embedding_dim,
        "cfg": model.cfg
    }
    torch.save(checkpoint, save_path)
    logger.info("✅ Final M3 model weights saved successfully.")


def run_m3() -> None:
    """Run M3 multimodal sparse-event forecaster."""
    import torch
    if not _check_torch_available():
        logger.error(
            "PyTorch, transformers, and peft are required for M3. "
            "Install with: uv sync --extra deep"
        )
        return

    data_cfg, m3_cfg = load_configs()
    seed = m3_cfg.get("seed", 42)
    set_global_seed(seed)

    setup_experiment(m3_cfg.get("experiment_name", "m3_multimodal_sparse"))

    logger.info("=" * 60)
    logger.info("M3 — Sparse/Novel Event Multimodal Forecaster")
    logger.info("=" * 60)

    # Run Final Full-Dataset Training and Save Model
    train_and_save_m3(data_cfg, m3_cfg)

    logger.info("✅ M3 complete.")


if __name__ == "__main__":
    import torch.nn as nn
    run_m3()

