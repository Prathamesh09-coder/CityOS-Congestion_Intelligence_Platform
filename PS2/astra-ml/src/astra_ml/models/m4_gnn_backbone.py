"""M4 — Graph Spatio-Temporal Backbone (STUB / Architecture Validation Only).

⚠️  WARNING: This module is a STUB. It does NOT produce real model performance metrics.
⚠️  The ASTRAM event log CSV does NOT contain live traffic-speed data, which is
⚠️  REQUIRED to train this model for production use.

What this module does:
  1. Loads the shared cached OSM road graph (from data/interim/)
  2. Defines a Graph WaveNet-style architecture (PyTorch Geometric)
  3. Runs a training loop against SYNTHETIC node features for architecture validation

What it does NOT do:
  - Predict real traffic congestion
  - Use real traffic-speed signals
  - Report production-grade metrics

To make this production-ready, you need:
  - Live traffic speed per road segment (5-min or 15-min granularity)
  - Historical speed data matching the event log time range
  - Road segment IDs mapped to OSM way IDs
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
from omegaconf import OmegaConf

from astra_ml.utils.mlflow_utils import (
    log_dict_as_params,
    log_markdown_report,
    setup_experiment,
)
from astra_ml.utils.seeding import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_configs() -> tuple[dict, dict]:
    """Load data and M4 configs."""
    data_cfg = OmegaConf.to_container(OmegaConf.load("configs/data.yaml"), resolve=True)
    m4_cfg = OmegaConf.to_container(OmegaConf.load("configs/m4_gnn_backbone.yaml"), resolve=True)
    return data_cfg, m4_cfg  # type: ignore[return-value]


def _check_deps() -> bool:
    """Check if PyTorch Geometric and osmnx are available."""
    try:
        import torch
        import networkx
        return True
    except ImportError:
        return False


def _build_graph_wavenet(m4_cfg: dict):  # type: ignore[no-untyped-def]
    """Define a simplified Graph WaveNet architecture.

    This is a model DEFINITION only — architecture validation, not production training.
    """
    import torch
    import torch.nn as nn

    model_cfg = m4_cfg["model"]

    class GraphWaveNetStub(nn.Module):
        """Simplified Graph WaveNet for architecture validation.

        ⚠️  STUB: Uses synthetic data. Not for production inference.
        """

        def __init__(
            self,
            num_nodes: int,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.num_nodes = num_nodes

            # Adaptive adjacency matrix (learnable)
            self.node_emb1 = nn.Embedding(num_nodes, 10)
            self.node_emb2 = nn.Embedding(num_nodes, 10)

            # Temporal convolutions (dilated causal)
            self.temporal_convs = nn.ModuleList()
            self.skip_convs = nn.ModuleList()
            self.graph_convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            self.input_proj = nn.Linear(in_channels, hidden_channels)

            for i in range(num_layers):
                self.temporal_convs.append(
                    nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=2**i, dilation=2**i)
                )
                self.skip_convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
                self.graph_convs.append(nn.Linear(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, num_nodes, timesteps, in_channels).

            Returns:
                Output tensor of shape (batch, num_nodes, out_channels).
            """
            batch_size, n_nodes, timesteps, in_ch = x.shape

            # Project input
            h = self.input_proj(x)  # (B, N, T, H)
            h = h.permute(0, 1, 3, 2)  # (B, N, H, T)
            h = h.reshape(batch_size * n_nodes, -1, timesteps)  # (B*N, H, T)

            skip_sum = 0
            for tc, sc, gc, norm in zip(
                self.temporal_convs, self.skip_convs, self.graph_convs, self.norms
            ):
                # Temporal conv with gated activation
                tc_out = tc(h)[:, :, :timesteps]
                gate = torch.sigmoid(tc_out[:, tc_out.shape[1]//2:])
                filter_ = torch.tanh(tc_out[:, :tc_out.shape[1]//2])
                h_temp = gate * filter_

                # Skip connection
                skip_sum = skip_sum + sc(h_temp)[:, :, -1:]

                # Simple graph mixing (placeholder for full graph conv)
                h_node = h_temp[:, :, -1]  # (B*N, H)
                h_graph = gc(h_node)  # (B*N, H)
                h_graph = h_graph.unsqueeze(-1)  # (B*N, H, 1)

                # Residual
                h = h[:, :, :timesteps] + h_graph.expand_as(h[:, :, :timesteps])
                h = self.dropout(h)

            # Output from skip connections
            out = skip_sum.squeeze(-1)  # (B*N, H)
            out = self.output(out)  # (B*N, out_ch)
            out = out.reshape(batch_size, n_nodes, -1)  # (B, N, out_ch)

            return out

    return GraphWaveNetStub


def run_m4() -> None:
    """Run M4 graph backbone stub — architecture validation only."""
    if not _check_deps():
        logger.error(
            "PyTorch and networkx are required for M4. "
            "Install with: uv sync --extra deep"
        )
        return

    import torch

    data_cfg, m4_cfg = load_configs()
    seed = m4_cfg.get("seed", 42)
    set_global_seed(seed)

    setup_experiment(m4_cfg.get("experiment_name", "m4_gnn_backbone"))

    logger.info("=" * 60)
    logger.info("M4 — Graph Spatio-Temporal Backbone")
    logger.info("⚠️  STUB: Architecture validation with SYNTHETIC data only")
    logger.info("=" * 60)

    model_cfg = m4_cfg["model"]
    train_cfg = m4_cfg["training"]

    # Try to load road graph for node count and feature extraction
    try:
        from astra_ml.data.road_graph import get_or_build_road_graph
        graph = get_or_build_road_graph(data_cfg)
        num_nodes = min(graph.number_of_nodes(), 200)  # Cap for toy training
        logger.info("Using %d nodes from OSM graph (capped from %d)", num_nodes, graph.number_of_nodes())
        
        # Extract static road graph features
        import networkx as nx
        nodes = list(graph.nodes())[:num_nodes]
        degrees = dict(graph.degree())
        try:
            k = min(20, len(nodes))
            bet_cent = nx.betweenness_centrality(graph, k=k, normalized=True)
        except Exception:
            bet_cent = {n: 0.0 for n in nodes}
            
        static_features_list = []
        for node in nodes:
            deg = float(degrees.get(node, 0))
            bc = float(bet_cent.get(node, 0.0))
            is_intersect = 1.0 if deg > 2 else 0.0
            static_features_list.append([deg, bc, is_intersect])
        static_features = np.array(static_features_list, dtype=np.float32)
    except Exception as e:
        logger.warning("Could not load road graph or compute features (%s) — using 50 synthetic nodes", e)
        num_nodes = 50
        static_features = np.random.randn(num_nodes, 3).astype(np.float32)

    in_channels_base = model_cfg.get("in_channels", 4)
    in_channels_total = in_channels_base + 3

    # Build model
    GraphWaveNet = _build_graph_wavenet(m4_cfg)
    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_channels=in_channels_total,
        out_channels=model_cfg.get("out_channels", 1),
        hidden_channels=model_cfg.get("hidden_channels", 32),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.3),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total_params)

    # Generate SYNTHETIC data for architecture validation
    logger.info("⚠️  Generating SYNTHETIC traffic data — NOT real observations")
    num_timesteps = train_cfg.get("num_timesteps", 168)
    batch_size = train_cfg.get("batch_size", 32)

    # Base Synthetic: random speed, flow, hour_sin, hour_cos
    X_base = torch.randn(batch_size, num_nodes, num_timesteps, in_channels_base).to(device)
    
    # Static features: (num_nodes, 3) -> (1, num_nodes, 1, 3) -> (batch_size, num_nodes, num_timesteps, 3)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).to(device)
    static_expanded = static_tensor.unsqueeze(0).unsqueeze(2).expand(batch_size, num_nodes, num_timesteps, 3)
    
    # Concatenate temporal synthetic features and static graph features
    X_synthetic = torch.cat([X_base, static_expanded], dim=-1)
    y_synthetic = torch.randn(batch_size, num_nodes, 1).to(device)

    # Training loop (architecture validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 0.001))
    n_epochs = train_cfg.get("epochs", 10)

    with mlflow.start_run(run_name="m4_stub_architecture_validation"):
        mlflow.log_param("is_stub", True)
        mlflow.log_param("synthetic_data", True)
        mlflow.log_param("num_nodes", num_nodes)
        mlflow.log_param("total_params", total_params)
        log_dict_as_params(model_cfg, prefix="model")

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            y_pred = model(X_synthetic)
            loss = torch.nn.functional.mse_loss(y_pred, y_synthetic)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            mlflow.log_metric("train_loss_synthetic", loss_val, step=epoch)
            logger.info("Epoch %d/%d — synthetic loss: %.6f", epoch + 1, n_epochs, loss_val)

        mlflow.log_metric("final_synthetic_loss", loss_val)

    # Generate report
    report_path = Path("reports/m4_gnn_backbone.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_text = f"""# M4 — Graph Spatio-Temporal Backbone

> ⚠️ **STUB: Architecture Validation Only**
>
> This model was trained on **SYNTHETIC** data. The metrics below do NOT represent
> real traffic prediction performance. They only validate that the architecture
> compiles, trains, and converges on toy data.

## Architecture

- **Type**: Graph WaveNet (simplified)
- **Nodes**: {num_nodes} (from OSM road graph, capped)
- **Input channels**: {in_channels_total} (synthetic: speed, flow, hour_sin, hour_cos + static graph: degree, betweenness centrality, is_intersection)
- **Hidden channels**: {model_cfg.get('hidden_channels', 32)}
- **Layers**: {model_cfg.get('num_layers', 3)}
- **Total parameters**: {total_params:,}

## Training (Synthetic Data)

- **Epochs**: {n_epochs}
- **Final loss**: {loss_val:.6f}
- **Data**: Random Gaussian tensors — NOT real traffic

## What's Needed for Production

To make this model production-ready, the following external data is required:

1. **Live traffic speed** per road segment (5-min or 15-min granularity)
   from Google Maps Traffic API, HERE, TomTom, or BTRAC feeds.
2. **Historical speed data** covering November 2023 – April 2024 to align
   with the ASTRAM event log.
3. **Road segment IDs** mapped to OSM way IDs for graph alignment.

## Known Limitations

- Architecture validation only — no real traffic data was used.
- The graph is simplified (capped at {num_nodes} nodes) from the full Bengaluru network.
- Full DSTAGNN/Graph WaveNet requires PyTorch Geometric's message-passing
  operators; this stub uses simplified linear graph mixing.
- Production deployment would need streaming inference, which is not implemented.
"""

    report_path.write_text(report_text)

    with mlflow.start_run(run_name="m4_final_report"):
        mlflow.log_param("is_stub", True)
        log_markdown_report(report_path)

    logger.info("✅ M4 stub complete. Report: %s", report_path)


if __name__ == "__main__":
    run_m4()
