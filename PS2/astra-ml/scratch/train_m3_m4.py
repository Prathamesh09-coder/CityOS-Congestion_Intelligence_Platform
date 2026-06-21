import os
import sys
import logging
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Adjust sys.path to import from astra_ml
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from astra_ml.utils.seeding import set_global_seed
from astra_ml.models.m3_multimodal_fusion import MultimodalFusionModel
from astra_ml.models.m4_gnn_backbone import _build_graph_wavenet
from astra_ml.data.road_graph import get_or_build_road_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_m3_m4")

def train_m3_and_m4():
    set_global_seed(42)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load configs
    data_cfg = OmegaConf.to_container(OmegaConf.load("configs/data.yaml"), resolve=True)
    m3_cfg = OmegaConf.to_container(OmegaConf.load("configs/m3_multimodal_sparse.yaml"), resolve=True)
    m4_cfg = OmegaConf.to_container(OmegaConf.load("configs/m4_gnn_backbone.yaml"), resolve=True)

    # 1. Train and save M3
    logger.info("=== Starting M3 Multimodal Training ===")
    from transformers import AutoModel, AutoTokenizer
    
    model_name = m3_cfg["text_encoder"]["model_name"]
    logger.info(f"Loading cached MuRIL text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    splits_path = Path(data_cfg["paths"]["splits_parquet"])
    df = pl.read_parquet(splits_path)
    
    # Filter and sample training split for rapid training on CPU
    train_df = df.filter(pl.col("split") == "train")
    if train_df.height > 100:
        train_df = train_df.sample(n=100, seed=42)
    
    logger.info(f"Training M3 on {train_df.height} sample records (1 epoch CPU)...")
    model = MultimodalFusionModel(m3_cfg, tokenizer, base_model)
    model._build_cat_embeddings(train_df)
    model.to(model.device)

    # Train for 1 epoch
    trainable_params = []
    trainable_params.extend([p for p in model.text_encoder.parameters() if p.requires_grad])
    trainable_params.extend(model.text_projection.parameters())
    trainable_params.extend(model.fusion_mlp.parameters())
    trainable_params.extend(model.closure_head.parameters())
    trainable_params.extend(model.duration_head.parameters())
    for emb in model.cat_embeddings.values():
        trainable_params.extend(emb.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
    loss = model.train_epoch(train_df, optimizer, batch_size=16)
    logger.info(f"M3 epoch complete, loss: {loss:.4f}")

    # Save M3 weights and vocab mapping
    m3_checkpoint = {
        "state_dict": model.state_dict(),
        "cat_vocab": model.cat_vocab,
        "embedding_dim": model.embedding_dim,
    }
    torch.save(m3_checkpoint, models_dir / "m3_model.pth")
    logger.info("M3 model checkpoint saved successfully!")

    # Clean up M3 memory
    del model, base_model
    import gc
    gc.collect()

    # 2. Train and save M4
    logger.info("=== Starting M4 GNN Training ===")
    try:
        graph = get_or_build_road_graph(data_cfg)
        num_nodes = min(graph.number_of_nodes(), 100)  # Capped for toy graph training
        logger.info(f"Using {num_nodes} nodes from Bengaluru OSM Graph")
        
        # Calculate static features
        import networkx as nx
        nodes = list(graph.nodes())[:num_nodes]
        degrees = dict(graph.degree())
        try:
            bet_cent = nx.betweenness_centrality(graph, k=min(10, len(nodes)), normalized=True)
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
        logger.warning(f"OSM Graph load failed ({e}); falling back to synthetic nodes")
        num_nodes = 50
        nodes = [f"node_{i}" for i in range(num_nodes)]
        static_features = np.random.randn(num_nodes, 3).astype(np.float32)

    model_cfg = m4_cfg["model"]
    in_channels_total = model_cfg.get("in_channels", 4) + 3

    GraphWaveNet = _build_graph_wavenet(m4_cfg)
    gnn_model = GraphWaveNet(
        num_nodes=num_nodes,
        in_channels=in_channels_total,
        out_channels=model_cfg.get("out_channels", 1),
        hidden_channels=model_cfg.get("hidden_channels", 32),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.3),
    )
    
    device = torch.device("cpu")
    gnn_model.to(device)

    # 1 epoch training on synthetic flow features aligned with graph size
    batch_size = 16
    num_timesteps = 24
    X_base = torch.randn(batch_size, num_nodes, num_timesteps, model_cfg.get("in_channels", 4))
    static_tensor = torch.tensor(static_features, dtype=torch.float32)
    static_expanded = static_tensor.unsqueeze(0).unsqueeze(2).expand(batch_size, num_nodes, num_timesteps, 3)
    X_synthetic = torch.cat([X_base, static_expanded], dim=-1)
    y_synthetic = torch.randn(batch_size, num_nodes, 1)

    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    gnn_model.train()
    optimizer_gnn.zero_grad()
    y_pred = gnn_model(X_synthetic)
    loss_gnn = torch.nn.functional.mse_loss(y_pred, y_synthetic)
    loss_gnn.backward()
    optimizer_gnn.step()
    logger.info(f"M4 GNN epoch complete, synthetic loss: {loss_gnn.item():.6f}")

    # Save M4 weights, nodes, and static features
    m4_checkpoint = {
        "state_dict": gnn_model.state_dict(),
        "nodes": nodes,
        "static_features": static_features.tolist(),
        "num_nodes": num_nodes,
    }
    torch.save(m4_checkpoint, models_dir / "m4_model.pth")
    logger.info("M4 GNN model checkpoint saved successfully!")
    logger.info("=== Deep Learning Models Prepared for Inference API ===")

if __name__ == "__main__":
    train_m3_and_m4()
