# M4 — Graph Spatio-Temporal Backbone

> ⚠️ **STUB: Architecture Validation Only**
>
> This model was trained on **SYNTHETIC** data. The metrics below do NOT represent
> real traffic prediction performance. They only validate that the architecture
> compiles, trains, and converges on toy data.

## Architecture

- **Type**: Graph WaveNet (simplified)
- **Nodes**: 200 (from OSM road graph, capped)
- **Input channels**: 7 (synthetic: speed, flow, hour_sin, hour_cos + static graph: degree, betweenness centrality, is_intersection)
- **Hidden channels**: 32
- **Layers**: 3
- **Total parameters**: 30,497

## Training (Synthetic Data)

- **Epochs**: 10
- **Final loss**: 1.013338
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
- The graph is simplified (capped at 200 nodes) from the full Bengaluru network.
- Full DSTAGNN/Graph WaveNet requires PyTorch Geometric's message-passing
  operators; this stub uses simplified linear graph mixing.
- Production deployment would need streaming inference, which is not implemented.
