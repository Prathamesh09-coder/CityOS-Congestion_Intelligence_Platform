# End-to-End Model Integration & Dynamic Generation Plan

This plan addresses the identified flaws in the CityOS platform by generating the missing M3 model, implementing dynamic values for the M4 model, and connecting the frontend strictly to the real models without silent fallbacks.

## User Review Required

> [!WARNING]
> **Strict Backend Dependency**: Once this plan is implemented, the React frontend will no longer silently mock values when the FastAPI backend is down. If the backend models fail or the server is off, the frontend will display an error. This is a crucial step for making the app truly "end-to-end", but it means you *must* have the backend running for the UI to work.
> 
> **M4 Routing Graph**: Generating dynamic diversions requires `networkx` to calculate shortest paths over the OSM road graph. I will implement a dynamic routing algorithm that takes the current junction and calculates a route to avoid the blocked node.

## Proposed Changes

### Backend Training (M3 Model)

#### [MODIFY] `PS2/astra-ml/src/astra_ml/models/m3_multimodal_fusion.py`
- The current script only evaluates the model using Leave-One-Out Cross-Validation (LOO-CV) but forgets to save the final weights.
- **Change**: Add a new function `train_and_save_m3()` that trains a final model on the entire dataset.
- **Change**: Save `state_dict`, `cat_vocab`, and `embedding_dim` to `models/m3_model.pth` so the FastAPI backend can load it.

---

### Backend API (M3 & M4 Generation)

#### [MODIFY] `PS2/astra-ml/src/astra_ml/api/main.py`
- **M4 Traffic Prediction**: Remove the `rng.uniform()` random number logic. Ensure the `predicted_speed` and `average_delay_minutes` are directly derived from the `m4_model` GNN output tensor. 
- **M4 Dynamic Routing**: Remove the hardcoded `PREDEFINED_DIVERSIONS` dictionary. Integrate `networkx` to run a shortest-path algorithm (A* or Dijkstra) on the loaded `road_graph` to dynamically generate a diversion route bypassing the congested node.
- **M3 Multimodal**: Remove the mathematical simulation (`np.sin` projection). Ensure it requires `m3_model` to run, otherwise raising an HTTP 503 error.

---

### Frontend Sync

#### [MODIFY] `frontend/src/lib/api.ts`
- **Change**: Completely remove the `fallbackGenerator` and the simulated deterministic backup data.
- **Change**: The `callApi` function will now throw a clear error if the API request fails, ensuring the frontend is 100% synced to the live ML models and never fakes data.
- The `use-predict.ts` hooks are already wired to handle errors and will show a `toast.error` popup if the backend is unreachable.

## Verification Plan

### Automated/Manual Verification
1. Run `make train-m3` in the `astra-ml` directory to train and verify that `models/m3_model.pth` is created successfully.
2. Start the FastAPI backend and verify that it successfully loads all models, including M3.
3. Use the frontend UI to predict a traffic delay and visualize the diversion route, confirming that it generates a dynamic `[lat, lng]` path array calculated via `networkx`.
4. Turn off the FastAPI backend and verify the frontend throws an explicit error rather than displaying simulated values.
