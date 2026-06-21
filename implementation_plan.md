# Hackathon Winner Upgrade: Deep Learning, Streaming, Preprocessing & Cloud Deployment

This implementation plan details the strategy to upgrade the CityOS and ASTRA ML application for a hackathon-winning production release.

## User Review Required

> [!IMPORTANT]
> 1. **Model Training:** We will run a script `scratch/train_m3_m4.py` to train both M3 (Multimodal PEFT/LoRA) and M4 (Graph WaveNet) for 2 epochs on the CPU and save their weights locally to `models/m3_model.pth` and `models/m4_model.pth`. The FastAPI backend will load these weights at startup to replace the simulators with actual live PyTorch deep learning inference.
> 2. **Dockerization:** We will add a production `Dockerfile` to the `PS2/astra-ml` directory to make the FastAPI backend ready for cloud hosting (Render, Railway, or AWS).
> 3. **Kafka/Real-Time APIs Simulation:** We will build in-memory streaming databases inside the FastAPI app and expose `POST /api/v1/stream/ingest` and `POST /api/v1/stream/traffic` endpoints to simulate real-time sensor/stream updates.
> 4. **Frontend Wiring:** Once the M3 and M4 models are serving live predictions, we will configure the React frontend to fetch directly from these endpoints (removing mock variables). Specifically:
>    * `diversion.tsx` will query `predictTraffic` for OSM Graph GNN telemetry metrics.
>    * `demo.tsx` and `forecast.tsx` will call `predictMultimodal` for dynamic zero-shot sentiment/event analyses.

## Open Questions

> [!NOTE]
> * **Hardware Accelerator:** The PyTorch model training and inference will run on CPU since the backend server is running on a mac environment, ensuring compatibility across all dev environments.

---

## Proposed Changes

### [Component: Model Training Script]
Summary: Create a Python script to train the PEFT/LoRA Multimodal Fusion model (M3) and the Graph WaveNet model (M4) on the available splits, exporting their final weights to the `models/` folder.

#### [NEW] [train_m3_m4.py](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/PS2/astra-ml/scratch/train_m3_m4.py)
* Load data splits `events_splits.parquet`.
* **M3 Multimodal Training:**
  * Initialize the `MultimodalFusionModel` with a dynamic LoRA config on top of the cached `google/muril-base-cased`.
  * Fit category embeddings based on training data.
  * Train for 2 epochs using multi-task BCE and MSE loss and save state dict + vocab configs to `models/m3_model.pth`.
* **M4 GNN Training:**
  * Load the OSM road graph and extract node features (degree, betweenness centrality, intersection).
  * Build the `GraphWaveNet` model.
  * Train for 2 epochs on synthetic-aligned spatio-temporal nodes and save weights + node mapping configurations to `models/m4_model.pth`.

---

### [Component: FastAPI Backend Upgrades]
Summary: Modify `main.py` to load the real model weights, perform live tensor inferences, simulate real-time streaming data ingestion, and implement robust out-of-vocabulary category mapping.

#### [MODIFY] [main.py](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/PS2/astra-ml/src/astra_ml/api/main.py)
* **Out-of-Vocabulary (OOV) Preprocessing:**
  * Write a robust wrapper function `safe_encode_category(le, val_str)` that handles unseen category strings by returning the code for `"__MISSING__"` or fallback to `0` rather than throwing `ValueError`.
* **M3 & M4 Weight Loading:**
  * Load M3 model architecture, category embedding configurations, and state dict from `models/m3_model.pth` on startup.
  * Load M4 GNN weights and node ID lookup mapping from `models/m4_model.pth` on startup.
* **M3 Live Inference:**
  * Inside `/api/v1/predict/multimodal`, dynamically tokenize the input description/comment, embed structured features using trained embeddings, and pass them through the PEFT/LoRA Multimodal Fusion model to retrieve live outputs.
* **M4 Live Inference:**
  * Inside `/api/v1/predict/traffic`, pull live spatial attributes from the road graph, run the Graph WaveNet model, and return actual GNN-predicted speeds and delays.
* **Streaming Data Simulation:**
  * Initialize in-memory logs `LIVE_EVENTS_STREAM` and `LIVE_TRAFFIC_TELEMETRY`.
  * Expose `POST /api/v1/stream/ingest` to receive mock Kafka/real-time event streaming payloads.
  * Expose `POST /api/v1/stream/traffic` to receive real-time speed/congestion updates from external sensors/TomTom.
  * Integrate these live streams directly into the prediction handlers, overriding historical averages with current telemetry when available.

---

### [Component: React Frontend Updates]
Summary: Wire the React dashboard pages to the live deep-learning inference endpoints.

#### [MODIFY] [api.ts](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/frontend/src/lib/api.ts)
* Update `predictMultimodal` and `predictTraffic` calls to point directly to `http://localhost:8000/api/v1/predict/multimodal` and `/api/v1/predict/traffic` respectively, and ensure they fall back gracefully if the backend is down.

#### [MODIFY] [use-predict.ts](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/frontend/src/hooks/use-predict.ts)
* Create React Query hooks: `useMultimodalPrediction` and `useTrafficPrediction`.

#### [MODIFY] [diversion.tsx](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/frontend/src/routes/diversion.tsx)
* Wire the smart diversion engine page to the `useTrafficPrediction` hook. Select a junction on the map, query the GNN backend, and populate the telemetry cards (Predicted Speed, Predicted Flow, Average Delay, Congestion Index) using live model weights.

#### [MODIFY] [demo.tsx](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/frontend/src/routes/demo.tsx)
* Wire the storytelling demo narrative to call the GNN traffic and multimodal zero-shot prediction hooks, visualizing real-time deep learning inference stats rather than mock variables.

---

### [Component: Containerization & Cloud Deployment]
Summary: Add Docker files and configuration assets to enable immediate cloud deployment for the frontend and backend.

#### [NEW] [Dockerfile](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/PS2/astra-ml/Dockerfile)
* Containerize the FastAPI backend using `python:3.12-slim` base image.
* Install necessary dependencies (`torch` CPU version, `transformers`, `peft`, `scikit-learn`, `fastapi`, `uvicorn`, `polars`).
* Copy the trained model assets in `/models` and run Uvicorn.

#### [NEW] [docker-compose.yml](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/docker-compose.yml)
* Orchestrate the backend and frontend containers for local cluster simulation.

#### [NEW] [deployment_guide.md](file:///Users/prathameshnawale/Desktop/Flipkart%20Grid%202.0/deployment_guide.md)
* Create a detailed instructions manual for judges or developers showing how to deploy the Vite frontend to Vercel/Netlify and the Python backend to Render/Railway.

---

## Verification Plan

### Automated Tests
- Run `scratch/train_m3_m4.py` to train and generate the model weights.
- Start the upgraded FastAPI backend and run automated tests using a script `/scratch/test_production_api.py` validating:
  * Model predictions for M3 (live neural network logits).
  * Model predictions for M4 (live graph convolutions).
  * Safe error mapping of unknown categorical categories.
  * Ingestion of live Kafka streaming mock inputs.

### Manual Verification
- Open the React frontend, visit all pages, and verify that the UI renders live forecasts correctly.
