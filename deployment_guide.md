# CityOS & ASTRA ML Production Deployment Guide

This guide details the step-by-step instructions to build, run, and deploy the **CityOS** React dashboard and the **ASTRA ML** FastAPI backend to production cloud hosting platforms or run them locally using Docker.

---

## Architecture Overview

* **Frontend**: React SPA built with Vite, TanStack Router, Tailwind CSS v4, and Radix UI. Serves static HTML/JS assets.
* **Backend**: FastAPI Python API serving live PyTorch models:
  * **M1**: Road Closure Classifier (LightGBM)
  * **M2**: Duration Estimator (CatBoost / GBST)
  * **M3**: Multimodal Zero-Shot Predictor (MuRIL + LoRA PEFT)
  * **M4**: Spatio-Temporal Graph WaveNet (Graph Convolutional network over OSM topology)

---

## Option 1: Local Deployment with Docker Compose

To build and run the entire ecosystem locally inside isolated Docker containers:

### Prerequisites
* Install [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose.

### Steps
1. Navigate to the project root directory.
2. Run the following command to build the images and launch the containers:
   ```bash
   docker compose up --build
   ```
3. Once running, access:
   * **React Dashboard**: [http://localhost:80](http://localhost:80)
   * **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Option 2: Cloud Deployment

### 1. Backend: Deploying FastAPI to Render, Railway, or AWS ECS

Since the backend loads deep learning weights (`m3_model.pth` and `m4_model.pth`), it requires an environment with **at least 2GB of RAM** (4GB recommended).

#### Option A: Render (Fastest Container Deployment)
1. Sign in to [Render](https://render.com/).
2. Create a new **Web Service** and connect your GitHub repository.
3. Configure the following service parameters:
   * **Root Directory**: `PS2/astra-ml`
   * **Runtime**: `Docker`
   * **Plan**: Starter or above (ensure at least 2GB RAM for model weight allocations)
4. Add the following **Environment Variables**:
   * `HOST`: `0.0.0.0`
   * `PORT`: `10000` (Render binds automatically to this port)
   * `KMP_DUPLICATE_LIB_OK`: `TRUE`
5. Render will automatically detect the `Dockerfile` inside `PS2/astra-ml/`, build the CPU-optimized PyTorch environment, and deploy your live model endpoints.

#### Option B: Railway
1. Sign in to [Railway](https://railway.app/).
2. Create a **New Project** and connect your repository.
3. Select the `PS2/astra-ml` subdirectory to deploy.
4. Railway will automatically build and deploy via the `Dockerfile`.
5. Under service settings, add Environment Variables:
   * `KMP_DUPLICATE_LIB_OK`: `TRUE`

---

### 2. Frontend: Deploying React to Vercel or Netlify

Since this frontend uses TanStack Start with a Nitro engine, it can be deployed either as a Node server or compiled directly to Vercel/Netlify serverless functions.

#### Option A: Vercel (Auto-detected Serverless Functions)
1. Sign in to [Vercel](https://vercel.com/).
2. Select **Import Project** and connect your repository.
3. Configure the project settings:
   * **Framework Preset**: Leave as `Other` or `Vite` (Vercel automatically detects TanStack Start/Nitro).
   * **Root Directory**: `frontend`
   * **Build Command**: `npm run build`
   * **Output Directory**: Vercel automatically reads the generated `.vercel/output` directory, so leave this configuration blank.
4. Under **Environment Variables**, define your API base url:
   * Set `VITE_API_BASE_URL` to your public backend URL (e.g., `https://astra-ml-backend.onrender.com/api/v1`).
5. Click **Deploy**. Vercel will automatically host the application with serverless routing and SSL.

---

## Environment Configuration Summary

| Variable | Recommended Value | Description |
| :--- | :--- | :--- |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Prevents macOS/Linux OpenMP compilation conflicts during PyTorch imports. |
| `HOST` | `0.0.0.0` | Binds the API server to all interfaces inside Docker. |
| `PORT` | `8000` / `3000` | Port for the FastAPI server (8000) or React server (3000) to listen on. |
| `VITE_API_BASE_URL` | *(Public backend URI)* | The public FastAPI backend base endpoint URL for the frontend to connect to (e.g. `https://my-backend.onrender.com/api/v1`). |
| `NITRO_PRESET` | `node-server` / `vercel` | Specifies the build output preset target for the Nitro server engine. |

