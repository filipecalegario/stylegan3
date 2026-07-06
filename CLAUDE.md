# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Two layers coexist here:

1. **Upstream StyleGAN3** — NVIDIA's official Alias-Free GAN reference implementation (`train.py`, `gen_images.py`, `gen_video.py`, `visualizer.py`, `calc_metrics.py`, `avg_spectra.py`, `dataset_tool.py`, plus `training/`, `metrics/`, `torch_utils/`, `dnnlib/`, `gui_utils/`, `viz/`). Treated by NVIDIA as a one-time code drop — see `README.md` for the full upstream workflow (training, dataset prep, metrics, spectral analysis).
2. **Latent Space Explorer** — a custom web app added on top of StyleGAN3 for *interactive latent-space exploration*. This is where local development happens. The rest of this file focuses on it.

## Latent Space Explorer architecture

A FastAPI backend serves generation over REST + WebSocket; a React (Vite) frontend consumes it. Both share one convention: **models are `.pkl` files in `models/`, exports are written under `exports/`.**

### Backend (Python, repo root)

- `api_server.py` — FastAPI app (the main entry point). Defines all REST endpoints and two WebSocket endpoints (`/ws/generate` for the UMAP latent map, `/ws/interpolation` for video). Owns model caching and the UMAP latent-map generation. Runs on **port 8000**; CORS is hardcoded to allow the Vite dev server at `localhost:5173`.
- `stylegan_utils.py` — the shared generation primitive. **`generate_image_from_w(w_vector, model_path, size)`** turns a raw 512-float W vector into a PIL image and is used by every feature. Also holds the module-level `_model_cache`. When touching generation, prefer extending this rather than duplicating the tensor plumbing.
- `genetic_engine.py` — `GeneticEngine` / `Individual`: interactive genetic algorithm over W vectors (crossover, mutation, elitism, roulette/tournament selection). Stateful — `api_server` holds a single global `_genetic_engine` instance mutated across `/api/genetic/*` calls.
- `interpolation_engine.py` — `InterpolationEngine`: builds a scipy interpolation function through a sequence of W vectors and renders frames to an mp4/gif via **ffmpeg (`subprocess`)**. Requires ffmpeg on PATH.
- `gradio_app.py` — a standalone **alternative** Gradio UI for the latent map. Independent of the React frontend; do not assume changes to one affect the other.
- `export_latent_map.py` — CLI variant of the latent-map export.

Key domain fact: features operate on **W-space** (the 512-dim intermediate latent, `G.mapping` output), not Z-space. A single 512-vector is broadcast to all `G.num_ws` layers before `G.synthesis`. Images are rendered at the model's native `img_resolution` then resized to the requested `image_size`.

### Frontend (`frontend/`, React + Vite)

- One page per backend feature, wired in `src/App.jsx`:
  - `/` → `LatentExplorer` (UMAP map, WebSocket)
  - `/genetic` → `GeneticEvolution`
  - `/interpolation` → `LatentInterpolation` (WebSocket)
  - `/w-editor` → `WVectorEditor` (direct per-dimension W editing, with a sequencer)
- Each page has a matching hook in `src/hooks/` (`useGeneticAlgorithm`, `useInterpolation`, `useWVectorEditor`, `useWebSocket`) that encapsulates all backend calls. Put API interaction in the hook, not the component.
- Feature components are grouped under `src/components/<feature>/`.

## Common commands

Full dev stack — one command, from `frontend/`:

```bash
npm run dev       # starts BOTH: FastAPI backend (via `conda run -n stylegan3`) + Vite dev server
npm run dev:api   # backend only (port 8000, interactive docs at /docs)
npm run dev:web   # frontend only (port 5173)
npm run build
npm run preview
```

The `dev:api` script uses `conda run --no-capture-output -n stylegan3`, so it works without activating the env first — but it assumes a conda env named `stylegan3` exists.

Backend alternatives (from repo root, with the `stylegan3` env active):

```bash
python api_server.py    # FastAPI backend directly
python gradio_app.py    # Alternative Gradio UI instead of the React app
```

GA logic tests (no GPU/model needed — stubs the generator):

```bash
python test_genetic_engine.py
```

Note: the Explorer's extra Python deps (`fastapi`, `uvicorn`, `umap-learn`, `scipy`, `gradio`, `pydantic`) are **not** in `environment.yml` — install them alongside the base StyleGAN3 environment.

## Gotchas

- The custom PyTorch ops in `torch_utils/ops/` compile on the fly via NVCC on first run (needs CUDA toolkit + a host compiler; Visual Studio on Windows). The first generation call after startup is slow because of this and model loading.
- Networks load through `legacy.load_network_pkl(...)['G_ema']` and are cached per-process. Backend runs on CUDA when available, else CPU.
- The genetic engine keeps global mutable state across requests — `/api/genetic/init` must be called before evolve/config/export.
- Ports (8000 backend, 5173 frontend) and the `models/` + `exports/` paths are assumed in multiple places; changing one means changing the CORS list, the hooks, and the file-serving endpoints together.
