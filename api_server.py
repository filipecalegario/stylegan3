# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for FastAPI backend with WebSocket support.

"""FastAPI backend for StyleGAN3 Latent Space Explorer."""

import os
import glob
import time
import json
import asyncio
import numpy as np
import torch
from PIL import Image
import umap
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import dnnlib
import legacy
from genetic_engine import GeneticEngine
from pydantic import BaseModel
from typing import Dict, List, Optional


# Pydantic models for genetic algorithm endpoints
class GeneticInitRequest(BaseModel):
    model: str
    population_size: int = 9
    seed: Optional[int] = None
    image_size: int = 256


class GeneticEvolveRequest(BaseModel):
    fitness: Dict[str, float]


class GeneticConfigRequest(BaseModel):
    crossover_enabled: Optional[bool] = None
    crossover_method: Optional[str] = None
    mutation_enabled: Optional[bool] = None
    mutation_rate: Optional[float] = None
    mutation_strength: Optional[float] = None
    elitism_count: Optional[int] = None
    selection_method: Optional[str] = None
    image_size: Optional[int] = None


class GeneticExportRequest(BaseModel):
    individual_id: str


app = FastAPI(title="StyleGAN3 Latent Space Explorer API")

# Global genetic engine instance
_genetic_engine: Optional[GeneticEngine] = None

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for model
_cached_model = None
_cached_model_path = None


def list_models():
    """List available .pkl models in models/ folder."""
    models = glob.glob('models/*.pkl')
    return [os.path.basename(m) for m in models]


def load_model(model_path):
    """Load model with caching to avoid reloading."""
    global _cached_model, _cached_model_path

    full_path = os.path.join('models', model_path)

    if _cached_model_path != full_path:
        print(f'Loading model: {full_path}...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with dnnlib.util.open_url(full_path) as f:
            _cached_model = legacy.load_network_pkl(f)['G_ema'].to(device)

        _cached_model_path = full_path
        print(f'Model loaded. Resolution: {_cached_model.img_resolution}')

    return _cached_model


async def generate_latent_map(websocket: WebSocket, params: dict):
    """Generate a 2D UMAP visualization of the latent space with progress updates."""

    start_time = time.time()

    # Parse parameters
    model_path = params.get('model')
    num_samples = int(params.get('num_samples', 400))
    thumb_size = int(params.get('thumb_size', 64))
    layout_mode = params.get('layout_mode', 'overlap')
    spread = float(params.get('spread', 0.5))
    truncation_psi = float(params.get('truncation_psi', 0.7))
    canvas_size = int(params.get('canvas_size', 3000))
    seed = params.get('seed')

    overlap = (layout_mode == "overlap")

    # Set random seed
    if seed is None or seed == "" or seed == 0:
        seed = np.random.randint(0, 2**31)
    else:
        seed = int(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Progress: Loading model
    await websocket.send_json({
        "type": "progress",
        "value": 0.05,
        "message": "Loading model..."
    })

    G = load_model(model_path)

    # Progress: Generating latent vectors
    await websocket.send_json({
        "type": "progress",
        "value": 0.1,
        "message": "Generating latent vectors..."
    })

    z_init = torch.randn(1, G.z_dim, device=device)
    c_init = torch.zeros(1, G.c_dim, device=device) if G.c_dim > 0 else None
    w_init = G.mapping(z_init, c_init, truncation_psi=truncation_psi)

    # Generate nearby W vectors
    w_samples = []
    w_init_flat = w_init[0, 0].cpu().numpy()

    for i in range(num_samples):
        perturbation = np.random.randn(G.w_dim) * spread
        w_new = w_init_flat + perturbation
        w_samples.append(w_new)

    w_samples = np.array(w_samples)

    # Progress: UMAP
    await websocket.send_json({
        "type": "progress",
        "value": 0.15,
        "message": "Applying UMAP dimensionality reduction..."
    })

    reducer = umap.UMAP(
        n_neighbors=min(15, num_samples - 1),
        min_dist=0.1,
        n_components=2,
        random_state=seed,
    )
    coords_2d = reducer.fit_transform(w_samples)

    # Normalize coordinates
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_normalized = (coords_2d - coords_min) / (coords_max - coords_min + 1e-8)

    # Generate images
    images = []
    for i, w_flat in enumerate(w_samples):
        progress_value = 0.2 + 0.65 * (i / num_samples)

        if i % 20 == 0 or i == num_samples - 1:
            await websocket.send_json({
                "type": "progress",
                "value": progress_value,
                "message": f"Generating image {i+1}/{num_samples}..."
            })

        w_tensor = torch.from_numpy(w_flat).float().to(device)
        w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            img = G.synthesis(w_tensor, noise_mode='const')

        img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((thumb_size, thumb_size), Image.LANCZOS)
        images.append(img_pil)

    # Progress: Creating canvas
    await websocket.send_json({
        "type": "progress",
        "value": 0.9,
        "message": "Creating canvas..."
    })

    canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))

    margin = thumb_size
    usable_size = canvas_size - 2 * margin

    if overlap:
        indices = np.argsort(coords_normalized[:, 1])[::-1]

        for idx in indices:
            x = int(coords_normalized[idx, 0] * usable_size + margin - thumb_size // 2)
            y = int(coords_normalized[idx, 1] * usable_size + margin - thumb_size // 2)
            canvas.paste(images[idx], (x, y))
    else:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        cell_size = usable_size // grid_size

        grid_coords = (coords_normalized * (grid_size - 1)).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)

        occupied = set()
        placements = []

        for i in range(num_samples):
            gx, gy = grid_coords[i]

            if (gx, gy) in occupied:
                found = False
                for radius in range(1, grid_size):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in occupied:
                                gx, gy = nx, ny
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            occupied.add((gx, gy))
            placements.append((i, gx, gy))

        for i, gx, gy in placements:
            x = margin + gx * cell_size + (cell_size - thumb_size) // 2
            y = margin + gy * cell_size + (cell_size - thumb_size) // 2
            canvas.paste(images[i], (x, y))

    # Progress: Saving
    await websocket.send_json({
        "type": "progress",
        "value": 0.95,
        "message": "Saving image..."
    })

    os.makedirs('exports', exist_ok=True)
    filename = f'latent_map_{seed}.png'
    output_path = os.path.join('exports', filename)
    canvas.save(output_path, 'PNG')

    elapsed_time = time.time() - start_time

    # Complete
    await websocket.send_json({
        "type": "complete",
        "image_url": f"/api/exports/{filename}",
        "seed": seed,
        "time": round(elapsed_time, 1),
        "num_samples": num_samples,
        "layout_mode": layout_mode
    })


# REST Endpoints

@app.get("/api/models")
async def get_models():
    """List available models."""
    models = list_models()
    return {"models": models}


@app.get("/api/exports/{filename}")
async def get_export(filename: str):
    """Serve exported images."""
    filepath = os.path.join('exports', filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return {"error": "File not found"}, 404


# WebSocket Endpoint

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for generation with progress updates."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("action") == "generate":
                params = message.get("params", {})
                try:
                    await generate_latent_map(websocket, params)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        print("WebSocket disconnected")


# ============ Genetic Algorithm Endpoints ============

@app.post("/api/genetic/init")
async def genetic_init(request: GeneticInitRequest):
    """Initialize a new population for the genetic algorithm."""
    global _genetic_engine

    model_path = os.path.join('models', request.model)
    _genetic_engine = GeneticEngine(model_path)

    # Set image size before generating
    _genetic_engine.update_config({'image_size': request.image_size})

    population = _genetic_engine.initialize_population(
        size=request.population_size,
        seed=request.seed
    )

    return _genetic_engine.get_population_state()


@app.get("/api/genetic/population")
async def genetic_population():
    """Get the current population state."""
    global _genetic_engine

    if _genetic_engine is None:
        return {"error": "Genetic engine not initialized"}, 400

    return _genetic_engine.get_population_state()


@app.post("/api/genetic/evolve")
async def genetic_evolve(request: GeneticEvolveRequest):
    """Evolve to the next generation based on fitness values."""
    global _genetic_engine

    if _genetic_engine is None:
        return {"error": "Genetic engine not initialized"}, 400

    _genetic_engine.evolve(request.fitness)

    return _genetic_engine.get_population_state()


@app.post("/api/genetic/config")
async def genetic_config(request: GeneticConfigRequest):
    """Update genetic algorithm configuration."""
    global _genetic_engine

    if _genetic_engine is None:
        return {"error": "Genetic engine not initialized"}, 400

    config_update = {k: v for k, v in request.model_dump().items() if v is not None}
    _genetic_engine.update_config(config_update)

    return {"config": _genetic_engine.config}


@app.post("/api/genetic/export")
async def genetic_export(request: GeneticExportRequest):
    """Export an individual's data (W vector and image)."""
    global _genetic_engine

    if _genetic_engine is None:
        return {"error": "Genetic engine not initialized"}, 400

    data = _genetic_engine.export_individual(request.individual_id)

    if data is None:
        return {"error": "Individual not found"}, 404

    return data


@app.get("/api/genetic/image/{image_id}.png")
async def genetic_image(image_id: str):
    """Serve genetic algorithm images."""
    filepath = os.path.join('exports', 'genetic', f'{image_id}.png')
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return {"error": "Image not found"}, 404


if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
