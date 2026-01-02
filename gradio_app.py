# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for Gradio web interface.

"""Gradio web interface for StyleGAN3 Latent Space Explorer."""

import os
import glob
import time
import numpy as np
import torch
from PIL import Image
import umap
import gradio as gr
import dnnlib
import legacy


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


def generate_latent_map(
    model_path,
    num_samples,
    thumb_size,
    layout_mode,
    spread,
    truncation_psi,
    canvas_size,
    seed,
    progress=gr.Progress()
):
    """Generate a 2D UMAP visualization of the latent space."""

    start_time = time.time()

    # Parse parameters
    thumb_size = int(thumb_size)
    canvas_size = int(canvas_size)
    num_samples = int(num_samples)
    overlap = (layout_mode == "overlap")

    # Set random seed
    if seed is None or seed == "":
        seed = np.random.randint(0, 2**31)
    else:
        seed = int(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (with cache)
    progress(0, desc="Loading model...")
    G = load_model(model_path)

    # Generate initial random W vector
    progress(0.1, desc="Generating latent vectors...")

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

    # Apply UMAP
    progress(0.2, desc="Applying UMAP...")
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
        if i % 10 == 0:
            progress(0.2 + 0.6 * (i / num_samples), desc=f"Generating image {i+1}/{num_samples}...")

        w_tensor = torch.from_numpy(w_flat).float().to(device)
        w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            img = G.synthesis(w_tensor, noise_mode='const')

        img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((thumb_size, thumb_size), Image.LANCZOS)
        images.append(img_pil)

    # Create canvas
    progress(0.85, desc="Creating canvas...")
    canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))

    margin = thumb_size
    usable_size = canvas_size - 2 * margin

    if overlap:
        # Exact UMAP positions
        indices = np.argsort(coords_normalized[:, 1])[::-1]

        for idx in indices:
            x = int(coords_normalized[idx, 0] * usable_size + margin - thumb_size // 2)
            y = int(coords_normalized[idx, 1] * usable_size + margin - thumb_size // 2)
            canvas.paste(images[idx], (x, y))
    else:
        # Grid layout
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

    # Save to exports folder
    progress(0.95, desc="Saving...")
    os.makedirs('exports', exist_ok=True)
    output_path = f'exports/latent_map_{seed}.png'
    canvas.save(output_path, 'PNG')

    elapsed_time = time.time() - start_time

    info = f"""**Generation Complete!**
- Seed: `{seed}`
- Samples: {num_samples}
- Time: {elapsed_time:.1f}s
- Layout: {layout_mode}
- Saved to: `{output_path}`"""

    progress(1.0, desc="Done!")

    return canvas, info


# Build Gradio interface
def create_interface():
    models = list_models()
    default_model = models[0] if models else None

    with gr.Blocks(title="StyleGAN3 Latent Space Explorer") as demo:
        gr.Markdown("# StyleGAN3 Latent Space Explorer")
        gr.Markdown("Generate 2D UMAP visualizations of the latent space.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Parameters")

                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label="Model",
                    info="Select a .pkl model from models/ folder"
                )

                num_samples_slider = gr.Slider(
                    minimum=25,
                    maximum=1600,
                    value=400,
                    step=25,
                    label="Number of Samples",
                    info="More samples = longer generation time"
                )

                thumb_size_radio = gr.Radio(
                    choices=["32", "64"],
                    value="64",
                    label="Thumbnail Size",
                    info="Size of each image in pixels"
                )

                layout_radio = gr.Radio(
                    choices=["overlap", "grid"],
                    value="overlap",
                    label="Layout Mode",
                    info="Overlap: exact UMAP positions (better proximity). Grid: no overlap."
                )

                spread_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Spread",
                    info="Variation around initial point (std dev in W space)"
                )

                truncation_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Truncation Psi",
                    info="Lower = more typical, higher = more diverse"
                )

                canvas_slider = gr.Slider(
                    minimum=1000,
                    maximum=5000,
                    value=3000,
                    step=500,
                    label="Canvas Size",
                    info="Output image size in pixels"
                )

                seed_input = gr.Textbox(
                    value="",
                    label="Seed (optional)",
                    info="Leave empty for random seed"
                )

                generate_btn = gr.Button("Generate Map", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                output_image = gr.Image(
                    label="Latent Space Map",
                    type="pil"
                )

                output_info = gr.Markdown(label="Info")

        # Connect button to function
        generate_btn.click(
            fn=generate_latent_map,
            inputs=[
                model_dropdown,
                num_samples_slider,
                thumb_size_radio,
                layout_radio,
                spread_slider,
                truncation_slider,
                canvas_slider,
                seed_input
            ],
            outputs=[output_image, output_info]
        )

        gr.Markdown("---")
        gr.Markdown("*Tip: Use 'overlap' mode to see true UMAP proximity. Use 'grid' to avoid overlapping images.*")

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.launch()
