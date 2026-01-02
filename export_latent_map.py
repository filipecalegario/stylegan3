# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for latent space visualization.

"""Latent space UMAP exporter for StyleGAN3."""

import os
import sys
import click
import numpy as np
import torch
from PIL import Image
import umap
import dnnlib
import legacy


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-samples', type=int, default=400, help='Number of samples to generate')
@click.option('--thumb-size', type=click.Choice(['32', '64']), default='64', help='Thumbnail size (32 or 64)')
@click.option('--overlap', is_flag=True, default=False, help='Allow overlap (exact UMAP positions) vs grid layout')
@click.option('--spread', type=float, default=0.5, help='Spread around initial point in W space (std dev)')
@click.option('--seed', type=int, default=None, help='Random seed for reproducibility')
@click.option('--output', type=str, default='exports/latent_map.png', help='Output PNG filename')
@click.option('--canvas-size', type=int, default=3000, help='Canvas size in pixels')
@click.option('--trunc', 'truncation_psi', type=float, default=0.7, help='Truncation psi')
def main(
    network_pkl: str,
    num_samples: int,
    thumb_size: str,
    overlap: bool,
    spread: float,
    seed: int,
    output: str,
    canvas_size: int,
    truncation_psi: float,
):
    """Generate a 2D UMAP visualization of the latent space."""

    thumb_size = int(thumb_size)

    # Create output directory if needed
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Random seed: {seed}')
    print(f'Loading network from "{network_pkl}"...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    print(f'Network loaded. Resolution: {G.img_resolution}x{G.img_resolution}')
    print(f'W dimension: {G.w_dim}, Number of W layers: {G.num_ws}')

    # Generate initial random W vector
    print(f'Generating {num_samples} samples around initial point (spread={spread})...')

    # Get w_avg for truncation
    w_avg = G.mapping.w_avg

    # Generate initial z and map to w
    z_init = torch.randn(1, G.z_dim, device=device)
    c_init = torch.zeros(1, G.c_dim, device=device) if G.c_dim > 0 else None
    w_init = G.mapping(z_init, c_init, truncation_psi=truncation_psi)  # [1, num_ws, w_dim]

    # Generate nearby W vectors by adding perturbations
    w_samples = []
    w_init_flat = w_init[0, 0].cpu().numpy()  # Use first layer's w

    for i in range(num_samples):
        # Add gaussian noise to initial W
        perturbation = np.random.randn(G.w_dim) * spread
        w_new = w_init_flat + perturbation
        w_samples.append(w_new)

    w_samples = np.array(w_samples)  # [num_samples, w_dim]

    # Apply UMAP
    print('Applying UMAP dimensionality reduction...')
    reducer = umap.UMAP(
        n_neighbors=min(15, num_samples - 1),
        min_dist=0.1,
        n_components=2,
        random_state=seed,
    )
    coords_2d = reducer.fit_transform(w_samples)  # [num_samples, 2]

    # Normalize coordinates to [0, 1]
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_normalized = (coords_2d - coords_min) / (coords_max - coords_min + 1e-8)

    # Generate images
    print('Generating images...')
    images = []

    for i, w_flat in enumerate(w_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  Generating image {i + 1}/{num_samples}...')

        # Expand W to all layers
        w_tensor = torch.from_numpy(w_flat).float().to(device)
        w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)  # [1, num_ws, w_dim]

        # Generate image
        with torch.no_grad():
            img = G.synthesis(w_tensor, noise_mode='const')

        # Convert to PIL
        img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((thumb_size, thumb_size), Image.LANCZOS)
        images.append(img_pil)

    # Create canvas
    print(f'Creating {canvas_size}x{canvas_size} canvas...')
    canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))

    margin = thumb_size  # Margin to prevent images from being cut off at edges
    usable_size = canvas_size - 2 * margin

    if overlap:
        # Place images at exact UMAP positions
        print('Placing images at exact UMAP positions (overlap enabled)...')

        # Sort by y coordinate to draw bottom images first (so top images appear on top)
        indices = np.argsort(coords_normalized[:, 1])[::-1]

        for idx in indices:
            x = int(coords_normalized[idx, 0] * usable_size + margin - thumb_size // 2)
            y = int(coords_normalized[idx, 1] * usable_size + margin - thumb_size // 2)
            canvas.paste(images[idx], (x, y))
    else:
        # Grid layout based on UMAP positions
        print('Placing images in grid layout (no overlap)...')

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        cell_size = usable_size // grid_size

        # Assign each sample to a grid cell based on UMAP coordinates
        grid_coords = (coords_normalized * (grid_size - 1)).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)

        # Handle collisions by finding nearest empty cell
        occupied = set()
        placements = []

        for i in range(num_samples):
            gx, gy = grid_coords[i]

            # If cell is occupied, find nearest empty cell
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

        # Draw images
        for i, gx, gy in placements:
            x = margin + gx * cell_size + (cell_size - thumb_size) // 2
            y = margin + gy * cell_size + (cell_size - thumb_size) // 2
            canvas.paste(images[i], (x, y))

    # Save
    print(f'Saving to {output}...')
    canvas.save(output, 'PNG')
    print(f'Done! Output saved to: {output}')
    print(f'Seed used: {seed}')


if __name__ == '__main__':
    main()
