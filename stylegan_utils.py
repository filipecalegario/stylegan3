# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Utility functions for StyleGAN3 image generation.

"""Shared utility functions for StyleGAN3 image generation."""

import torch
import numpy as np
from PIL import Image
import dnnlib
import legacy

# Global model cache to avoid reloading
_model_cache = {}


def load_model(model_path: str, device=None):
    """
    Load StyleGAN3 model with caching.

    Args:
        model_path: Path to the .pkl model file
        device: PyTorch device (defaults to CUDA if available)

    Returns:
        Tuple of (Generator model, device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path not in _model_cache:
        print(f'Loading model: {model_path}...')
        with dnnlib.util.open_url(model_path) as f:
            _model_cache[model_path] = legacy.load_network_pkl(f)['G_ema'].to(device)
        print(f'Model loaded successfully.')

    return _model_cache[model_path], device


def generate_image_from_w(
    w_vector: np.ndarray,
    model_path: str,
    size: int = 512
) -> Image.Image:
    """
    Generate image directly from a raw W vector (512 floats).

    This is the main utility function used by:
    - W Vector Editor
    - Interpolation Engine
    - Genetic Algorithm

    Args:
        w_vector: NumPy array of shape (512,) with W latent values
        model_path: Path to the .pkl model file
        size: Output image size (will resize if different from model resolution)

    Returns:
        PIL Image in RGB format
    """
    G, device = load_model(model_path)

    # Convert to tensor and expand to all layers: (1, num_ws, w_dim)
    w_tensor = torch.from_numpy(w_vector).float().to(device)
    w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)

    # Generate image
    with torch.no_grad():
        img = G.synthesis(w_tensor, noise_mode='const')

    # Convert to PIL Image
    img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    img_pil = Image.fromarray(img, 'RGB')

    # Resize if needed
    if size != G.img_resolution:
        img_pil = img_pil.resize((size, size), Image.LANCZOS)

    return img_pil


def generate_random_w(model_path: str, truncation_psi: float = 0.7, seed: int = None) -> np.ndarray:
    """
    Generate a random W vector using the model's mapping network.

    Args:
        model_path: Path to the .pkl model file
        truncation_psi: Truncation value (0-1, lower = more average faces)
        seed: Random seed for reproducibility

    Returns:
        NumPy array of shape (512,) with W latent values
    """
    G, device = load_model(model_path)

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random Z
    z = torch.randn(1, G.z_dim, device=device)

    # Map to W
    c = torch.zeros(1, G.c_dim, device=device) if G.c_dim > 0 else None
    with torch.no_grad():
        w = G.mapping(z, c, truncation_psi=truncation_psi)

    # Return first layer's W (all layers have same W initially)
    return w[0, 0].cpu().numpy()


def clear_model_cache():
    """Clear the model cache to free GPU memory."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()
