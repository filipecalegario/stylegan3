# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for Latent Space Interpolation Video Generation.

"""Interpolation Engine for StyleGAN3 Latent Space Videos."""

import os
import uuid
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import scipy.interpolate
from PIL import Image
from typing import List, Dict, Optional, Callable
import dnnlib
import legacy


class InterpolationEngine:
    """Engine for generating interpolation videos between W vectors."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.G = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = 'exports/videos'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        """Load the StyleGAN3 model."""
        if self.G is None:
            print(f'Loading model: {self.model_path}...')
            with dnnlib.util.open_url(self.model_path) as f:
                self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            print(f'Model loaded. Resolution: {self.G.img_resolution}')
        return self.G

    def create_interpolation_function(
        self,
        w_vectors: List[np.ndarray],
        kind: str = 'cubic',
        loop: bool = True
    ) -> scipy.interpolate.interp1d:
        """
        Create an interpolation function for a sequence of W vectors.

        Args:
            w_vectors: List of W vectors, each shape (512,)
            kind: Interpolation type ('linear', 'cubic', 'quadratic')
            loop: If True, creates a seamless loop back to the first vector

        Returns:
            Interpolation function that takes t in [0, num_keyframes] and returns W
        """
        # Stack W vectors: shape (num_keyframes, w_dim)
        ws = np.array(w_vectors)
        num_keyframes = len(w_vectors)

        if loop:
            # Add first keyframe at the end for seamless loop
            ws = np.concatenate([ws, ws[0:1]], axis=0)
            x = np.arange(num_keyframes + 1)
        else:
            x = np.arange(num_keyframes)

        # Create interpolation function
        interp = scipy.interpolate.interp1d(x, ws, kind=kind, axis=0)

        return interp, num_keyframes

    def generate_frame(self, w_vector: np.ndarray, size: int = 512) -> Image.Image:
        """Generate a single frame from a W vector."""
        G = self.load_model()

        # Expand W vector to all layers: (1, num_ws, w_dim)
        w_tensor = torch.from_numpy(w_vector).float().to(self.device)
        w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)

        with torch.no_grad():
            img = G.synthesis(w_tensor, noise_mode='const')

        # Convert to PIL Image
        img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = Image.fromarray(img, 'RGB')

        # Resize if needed
        if size != G.img_resolution:
            img_pil = img_pil.resize((size, size), Image.LANCZOS)

        return img_pil

    def generate_video(
        self,
        w_vectors: List[np.ndarray],
        fps: int = 30,
        frames_per_transition: int = 60,
        interpolation_kind: str = 'cubic',
        loop: bool = True,
        image_size: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """
        Generate an interpolation video from a sequence of W vectors.

        Args:
            w_vectors: List of W vectors to interpolate between
            fps: Frames per second for output video
            frames_per_transition: Number of frames between each keyframe
            interpolation_kind: Type of interpolation ('linear', 'cubic', 'quadratic')
            loop: If True, creates a seamless loop
            image_size: Output image size
            progress_callback: Optional callback(progress, message)

        Returns:
            Dict with video info (path, duration, frames, etc.)
        """
        if len(w_vectors) < 2:
            raise ValueError("Need at least 2 W vectors for interpolation")

        # Create interpolation function
        interp, num_keyframes = self.create_interpolation_function(
            w_vectors, kind=interpolation_kind, loop=loop
        )

        # Calculate total frames
        total_frames = num_keyframes * frames_per_transition

        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        video_filename = f'interpolation_{video_id}.mp4'
        video_path = os.path.join(self.output_dir, video_filename)

        # Also save frames as GIF option
        gif_filename = f'interpolation_{video_id}.gif'
        gif_path = os.path.join(self.output_dir, gif_filename)

        if progress_callback:
            progress_callback(0.05, "Generating frames...")

        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()
        frames_for_gif = []

        try:
            for frame_idx in range(total_frames):
                # Calculate interpolation parameter
                t = frame_idx / frames_per_transition

                # Get interpolated W vector
                w_interp = interp(t)

                # Generate frame
                frame = self.generate_frame(w_interp, size=image_size)

                # Save frame to temp directory
                frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
                frame.save(frame_path, 'PNG')

                # Store every Nth frame for GIF (to reduce size)
                if frame_idx % 4 == 0:
                    frames_for_gif.append(frame.copy())

                # Progress callback
                if progress_callback and frame_idx % 10 == 0:
                    progress = 0.1 + 0.75 * (frame_idx / total_frames)
                    progress_callback(
                        progress,
                        f"Generating frame {frame_idx + 1}/{total_frames}..."
                    )

            if progress_callback:
                progress_callback(0.88, "Encoding video with ffmpeg...")

            # Use ffmpeg to create H.264 video
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                video_path
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(0.95, "Saving GIF preview...")

        # Save GIF using PIL
        gif_duration = 1000 // max(fps // 4, 1)  # ms per frame
        frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=gif_duration,
            loop=0
        )

        if progress_callback:
            progress_callback(1.0, "Complete!")

        duration = total_frames / fps

        return {
            'video_id': video_id,
            'video_url': f'/api/interpolation/video/{video_filename}',
            'gif_url': f'/api/interpolation/video/{gif_filename}',
            'video_path': video_path,
            'gif_path': gif_path,
            'duration': round(duration, 2),
            'total_frames': total_frames,
            'fps': fps,
            'keyframes': len(w_vectors),
            'frames_per_transition': frames_per_transition,
            'interpolation_kind': interpolation_kind,
            'loop': loop,
            'image_size': image_size
        }

    def generate_preview_frames(
        self,
        w_vectors: List[np.ndarray],
        num_preview_frames: int = 10,
        interpolation_kind: str = 'cubic',
        loop: bool = True,
        image_size: int = 256
    ) -> List[str]:
        """
        Generate preview frames without creating full video.
        Returns list of image paths.
        """
        if len(w_vectors) < 2:
            raise ValueError("Need at least 2 W vectors for interpolation")

        interp, num_keyframes = self.create_interpolation_function(
            w_vectors, kind=interpolation_kind, loop=loop
        )

        preview_id = str(uuid.uuid4())[:8]
        preview_dir = os.path.join(self.output_dir, f'preview_{preview_id}')
        os.makedirs(preview_dir, exist_ok=True)

        frame_urls = []

        for i in range(num_preview_frames):
            # Sample evenly across the interpolation
            t = (i / num_preview_frames) * num_keyframes
            w_interp = interp(t)

            frame = self.generate_frame(w_interp, size=image_size)

            frame_filename = f'frame_{i:03d}.png'
            frame_path = os.path.join(preview_dir, frame_filename)
            frame.save(frame_path, 'PNG')

            frame_urls.append(f'/api/interpolation/preview/{preview_id}/{frame_filename}')

        return frame_urls


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two vectors.
    Better for Z space interpolation.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)
    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)

    if theta < 1e-6:
        return a

    sin_theta = np.sin(theta)

    result = (np.sin((1 - t) * theta) / sin_theta) * a + (np.sin(t * theta) / sin_theta) * b

    return result
