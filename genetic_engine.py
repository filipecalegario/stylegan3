# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for Interactive Genetic Algorithm.

"""Genetic Algorithm Engine for StyleGAN3 Interactive Evolution."""

import os
import uuid
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
import dnnlib
import legacy


# Faithful configuration of the interactive GA from Carvalho (2020),
# "Exploring the latent space of StyleGAN2 through an interactive genetic
# algorithm" (Latent Hiker): Z-space DNA, fitness-proportionate mating pool,
# uniform crossover, gene-reset mutation at 1%, no elitism, no truncation.
LATENT_HIKER_PRESET = {
    'latent_space': 'z',
    'crossover_enabled': True,
    'crossover_method': 'uniform',
    'mutation_enabled': True,
    'mutation_method': 'reset',
    'mutation_rate': 0.01,
    'elitism_count': 0,
    'selection_method': 'mating_pool',
    'truncation_psi': 1.0,
}


class Individual:
    """Represents an individual in the population.

    Each StyleGAN image is an individual; its latent vector is its DNA.
    Depending on the engine's `latent_space` config, the DNA lives in
    Z-space (raw gaussian latent, as in Carvalho 2020 "Latent Hiker")
    or W-space (post-mapping intermediate latent).
    """

    def __init__(self, dna: np.ndarray, generation: int = 0,
                 parents: Optional[Tuple[str, str]] = None):
        self.id = str(uuid.uuid4())[:8]
        self.dna = dna  # Shape: (latent_dim,) — Z or W depending on engine config
        self.fitness = 5.0  # Default fitness (middle of 0-10)
        self.generation = generation
        self.parents = parents
        self.image_path = None

    # Backwards-compat alias (older code and DNA JSON files use `w_vector`)
    @property
    def w_vector(self) -> np.ndarray:
        return self.dna

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'dna': self.dna.tolist(),
            'w_vector': self.dna.tolist(),  # compat with existing frontend
            'fitness': self.fitness,
            'generation': self.generation,
            'parents': self.parents,
            'image_url': f'/api/genetic/image/{self.id}.png' if self.image_path else None
        }


class GeneticEngine:
    """Engine for interactive genetic algorithm with StyleGAN3."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.G = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.population: List[Individual] = []
        self.generation = 0
        self.config = {
            'latent_space': 'w',  # 'w' (post-mapping) or 'z' (raw latent, Latent Hiker)
            'crossover_enabled': True,
            'crossover_method': 'single_point',  # single_point, uniform, blend
            'mutation_enabled': True,
            'mutation_method': 'gaussian',  # gaussian (perturb) or reset (resample gene, Latent Hiker)
            'mutation_rate': 0.1,
            'mutation_strength': 0.3,
            'elitism_count': 1,
            'selection_method': 'roulette',  # roulette, tournament, mating_pool (Latent Hiker)
            'truncation_psi': 0.7,  # Latent Hiker uses 1.0
            'image_size': 256  # 256, 512, or 1024
        }
        self.image_dir = 'exports/genetic'
        os.makedirs(self.image_dir, exist_ok=True)

    def load_model(self):
        """Load the StyleGAN3 model."""
        if self.G is None:
            print(f'Loading model: {self.model_path}...')
            with dnnlib.util.open_url(self.model_path) as f:
                self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            print(f'Model loaded. W dimension: {self.G.w_dim}')
        return self.G

    def generate_random_dna(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random DNA vector in the configured latent space."""
        G = self.load_model()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.config['latent_space'] == 'z':
            # Latent Hiker: DNA is a raw Z vector sampled from N(0, 1)
            return np.random.randn(G.z_dim).astype(np.float32)

        z = torch.randn(1, G.z_dim, device=self.device)
        c = torch.zeros(1, G.c_dim, device=self.device) if G.c_dim > 0 else None

        with torch.no_grad():
            w = G.mapping(z, c, truncation_psi=self.config['truncation_psi'])

        # Return the first layer's W (they're all the same for fresh samples)
        return w[0, 0].cpu().numpy()

    # Backwards-compat alias
    def generate_random_w(self, seed: Optional[int] = None) -> np.ndarray:
        return self.generate_random_dna(seed=seed)

    def map_z_to_w(self, z_dna: np.ndarray) -> np.ndarray:
        """Map a Z-space DNA to a single W vector (for export compatibility)."""
        G = self.load_model()
        z = torch.from_numpy(z_dna).float().unsqueeze(0).to(self.device)
        c = torch.zeros(1, G.c_dim, device=self.device) if G.c_dim > 0 else None
        with torch.no_grad():
            w = G.mapping(z, c, truncation_psi=self.config['truncation_psi'])
        return w[0, 0].cpu().numpy()

    def generate_image(self, individual: Individual) -> Image.Image:
        """Generate an image from an individual's DNA."""
        G = self.load_model()
        size = self.config['image_size']

        if self.config['latent_space'] == 'z':
            # Full forward pass: Z -> mapping -> synthesis
            z = torch.from_numpy(individual.dna).float().unsqueeze(0).to(self.device)
            c = torch.zeros(1, G.c_dim, device=self.device) if G.c_dim > 0 else None
            with torch.no_grad():
                img = G(z, c, truncation_psi=self.config['truncation_psi'], noise_mode='const')
        else:
            w_tensor = torch.from_numpy(individual.dna).float().to(self.device)
            w_tensor = w_tensor.unsqueeze(0).unsqueeze(0).repeat(1, G.num_ws, 1)
            with torch.no_grad():
                img = G.synthesis(w_tensor, noise_mode='const')

        img = (img[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_pil = Image.fromarray(img, 'RGB')

        # Only resize if different from native resolution
        if size != G.img_resolution:
            img_pil = img_pil.resize((size, size), Image.LANCZOS)

        # Save image
        image_path = os.path.join(self.image_dir, f'{individual.id}.png')
        img_pil.save(image_path, 'PNG')
        individual.image_path = image_path

        return img_pil

    def initialize_population(self, size: int = 9, seed: Optional[int] = None) -> List[Individual]:
        """Initialize a random population."""
        if seed is not None:
            np.random.seed(seed)

        self.population = []
        self.generation = 0

        for i in range(size):
            dna = self.generate_random_dna(seed=seed + i if seed else None)
            individual = Individual(dna, generation=0)
            self.generate_image(individual)
            self.population.append(individual)

        return self.population

    def crossover_single_point(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Single-point crossover of two W vectors."""
        point = np.random.randint(1, len(parent_a))
        child = np.concatenate([parent_a[:point], parent_b[point:]])
        return child

    def crossover_uniform(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Uniform crossover - each gene has 50% chance from each parent."""
        mask = np.random.random(len(parent_a)) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def crossover_blend(self, parent_a: np.ndarray, parent_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend crossover - weighted average of parents."""
        # Random blend factor for each gene
        blend = np.random.uniform(0, 1, len(parent_a))
        child = blend * parent_a + (1 - blend) * parent_b
        return child

    def crossover(self, parent_a: Individual, parent_b: Individual) -> np.ndarray:
        """Perform crossover based on configured method."""
        method = self.config['crossover_method']

        if method == 'uniform':
            return self.crossover_uniform(parent_a.dna, parent_b.dna)
        elif method == 'blend':
            return self.crossover_blend(parent_a.dna, parent_b.dna)
        else:
            return self.crossover_single_point(parent_a.dna, parent_b.dna)

    def mutate(self, dna: np.ndarray) -> np.ndarray:
        """Apply mutation to a DNA vector.

        Methods:
        - 'gaussian': perturb selected genes with gaussian noise scaled by strength
        - 'reset': replace selected genes with fresh N(0,1) samples (Latent Hiker)
        """
        rate = self.config['mutation_rate']
        method = self.config['mutation_method']

        # Create mutation mask
        mask = np.random.random(len(dna)) < rate
        mutated = dna.copy()

        if method == 'reset':
            mutated[mask] = np.random.randn(int(mask.sum())).astype(dna.dtype)
        else:
            strength = self.config['mutation_strength']
            noise = np.random.randn(len(dna)) * strength
            mutated[mask] += noise[mask]

        return mutated

    def selection_roulette(self) -> Individual:
        """Roulette wheel selection based on fitness."""
        # Normalize fitness to sum to 1
        total_fitness = sum(ind.fitness for ind in self.population)
        if total_fitness == 0:
            return np.random.choice(self.population)

        probabilities = [ind.fitness / total_fitness for ind in self.population]

        # Roulette wheel
        r = np.random.random()
        cumsum = 0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                return self.population[i]

        return self.population[-1]

    def selection_tournament(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        contestants = np.random.choice(self.population, size=min(tournament_size, len(self.population)), replace=False)
        return max(contestants, key=lambda ind: ind.fitness)

    def build_mating_pool(self) -> List[Individual]:
        """Fitness-proportionate mating pool (Latent Hiker / Carvalho 2020).

        Each individual enters the pool round(fitness) times, so a 10-rated
        image is 10x more likely to reproduce than a 1-rated one. If every
        rating is zero the pool is filled with brand-new random individuals,
        restarting the search from a random region of the latent space.
        """
        pool = []
        for ind in self.population:
            pool.extend([ind] * int(round(ind.fitness)))

        if not pool:
            pool = [
                Individual(self.generate_random_dna(), generation=self.generation)
                for _ in range(len(self.population))
            ]

        return pool

    def select_parent(self, mating_pool: Optional[List[Individual]] = None) -> Individual:
        """Select a parent based on configured method."""
        method = self.config['selection_method']

        if method == 'mating_pool' and mating_pool:
            return mating_pool[np.random.randint(len(mating_pool))]
        elif method == 'tournament':
            return self.selection_tournament()
        else:
            return self.selection_roulette()

    def update_fitness(self, fitness_dict: Dict[str, float]):
        """Update fitness values from frontend."""
        for individual in self.population:
            if individual.id in fitness_dict:
                individual.fitness = fitness_dict[individual.id]

    def evolve(self, fitness_dict: Dict[str, float]) -> List[Individual]:
        """Evolve to the next generation."""
        # Update fitness values
        self.update_fitness(fitness_dict)

        population_size = len(self.population)
        self.generation += 1

        # Get elite individuals
        elitism_count = self.config['elitism_count']
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_pop[:elitism_count]

        # Build mating pool once per generation (Latent Hiker selection)
        mating_pool = None
        if self.config['selection_method'] == 'mating_pool':
            mating_pool = self.build_mating_pool()

        # Create new population
        new_population = []

        # Keep elite
        for ind in elite:
            new_ind = Individual(
                ind.dna.copy(),
                generation=self.generation,
                parents=(ind.id, ind.id)  # Self-parent for elite
            )
            new_population.append(new_ind)

        # Generate children
        while len(new_population) < population_size:
            parent_a = self.select_parent(mating_pool)
            parent_b = self.select_parent(mating_pool)

            # Crossover
            if self.config['crossover_enabled']:
                child_dna = self.crossover(parent_a, parent_b)
            else:
                # Just copy from one parent
                child_dna = parent_a.dna.copy()

            # Mutation
            if self.config['mutation_enabled']:
                child_dna = self.mutate(child_dna)

            child = Individual(
                child_dna,
                generation=self.generation,
                parents=(parent_a.id, parent_b.id)
            )
            new_population.append(child)

        # Generate images for new population
        self.population = new_population
        for individual in self.population:
            self.generate_image(individual)

        return self.population

    def export_individual(self, individual_id: str) -> Optional[Dict]:
        """Export an individual's data for saving.

        Always includes a `w_vector` so exported DNA files stay compatible
        with the Interpolation and W Editor pages. In Z-space mode the raw
        DNA is exported as `dna` and the mapped W vector as `w_vector`.
        """
        for ind in self.population:
            if ind.id == individual_id:
                latent_space = self.config['latent_space']
                if latent_space == 'z':
                    w_vector = self.map_z_to_w(ind.dna).tolist()
                else:
                    w_vector = ind.dna.tolist()
                return {
                    'id': ind.id,
                    'dna': ind.dna.tolist(),
                    'latent_space': latent_space,
                    'w_vector': w_vector,
                    'fitness': ind.fitness,
                    'generation': ind.generation,
                    'parents': ind.parents,
                    'image_path': ind.image_path
                }
        return None

    def import_individual(self, dna: List[float]) -> Individual:
        """Import a DNA vector (in the current latent space) as a new individual."""
        dna_array = np.array(dna, dtype=np.float32)
        individual = Individual(dna_array, generation=self.generation)
        self.generate_image(individual)
        return individual

    def get_population_state(self) -> Dict:
        """Get the current state of the population."""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'individuals': [ind.to_dict() for ind in self.population],
            'config': self.config
        }

    def update_config(self, new_config: Dict):
        """Update genetic algorithm configuration."""
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
