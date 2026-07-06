"""Smoke test for the Latent Hiker IGA logic (no GPU/model needed)."""
import numpy as np
from genetic_engine import GeneticEngine, Individual, LATENT_HIKER_PRESET


class FakeG:
    z_dim = 512
    c_dim = 0


def make_engine():
    e = GeneticEngine('dummy.pkl')
    e.update_config(LATENT_HIKER_PRESET)
    e.G = FakeG()  # bypass model loading
    e.generate_image = lambda ind: None  # bypass image synthesis
    return e


# --- Config preset applied
e = make_engine()
assert e.config['latent_space'] == 'z'
assert e.config['selection_method'] == 'mating_pool'
assert e.config['crossover_method'] == 'uniform'
assert e.config['mutation_method'] == 'reset'
assert e.config['mutation_rate'] == 0.01
assert e.config['elitism_count'] == 0
assert e.config['truncation_psi'] == 1.0
print('preset ok')

# --- Random DNA in Z space
dna = e.generate_random_dna()
assert dna.shape == (512,) and dna.dtype == np.float32
print('random z dna ok')

# --- Population + mating pool proportional to fitness
np.random.seed(0)
e.population = [Individual(np.random.randn(512).astype(np.float32)) for _ in range(4)]
fit = {e.population[0].id: 10, e.population[1].id: 5,
       e.population[2].id: 1, e.population[3].id: 0}
e.update_fitness(fit)
pool = e.build_mating_pool()
assert len(pool) == 16
assert sum(1 for p in pool if p.id == e.population[0].id) == 10
assert sum(1 for p in pool if p.id == e.population[3].id) == 0
print('mating pool ok')

# --- All-zero fitness -> pool of fresh random individuals
e2 = make_engine()
e2.population = [Individual(np.zeros(512, dtype=np.float32)) for _ in range(4)]
e2.update_fitness({ind.id: 0 for ind in e2.population})
pool2 = e2.build_mating_pool()
assert len(pool2) == 4
assert all(p.id not in {i.id for i in e2.population} for p in pool2)
assert not np.allclose(pool2[0].dna, 0)
print('zero-fitness restart ok')

# --- Reset mutation replaces genes rather than perturbing
e3 = make_engine()
e3.config['mutation_rate'] = 1.0  # mutate every gene
base = np.full(512, 100.0, dtype=np.float32)
mutated = e3.mutate(base)
assert np.all(np.abs(mutated) < 10), 'reset mutation should resample from N(0,1)'
assert np.all(base == 100.0), 'input must not be modified in place'
print('reset mutation ok')

# --- Uniform crossover mixes genes from both parents
a = np.zeros(512, dtype=np.float32)
b = np.ones(512, dtype=np.float32)
child = e.crossover_uniform(a, b)
assert set(np.unique(child)) == {0.0, 1.0}
assert 100 < child.sum() < 412  # roughly 50/50
print('uniform crossover ok')

# --- Full evolve step (Latent Hiker config, no elitism)
e4 = make_engine()
np.random.seed(1)
e4.population = [Individual(np.random.randn(512).astype(np.float32)) for _ in range(6)]
old_ids = {ind.id for ind in e4.population}
e4.evolve({ind.id: i for i, ind in enumerate(e4.population)})
assert len(e4.population) == 6
assert e4.generation == 1
assert all(ind.id not in old_ids for ind in e4.population), 'no elitism: all replaced'
assert all(ind.parents is not None for ind in e4.population)
print('evolve ok')

# --- W-space default mode still intact (backwards compat)
e5 = GeneticEngine('dummy.pkl')
assert e5.config['latent_space'] == 'w'
assert e5.config['selection_method'] == 'roulette'
assert e5.config['mutation_method'] == 'gaussian'
ind = Individual(np.arange(512, dtype=np.float32))
assert np.array_equal(ind.w_vector, ind.dna), 'w_vector alias'
d = ind.to_dict()
assert d['w_vector'] == d['dna']
print('w-space compat ok')

print('\nALL TESTS PASSED')
