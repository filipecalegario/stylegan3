import { useState, useCallback } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

export function useGeneticAlgorithm() {
  const [population, setPopulation] = useState([]);
  const [generation, setGeneration] = useState(0);
  const [config, setConfig] = useState({
    crossover_enabled: true,
    crossover_method: 'single_point',
    mutation_enabled: true,
    mutation_rate: 0.1,
    mutation_strength: 0.3,
    elitism_count: 1,
    selection_method: 'roulette',
    image_size: 256
  });
  const [fitness, setFitness] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [favorites, setFavorites] = useState([]);

  const updatePopulationState = useCallback((data) => {
    setPopulation(data.individuals || []);
    setGeneration(data.generation || 0);
    if (data.config) {
      setConfig(data.config);
    }
    // Initialize fitness values for new population
    const newFitness = {};
    (data.individuals || []).forEach(ind => {
      newFitness[ind.id] = ind.fitness || 5.0;
    });
    setFitness(newFitness);
  }, []);

  const initializePopulation = useCallback(async (model, populationSize = 9, seed = null, imageSize = 256) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/genetic/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          population_size: populationSize,
          seed,
          image_size: imageSize
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize: ${response.statusText}`);
      }

      const data = await response.json();
      updatePopulationState(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [updatePopulationState]);

  const evolve = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/genetic/evolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fitness })
      });

      if (!response.ok) {
        throw new Error(`Failed to evolve: ${response.statusText}`);
      }

      const data = await response.json();
      updatePopulationState(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fitness, updatePopulationState]);

  const updateConfig = useCallback(async (newConfig) => {
    try {
      const response = await fetch(`${API_BASE}/api/genetic/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newConfig)
      });

      if (!response.ok) {
        throw new Error(`Failed to update config: ${response.statusText}`);
      }

      const data = await response.json();
      setConfig(data.config);
      return data.config;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, []);

  const setIndividualFitness = useCallback((id, value) => {
    setFitness(prev => ({
      ...prev,
      [id]: value
    }));
  }, []);

  const exportIndividual = useCallback(async (individualId) => {
    try {
      const response = await fetch(`${API_BASE}/api/genetic/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ individual_id: individualId })
      });

      if (!response.ok) {
        throw new Error(`Failed to export: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, []);

  const addToFavorites = useCallback(async (individualId) => {
    const data = await exportIndividual(individualId);
    if (data && !data.error) {
      setFavorites(prev => {
        // Avoid duplicates
        if (prev.some(f => f.id === data.id)) {
          return prev;
        }
        return [...prev, data];
      });
    }
    return data;
  }, [exportIndividual]);

  const removeFromFavorites = useCallback((individualId) => {
    setFavorites(prev => prev.filter(f => f.id !== individualId));
  }, []);

  const downloadFavorite = useCallback((favorite) => {
    // Download JSON with W vector
    const jsonData = JSON.stringify({
      id: favorite.id,
      w_vector: favorite.w_vector,
      fitness: favorite.fitness,
      generation: favorite.generation
    }, null, 2);

    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `individual_${favorite.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const downloadFavoriteImage = useCallback((favorite) => {
    const imageUrl = `${API_BASE}/api/genetic/image/${favorite.id}.png`;
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `individual_${favorite.id}.png`;
    a.click();
  }, []);

  const getImageUrl = useCallback((individual) => {
    if (individual.image_url) {
      return `${API_BASE}${individual.image_url}`;
    }
    return `${API_BASE}/api/genetic/image/${individual.id}.png`;
  }, []);

  return {
    // State
    population,
    generation,
    config,
    fitness,
    isLoading,
    error,
    favorites,

    // Actions
    initializePopulation,
    evolve,
    updateConfig,
    setIndividualFitness,
    exportIndividual,
    addToFavorites,
    removeFromFavorites,
    downloadFavorite,
    downloadFavoriteImage,
    getImageUrl
  };
}
