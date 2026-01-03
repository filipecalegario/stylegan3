import { useState, useEffect } from 'react';
import { useGeneticAlgorithm } from '../hooks/useGeneticAlgorithm';
import { PopulationGrid } from '../components/genetic/PopulationGrid';
import { PipelineDiagram } from '../components/genetic/PipelineDiagram';
import { GenerationControls } from '../components/genetic/GenerationControls';
import { ExportPanel } from '../components/genetic/ExportPanel';
import './GeneticEvolution.css';

const API_BASE = 'http://127.0.0.1:8000';

export function GeneticEvolution() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [populationSize, setPopulationSize] = useState(9);
  const [imageSize, setImageSize] = useState(256);
  const [seed, setSeed] = useState('');
  const [showInitPanel, setShowInitPanel] = useState(true);

  const {
    population,
    generation,
    config,
    fitness,
    isLoading,
    error,
    favorites,
    initializePopulation,
    evolve,
    updateConfig,
    setIndividualFitness,
    addToFavorites,
    removeFromFavorites,
    downloadFavorite,
    downloadFavoriteImage,
    getImageUrl
  } = useGeneticAlgorithm();

  // Fetch available models
  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then(res => res.json())
      .then(data => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      })
      .catch(console.error);
  }, []);

  const handleInitialize = async () => {
    if (!selectedModel) return;

    try {
      await initializePopulation(
        selectedModel,
        populationSize,
        seed ? parseInt(seed, 10) : null,
        imageSize
      );
      setShowInitPanel(false);
    } catch (err) {
      console.error('Failed to initialize:', err);
    }
  };

  const handleReset = () => {
    setShowInitPanel(true);
  };

  const handleEvolve = async () => {
    try {
      await evolve();
    } catch (err) {
      console.error('Failed to evolve:', err);
    }
  };

  return (
    <div className="genetic-evolution">
      <header className="genetic-header">
        <h1>Interactive Genetic Evolution</h1>
        <p>Evolve StyleGAN3 images using genetic algorithms</p>
      </header>

      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}

      {showInitPanel ? (
        <div className="init-panel">
          <h2>Initialize Population</h2>

          <div className="init-form">
            <div className="form-group">
              <label>Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Population Size</label>
              <select
                value={populationSize}
                onChange={(e) => setPopulationSize(parseInt(e.target.value, 10))}
              >
                <option value={4}>4 (2x2)</option>
                <option value={6}>6 (2x3)</option>
                <option value={9}>9 (3x3)</option>
                <option value={12}>12 (3x4)</option>
                <option value={16}>16 (4x4)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Image Size</label>
              <select
                value={imageSize}
                onChange={(e) => setImageSize(parseInt(e.target.value, 10))}
              >
                <option value={256}>256px (fast)</option>
                <option value={512}>512px (balanced)</option>
                <option value={1024}>1024px (native)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Seed (optional)</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="Random"
              />
            </div>

            <button
              className="init-button"
              onClick={handleInitialize}
              disabled={isLoading || !selectedModel}
            >
              {isLoading ? 'Generating...' : 'Generate Population'}
            </button>
          </div>
        </div>
      ) : (
        <div className="evolution-workspace">
          <div className="main-content">
            <GenerationControls
              generation={generation}
              populationSize={population.length}
              onEvolve={handleEvolve}
              onReset={handleReset}
              isLoading={isLoading}
              hasPopulation={population.length > 0}
            />

            <PipelineDiagram
              config={config}
              onConfigChange={updateConfig}
            />

            <PopulationGrid
              population={population}
              fitness={fitness}
              getImageUrl={getImageUrl}
              onFitnessChange={setIndividualFitness}
              onAddFavorite={addToFavorites}
              isLoading={isLoading}
            />
          </div>

          <aside className="sidebar">
            <ExportPanel
              favorites={favorites}
              onRemove={removeFromFavorites}
              onDownloadJson={downloadFavorite}
              onDownloadImage={downloadFavoriteImage}
            />
          </aside>
        </div>
      )}
    </div>
  );
}
