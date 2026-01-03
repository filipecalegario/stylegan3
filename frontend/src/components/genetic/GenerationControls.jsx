import './GenerationControls.css';

export function GenerationControls({
  generation,
  populationSize,
  onEvolve,
  onReset,
  isLoading,
  hasPopulation
}) {
  return (
    <div className="generation-controls">
      <div className="generation-info">
        <div className="generation-counter">
          <span className="label">Generation</span>
          <span className="value">{generation}</span>
        </div>
        <div className="population-size">
          <span className="label">Population</span>
          <span className="value">{populationSize}</span>
        </div>
      </div>

      <div className="control-buttons">
        <button
          className="evolve-button"
          onClick={onEvolve}
          disabled={isLoading || !hasPopulation}
        >
          {isLoading ? 'Evolving...' : 'Evolve'}
        </button>
        <button
          className="reset-button"
          onClick={onReset}
          disabled={isLoading}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
