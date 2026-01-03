import { useState } from 'react';
import './PipelineDiagram.css';

export function PipelineDiagram({ config, onConfigChange }) {
  const [expandedNode, setExpandedNode] = useState(null);

  const handleToggle = (key) => {
    onConfigChange({ [key]: !config[key] });
  };

  const handleMethodChange = (key, value) => {
    onConfigChange({ [key]: value });
  };

  const handleValueChange = (key, value) => {
    onConfigChange({ [key]: parseFloat(value) });
  };

  const handleIntChange = (key, value) => {
    onConfigChange({ [key]: parseInt(value, 10) });
  };

  return (
    <div className="pipeline-diagram">
      <svg viewBox="0 0 700 180" className="pipeline-svg">
        {/* Arrows */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#4a9eff" />
          </marker>
        </defs>

        {/* Connection lines */}
        <line x1="140" y1="70" x2="170" y2="70" stroke="#4a9eff" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="310" y1="70" x2="340" y2="70" stroke="#4a9eff" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="480" y1="70" x2="510" y2="70" stroke="#4a9eff" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* Selection Node */}
        <g className="node selection-node" onClick={() => setExpandedNode(expandedNode === 'selection' ? null : 'selection')}>
          <rect x="10" y="30" width="130" height="80" rx="8" className="node-rect active" />
          <text x="75" y="60" className="node-title">Selection</text>
          <text x="75" y="85" className="node-value">{config.selection_method}</text>
        </g>

        {/* Crossover Node */}
        <g
          className={`node crossover-node ${config.crossover_enabled ? 'enabled' : 'disabled'}`}
          onClick={() => handleToggle('crossover_enabled')}
        >
          <rect x="180" y="30" width="130" height="80" rx="8" className={`node-rect ${config.crossover_enabled ? 'active' : 'inactive'}`} />
          <text x="245" y="60" className="node-title">Crossover</text>
          <text x="245" y="85" className="node-value">
            {config.crossover_enabled ? config.crossover_method : 'OFF'}
          </text>
        </g>

        {/* Mutation Node */}
        <g
          className={`node mutation-node ${config.mutation_enabled ? 'enabled' : 'disabled'}`}
          onClick={() => handleToggle('mutation_enabled')}
        >
          <rect x="350" y="30" width="130" height="80" rx="8" className={`node-rect ${config.mutation_enabled ? 'active' : 'inactive'}`} />
          <text x="415" y="60" className="node-title">Mutation</text>
          <text x="415" y="85" className="node-value">
            {config.mutation_enabled ? `${(config.mutation_rate * 100).toFixed(0)}%` : 'OFF'}
          </text>
        </g>

        {/* Elitism Node */}
        <g className="node elitism-node" onClick={() => setExpandedNode(expandedNode === 'elitism' ? null : 'elitism')}>
          <rect x="520" y="30" width="130" height="80" rx="8" className={`node-rect ${config.elitism_count > 0 ? 'active' : 'inactive'}`} />
          <text x="585" y="60" className="node-title">Elitism</text>
          <text x="585" y="85" className="node-value">Keep {config.elitism_count}</text>
        </g>

        {/* Legend */}
        <text x="10" y="160" className="legend-text">Click blocks to toggle on/off</text>
      </svg>

      {/* Config panels */}
      <div className="config-panels">
        {/* Selection config */}
        <div className="config-panel">
          <label>Selection Method</label>
          <select
            value={config.selection_method}
            onChange={(e) => handleMethodChange('selection_method', e.target.value)}
          >
            <option value="roulette">Roulette Wheel</option>
            <option value="tournament">Tournament</option>
          </select>
        </div>

        {/* Crossover config */}
        <div className={`config-panel ${!config.crossover_enabled ? 'disabled' : ''}`}>
          <label>Crossover Method</label>
          <select
            value={config.crossover_method}
            onChange={(e) => handleMethodChange('crossover_method', e.target.value)}
            disabled={!config.crossover_enabled}
          >
            <option value="single_point">Single Point</option>
            <option value="uniform">Uniform</option>
            <option value="blend">Blend</option>
          </select>
        </div>

        {/* Mutation config */}
        <div className={`config-panel ${!config.mutation_enabled ? 'disabled' : ''}`}>
          <label>Mutation Rate: {(config.mutation_rate * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.01"
            value={config.mutation_rate}
            onChange={(e) => handleValueChange('mutation_rate', e.target.value)}
            disabled={!config.mutation_enabled}
          />
          <label>Mutation Strength: {config.mutation_strength.toFixed(2)}</label>
          <input
            type="range"
            min="0.01"
            max="1"
            step="0.01"
            value={config.mutation_strength}
            onChange={(e) => handleValueChange('mutation_strength', e.target.value)}
            disabled={!config.mutation_enabled}
          />
        </div>

        {/* Elitism config */}
        <div className="config-panel">
          <label>Elite Count: {config.elitism_count}</label>
          <input
            type="range"
            min="0"
            max="5"
            step="1"
            value={config.elitism_count}
            onChange={(e) => handleIntChange('elitism_count', e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}
