function ParameterPanel({ models, params, onChange, onGenerate, disabled }) {
  return (
    <div className="parameter-panel">
      <h2>Parameters</h2>

      <div className="param-group">
        <label>Model</label>
        <select
          value={params.model}
          onChange={(e) => onChange('model', e.target.value)}
          disabled={disabled}
        >
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      <div className="param-group">
        <label>Number of Samples</label>
        <input
          type="range"
          min="25"
          max="1600"
          step="25"
          value={params.num_samples}
          onChange={(e) => onChange('num_samples', parseInt(e.target.value))}
          disabled={disabled}
        />
        <div className="slider-value">{params.num_samples}</div>
      </div>

      <div className="param-group">
        <label>Thumbnail Size</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="thumb_size"
              value="32"
              checked={params.thumb_size === 32}
              onChange={() => onChange('thumb_size', 32)}
              disabled={disabled}
            />
            32px
          </label>
          <label>
            <input
              type="radio"
              name="thumb_size"
              value="64"
              checked={params.thumb_size === 64}
              onChange={() => onChange('thumb_size', 64)}
              disabled={disabled}
            />
            64px
          </label>
        </div>
      </div>

      <div className="param-group">
        <label>Layout Mode</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="layout_mode"
              value="overlap"
              checked={params.layout_mode === 'overlap'}
              onChange={() => onChange('layout_mode', 'overlap')}
              disabled={disabled}
            />
            Overlap
          </label>
          <label>
            <input
              type="radio"
              name="layout_mode"
              value="grid"
              checked={params.layout_mode === 'grid'}
              onChange={() => onChange('layout_mode', 'grid')}
              disabled={disabled}
            />
            Grid
          </label>
        </div>
      </div>

      <div className="param-group">
        <label>Spread (W-space std dev)</label>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.1"
          value={params.spread}
          onChange={(e) => onChange('spread', parseFloat(e.target.value))}
          disabled={disabled}
        />
        <div className="slider-value">{params.spread.toFixed(1)}</div>
      </div>

      <div className="param-group">
        <label>Truncation Psi</label>
        <input
          type="range"
          min="0.0"
          max="1.0"
          step="0.05"
          value={params.truncation_psi}
          onChange={(e) => onChange('truncation_psi', parseFloat(e.target.value))}
          disabled={disabled}
        />
        <div className="slider-value">{params.truncation_psi.toFixed(2)}</div>
      </div>

      <div className="param-group">
        <label>Canvas Size (px)</label>
        <input
          type="range"
          min="1000"
          max="5000"
          step="500"
          value={params.canvas_size}
          onChange={(e) => onChange('canvas_size', parseInt(e.target.value))}
          disabled={disabled}
        />
        <div className="slider-value">{params.canvas_size}</div>
      </div>

      <div className="param-group">
        <label>Seed (optional)</label>
        <input
          type="text"
          value={params.seed}
          onChange={(e) => onChange('seed', e.target.value)}
          placeholder="Random if empty"
          disabled={disabled}
        />
      </div>

      <button
        className="generate-btn"
        onClick={onGenerate}
        disabled={disabled}
      >
        {disabled ? 'Generating...' : 'Generate Map'}
      </button>
    </div>
  )
}

export default ParameterPanel
