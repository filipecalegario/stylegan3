import './InterpolationConfig.css';

export function InterpolationConfig({ config, onConfigChange, isLoading }) {
  const handleChange = (key, value) => {
    onConfigChange({ [key]: value });
  };

  return (
    <div className="interpolation-config">
      <h3>Interpolation Settings</h3>

      <div className="config-grid">
        <div className="config-item">
          <label>Interpolation Type</label>
          <select
            value={config.interpolation_kind}
            onChange={(e) => handleChange('interpolation_kind', e.target.value)}
            disabled={isLoading}
          >
            <option value="linear">Linear (sharp transitions)</option>
            <option value="quadratic">Quadratic (smooth)</option>
            <option value="cubic">Cubic (very smooth)</option>
          </select>
        </div>

        <div className="config-item">
          <label>Frames per Transition: {config.frames_per_transition}</label>
          <input
            type="range"
            min="15"
            max="120"
            step="5"
            value={config.frames_per_transition}
            onChange={(e) => handleChange('frames_per_transition', parseInt(e.target.value))}
            disabled={isLoading}
          />
          <div className="range-labels">
            <span>Fast</span>
            <span>Slow</span>
          </div>
        </div>

        <div className="config-item">
          <label>FPS: {config.fps}</label>
          <input
            type="range"
            min="15"
            max="60"
            step="5"
            value={config.fps}
            onChange={(e) => handleChange('fps', parseInt(e.target.value))}
            disabled={isLoading}
          />
          <div className="range-labels">
            <span>15</span>
            <span>60</span>
          </div>
        </div>

        <div className="config-item">
          <label>Image Size</label>
          <select
            value={config.image_size}
            onChange={(e) => handleChange('image_size', parseInt(e.target.value))}
            disabled={isLoading}
          >
            <option value={256}>256px (fast)</option>
            <option value={512}>512px (balanced)</option>
            <option value={1024}>1024px (native)</option>
          </select>
        </div>

        <div className="config-item checkbox-item">
          <label>
            <input
              type="checkbox"
              checked={config.loop}
              onChange={(e) => handleChange('loop', e.target.checked)}
              disabled={isLoading}
            />
            <span>Seamless Loop</span>
          </label>
          <span className="hint">Returns to first DNA at the end</span>
        </div>
      </div>

      <div className="duration-estimate">
        <span className="label">Estimated Duration:</span>
        <span className="value">
          {((config.frames_per_transition * (config.loop ? 1 : 1)) / config.fps).toFixed(1)}s per transition
        </span>
      </div>
    </div>
  );
}
