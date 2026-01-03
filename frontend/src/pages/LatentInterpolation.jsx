import { useState, useEffect } from 'react';
import { useInterpolation } from '../hooks/useInterpolation';
import { DNAUploader } from '../components/interpolation/DNAUploader';
import { InterpolationConfig } from '../components/interpolation/InterpolationConfig';
import { VideoPreview } from '../components/interpolation/VideoPreview';
import './LatentInterpolation.css';

const API_BASE = 'http://127.0.0.1:8000';

export function LatentInterpolation() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const {
    dnaFiles,
    config,
    previewFrames,
    result,
    progress,
    isLoading,
    error,
    addDnaFile,
    removeDnaFile,
    reorderDnaFiles,
    clearDnaFiles,
    updateConfig,
    generatePreview,
    generateVideo,
    cancelGeneration
  } = useInterpolation();

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

  const handlePreview = () => {
    if (selectedModel && dnaFiles.length >= 2) {
      generatePreview(selectedModel);
    }
  };

  const handleGenerate = () => {
    if (selectedModel && dnaFiles.length >= 2) {
      generateVideo(selectedModel);
    }
  };

  const canGenerate = dnaFiles.length >= 2 && selectedModel && !isLoading;

  return (
    <div className="latent-interpolation">
      <header className="interpolation-header">
        <h1>Latent Space Interpolation</h1>
        <p>Create smooth transition videos between DNA keyframes</p>
      </header>

      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}

      <div className="interpolation-layout">
        <aside className="interpolation-sidebar">
          <div className="model-selector">
            <label>Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isLoading}
            >
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <DNAUploader
            dnaFiles={dnaFiles}
            onAddFile={addDnaFile}
            onRemoveFile={removeDnaFile}
            onReorder={reorderDnaFiles}
            onClear={clearDnaFiles}
            isLoading={isLoading}
          />

          <InterpolationConfig
            config={config}
            onConfigChange={updateConfig}
            isLoading={isLoading}
          />

          <div className="action-buttons">
            <button
              className="preview-btn"
              onClick={handlePreview}
              disabled={!canGenerate}
            >
              Preview Frames
            </button>
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={!canGenerate}
            >
              {isLoading ? 'Generating...' : 'Generate Video'}
            </button>
            {isLoading && (
              <button
                className="cancel-btn"
                onClick={cancelGeneration}
              >
                Cancel
              </button>
            )}
          </div>
        </aside>

        <main className="interpolation-main">
          <VideoPreview
            previewFrames={previewFrames}
            result={result}
            progress={progress}
            isLoading={isLoading}
          />
        </main>
      </div>
    </div>
  );
}
