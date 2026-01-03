import { useState, useEffect } from 'react'
import { useWVectorEditor } from '../hooks/useWVectorEditor'
import { WVectorCanvas } from '../components/wvector/WVectorCanvas'
import './WVectorEditor.css'

const API_BASE = 'http://127.0.0.1:8000'

export function WVectorEditor() {
  const [models, setModels] = useState([])
  const {
    wVector,
    selectedModel,
    setSelectedModel,
    imageUrl,
    isGenerating,
    error,
    updateValues,
    resetVector,
    randomizeVector,
    randomizeVectorSoft,
    generateImage,
    saveDNA,
    loadDNA,
    // Sequencer
    sequencerRunning,
    sequencerPaused,
    sequencerDimension,
    sequencerValue,
    sequencerConfig,
    startSequencer,
    pauseSequencer,
    resumeSequencer,
    stopSequencer,
    updateSequencerConfig
  } = useWVectorEditor()

  // Fetch available models
  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then(res => res.json())
      .then(data => {
        setModels(data.models || [])
        if (data.models?.length > 0) {
          setSelectedModel(data.models[0])
        }
      })
      .catch(console.error)
  }, [setSelectedModel])

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0]
    if (file) loadDNA(file)
  }

  return (
    <div className="wvector-editor">
      <header className="editor-header">
        <h1>W Vector Editor</h1>
        <p>Paint the 512 dimensions to create custom images</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Toolbar horizontal */}
      <div className="toolbar">
        <div className="toolbar-group">
          <label>Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isGenerating}
          >
            {models.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        <div className="toolbar-group">
          <button onClick={resetVector} disabled={isGenerating}>
            Reset
          </button>
          <button onClick={randomizeVectorSoft} disabled={isGenerating}>
            Soft Random
          </button>
          <button onClick={randomizeVector} disabled={isGenerating}>
            Full Random
          </button>
          <button onClick={saveDNA} disabled={isGenerating}>
            Save DNA
          </button>
          <label className="file-upload-btn">
            Load DNA
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              hidden
            />
          </label>
        </div>
      </div>

      {/* Sequencer controls */}
      <div className="sequencer-bar">
        <div className="sequencer-controls">
          {!sequencerRunning ? (
            <button onClick={startSequencer} disabled={isGenerating} className="seq-btn start">
              ▶ Start Sequencer
            </button>
          ) : (
            <>
              {sequencerPaused ? (
                <button onClick={resumeSequencer} className="seq-btn resume">
                  ▶ Resume
                </button>
              ) : (
                <button onClick={pauseSequencer} className="seq-btn pause">
                  ⏸ Pause
                </button>
              )}
              <button onClick={stopSequencer} className="seq-btn stop">
                ⏹ Stop & Keep
              </button>
            </>
          )}
        </div>

        <div className="sequencer-config">
          <label>
            Increment:
            <input
              type="number"
              value={sequencerConfig.increment}
              onChange={(e) => updateSequencerConfig({ increment: parseFloat(e.target.value) || 0.1 })}
              step="0.05"
              min="0.01"
              max="1"
              disabled={sequencerRunning}
            />
          </label>
          <label>
            Delay (ms):
            <input
              type="number"
              value={sequencerConfig.delay}
              onChange={(e) => updateSequencerConfig({ delay: parseInt(e.target.value) || 20 })}
              step="10"
              min="10"
              max="1000"
              disabled={sequencerRunning}
            />
          </label>
        </div>

        {sequencerRunning && (
          <div className="sequencer-status">
            <span className="dim-label">Dimension:</span>
            <span className="dim-value">{sequencerDimension}/511</span>
            <span className="val-label">Value:</span>
            <span className="val-value">{sequencerValue.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Canvas full width */}
      <div className="canvas-section">
        <WVectorCanvas
          wVector={wVector}
          onUpdateValues={updateValues}
          onMouseUp={generateImage}
        />
        <p className="hint">
          Click and drag to paint. <span className="green">Green = positive</span> | <span className="red">Red = negative</span>
        </p>
      </div>

      {/* Image preview below */}
      <div className="preview-section">
        {isGenerating ? (
          <div className="generating">
            <div className="spinner"></div>
            <span>Generating...</span>
          </div>
        ) : imageUrl ? (
          <img src={imageUrl} alt="Generated" className="preview-image" />
        ) : (
          <div className="placeholder">
            <span>Paint the bars above and release to generate image</span>
          </div>
        )}
      </div>
    </div>
  )
}
