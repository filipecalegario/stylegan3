import { useState, useEffect } from 'react'
import ParameterPanel from '../components/ParameterPanel'
import ImageViewer from '../components/ImageViewer'
import ProgressBar from '../components/ProgressBar'
import ImageModal from '../components/ImageModal'
import useWebSocket from '../hooks/useWebSocket'
import './LatentExplorer.css'

export function LatentExplorer() {
  const [models, setModels] = useState([])
  const [params, setParams] = useState({
    model: '',
    num_samples: 400,
    thumb_size: 64,
    layout_mode: 'overlap',
    spread: 0.5,
    truncation_psi: 0.7,
    canvas_size: 3000,
    seed: '',
  })
  const [modalImage, setModalImage] = useState(null)

  const { connect, send, progress, result, error, isGenerating } = useWebSocket()

  // Fetch models on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/models')
      .then(res => {
        console.log('Response status:', res.status)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        return res.json()
      })
      .then(data => {
        console.log('Models loaded:', data)
        setModels(data.models || [])
        if (data.models && data.models.length > 0) {
          setParams(prev => ({ ...prev, model: data.models[0] }))
        }
      })
      .catch(err => console.error('Failed to fetch models:', err))
  }, [])

  const handleGenerate = () => {
    if (!params.model) {
      alert('Please select a model')
      return
    }
    connect()
    setTimeout(() => {
      send({ action: 'generate', params })
    }, 100)
  }

  const handleParamChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const handleImageClick = (e, imageUrl) => {
    setModalImage(imageUrl)
  }

  return (
    <div className="latent-explorer">
      <header className="header">
        <h1>StyleGAN3 Latent Space Explorer</h1>
      </header>

      <main className="main">
        <aside className="sidebar">
          <ParameterPanel
            models={models}
            params={params}
            onChange={handleParamChange}
            onGenerate={handleGenerate}
            disabled={isGenerating}
          />

          {(isGenerating || progress) && (
            <ProgressBar
              value={progress?.value || 0}
              message={progress?.message || 'Starting...'}
            />
          )}

          {result && (
            <div className="result-info">
              <h3>Generation Complete</h3>
              <p><strong>Seed:</strong> {result.seed}</p>
              <p><strong>Time:</strong> {result.time}s</p>
              <p><strong>Samples:</strong> {result.num_samples}</p>
              <p><strong>Layout:</strong> {result.layout_mode}</p>
            </div>
          )}

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
        </aside>

        <section className="content">
          {result?.image_url ? (
            <ImageViewer
              imageUrl={result.image_url}
              onImageClick={handleImageClick}
            />
          ) : (
            <div className="placeholder">
              <p>Configure parameters and click "Generate" to create a latent space map.</p>
            </div>
          )}
        </section>
      </main>

      {modalImage && (
        <ImageModal
          imageUrl={modalImage}
          onClose={() => setModalImage(null)}
        />
      )}
    </div>
  )
}
