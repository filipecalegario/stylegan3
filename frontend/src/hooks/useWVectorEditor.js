import { useState, useCallback } from 'react'

const API_BASE = 'http://127.0.0.1:8000'
const VALUE_RANGE = 3  // -3 to +3

export function useWVectorEditor() {
  // W vector state (512 floats, initially zeros)
  const [wVector, setWVector] = useState(() => new Float32Array(512).fill(0))
  const [selectedModel, setSelectedModel] = useState('')
  const [imageUrl, setImageUrl] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState(null)

  // Update a single value
  const updateValue = useCallback((index, value) => {
    setWVector(prev => {
      const newW = new Float32Array(prev)
      newW[index] = Math.max(-VALUE_RANGE, Math.min(VALUE_RANGE, value))
      return newW
    })
  }, [])

  // Update multiple values at once (for painting)
  const updateValues = useCallback((updates) => {
    // updates: Array of { index, value }
    setWVector(prev => {
      const newW = new Float32Array(prev)
      for (const { index, value } of updates) {
        newW[index] = Math.max(-VALUE_RANGE, Math.min(VALUE_RANGE, value))
      }
      return newW
    })
  }, [])

  // Reset to zeros and generate image
  const resetVector = useCallback(async () => {
    const newW = new Float32Array(512).fill(0)
    setWVector(newW)

    // Generate image with zero vector
    if (selectedModel) {
      setIsGenerating(true)
      setError(null)
      try {
        const response = await fetch(`${API_BASE}/api/wvector/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModel,
            w_vector: Array.from(newW),
            image_size: 512
          })
        })
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`)
        }
        const data = await response.json()
        setImageUrl(`${API_BASE}${data.image_url}`)
      } catch (err) {
        setError(err.message)
      } finally {
        setIsGenerating(false)
      }
    }
  }, [selectedModel])

  // Generate image from current W vector
  const generateImage = useCallback(async () => {
    if (!selectedModel) return

    setIsGenerating(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/wvector/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          w_vector: Array.from(wVector),
          image_size: 512
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      const data = await response.json()
      setImageUrl(`${API_BASE}${data.image_url}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsGenerating(false)
    }
  }, [selectedModel, wVector])

  // Save as DNA JSON
  const saveDNA = useCallback(() => {
    const dna = {
      w_vector: Array.from(wVector),
      timestamp: new Date().toISOString(),
      source: 'w-vector-editor'
    }
    const blob = new Blob([JSON.stringify(dna, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dna_manual_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [wVector])

  // Load DNA JSON and generate image
  const loadDNA = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = async (e) => {
      try {
        const dna = JSON.parse(e.target.result)
        if (dna.w_vector && dna.w_vector.length === 512) {
          const newVector = new Float32Array(dna.w_vector)
          setWVector(newVector)

          // Generate image with the loaded vector
          if (selectedModel) {
            setIsGenerating(true)
            setError(null)
            try {
              const response = await fetch(`${API_BASE}/api/wvector/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: selectedModel,
                  w_vector: Array.from(newVector),
                  image_size: 512
                })
              })
              if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`)
              }
              const data = await response.json()
              setImageUrl(`${API_BASE}${data.image_url}`)
            } catch (err) {
              setError(err.message)
            } finally {
              setIsGenerating(false)
            }
          }
        } else {
          setError('Invalid DNA file: must have 512 values')
        }
      } catch (err) {
        setError('Failed to parse DNA file')
      }
    }
    reader.readAsText(file)
  }, [selectedModel])

  // Helper to randomize and generate image
  const randomizeAndGenerate = useCallback(async (range) => {
    const newW = new Float32Array(512)
    for (let i = 0; i < 512; i++) {
      newW[i] = (Math.random() * 2 - 1) * range
    }
    setWVector(newW)

    // Generate image with the new random vector
    if (selectedModel) {
      setIsGenerating(true)
      setError(null)
      try {
        const response = await fetch(`${API_BASE}/api/wvector/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModel,
            w_vector: Array.from(newW),
            image_size: 512
          })
        })
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`)
        }
        const data = await response.json()
        setImageUrl(`${API_BASE}${data.image_url}`)
      } catch (err) {
        setError(err.message)
      } finally {
        setIsGenerating(false)
      }
    }
  }, [selectedModel])

  // Randomize vector full range (-3 to +3)
  const randomizeVector = useCallback(() => {
    randomizeAndGenerate(VALUE_RANGE)
  }, [randomizeAndGenerate])

  // Randomize vector soft range (-1 to +1)
  const randomizeVectorSoft = useCallback(() => {
    randomizeAndGenerate(1)
  }, [randomizeAndGenerate])

  return {
    wVector,
    selectedModel,
    setSelectedModel,
    imageUrl,
    isGenerating,
    error,
    updateValue,
    updateValues,
    resetVector,
    randomizeVector,
    randomizeVectorSoft,
    generateImage,
    saveDNA,
    loadDNA
  }
}
