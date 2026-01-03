import { useState, useCallback, useRef } from 'react'

const API_BASE = 'http://127.0.0.1:8000'
const VALUE_RANGE = 3  // -3 to +3

export function useWVectorEditor() {
  // W vector state (512 floats, initially zeros)
  const [wVector, setWVector] = useState(() => new Float32Array(512).fill(0))
  const [selectedModel, setSelectedModel] = useState('')
  const [imageUrl, setImageUrl] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState(null)

  // Sequencer state
  const [sequencerRunning, setSequencerRunning] = useState(false)
  const [sequencerPaused, setSequencerPaused] = useState(false)
  const [sequencerDimension, setSequencerDimension] = useState(0)
  const [sequencerValue, setSequencerValue] = useState(-VALUE_RANGE)
  const sequencerRef = useRef(null)
  const originalVectorRef = useRef(null)
  const sequencerStoppedRef = useRef(false)  // Flag to stop the loop
  const [sequencerConfig, setSequencerConfig] = useState({
    increment: 0.1,
    delay: 20,
    minValue: -VALUE_RANGE,
    maxValue: VALUE_RANGE
  })

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

  // Sequencer: generate image without blocking
  const sequencerGenerateImage = useCallback(async (vector) => {
    if (!selectedModel) return

    try {
      const response = await fetch(`${API_BASE}/api/wvector/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          w_vector: Array.from(vector),
          image_size: 512
        })
      })
      if (response.ok) {
        const data = await response.json()
        // Add timestamp to prevent caching
        setImageUrl(`${API_BASE}${data.image_url}?t=${Date.now()}`)
      }
    } catch (err) {
      // Ignore errors during sequencer to keep it running
    }
  }, [selectedModel])

  // Start sequencer
  const startSequencer = useCallback(() => {
    if (sequencerRunning) return

    // Save original vector
    originalVectorRef.current = new Float32Array(wVector)
    sequencerStoppedRef.current = false  // Reset stop flag

    setSequencerRunning(true)
    setSequencerPaused(false)
    setSequencerDimension(0)
    setSequencerValue(sequencerConfig.minValue)

    const runStep = async (dim, val, vector) => {
      // Check if stopped
      if (sequencerStoppedRef.current) return

      // Create new vector with modified dimension
      const newVector = new Float32Array(vector)
      newVector[dim] = val
      setWVector(newVector)
      setSequencerValue(val)

      // Generate image
      await sequencerGenerateImage(newVector)

      // Check again if stopped during image generation
      if (sequencerStoppedRef.current) return

      // Calculate next step
      let nextVal = val + sequencerConfig.increment
      let nextDim = dim
      let nextVector = vector

      if (nextVal > sequencerConfig.maxValue) {
        // Restore this dimension to original value
        nextVector = new Float32Array(vector)
        nextVector[dim] = originalVectorRef.current[dim]

        // Move to next dimension
        nextDim = dim + 1
        nextVal = sequencerConfig.minValue

        if (nextDim >= 512) {
          // Finished all dimensions
          setSequencerRunning(false)
          setSequencerPaused(false)
          setWVector(originalVectorRef.current)
          return
        }

        setSequencerDimension(nextDim)
      }

      // Schedule next step (only if not stopped)
      if (!sequencerStoppedRef.current) {
        sequencerRef.current = setTimeout(() => {
          runStep(nextDim, nextVal, nextVector)
        }, sequencerConfig.delay)
      }
    }

    // Start first step
    runStep(0, sequencerConfig.minValue, originalVectorRef.current)
  }, [sequencerRunning, wVector, sequencerConfig, sequencerGenerateImage])

  // Pause sequencer
  const pauseSequencer = useCallback(() => {
    sequencerStoppedRef.current = true  // Stop the loop temporarily
    if (sequencerRef.current) {
      clearTimeout(sequencerRef.current)
      sequencerRef.current = null
    }
    setSequencerPaused(true)
  }, [])

  // Resume sequencer
  const resumeSequencer = useCallback(() => {
    if (!sequencerRunning || !sequencerPaused) return

    sequencerStoppedRef.current = false  // Reset stop flag
    setSequencerPaused(false)

    const runStep = async (dim, val, vector) => {
      // Check if stopped
      if (sequencerStoppedRef.current) return

      // Create new vector with modified dimension
      const newVector = new Float32Array(vector)
      newVector[dim] = val
      setWVector(newVector)
      setSequencerValue(val)

      // Generate image
      await sequencerGenerateImage(newVector)

      // Check again if stopped during image generation
      if (sequencerStoppedRef.current) return

      // Calculate next step
      let nextVal = val + sequencerConfig.increment
      let nextDim = dim
      let nextVector = vector

      if (nextVal > sequencerConfig.maxValue) {
        // Restore this dimension to original value
        nextVector = new Float32Array(vector)
        nextVector[dim] = originalVectorRef.current[dim]

        // Move to next dimension
        nextDim = dim + 1
        nextVal = sequencerConfig.minValue

        if (nextDim >= 512) {
          // Finished all dimensions
          setSequencerRunning(false)
          setSequencerPaused(false)
          setWVector(originalVectorRef.current)
          return
        }

        setSequencerDimension(nextDim)
      }

      // Schedule next step (only if not stopped)
      if (!sequencerStoppedRef.current) {
        sequencerRef.current = setTimeout(() => {
          runStep(nextDim, nextVal, nextVector)
        }, sequencerConfig.delay)
      }
    }

    // Resume from current state
    runStep(sequencerDimension, sequencerValue, wVector)
  }, [sequencerRunning, sequencerPaused, sequencerDimension, sequencerValue, wVector, sequencerConfig, sequencerGenerateImage])

  // Stop sequencer and keep current state
  const stopSequencer = useCallback(() => {
    sequencerStoppedRef.current = true  // Set flag to stop the loop
    if (sequencerRef.current) {
      clearTimeout(sequencerRef.current)
      sequencerRef.current = null
    }
    setSequencerRunning(false)
    setSequencerPaused(false)
    // Keep current wVector as is - user found something interesting!
  }, [])

  // Update sequencer config
  const updateSequencerConfig = useCallback((updates) => {
    setSequencerConfig(prev => ({ ...prev, ...updates }))
  }, [])

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
  }
}
