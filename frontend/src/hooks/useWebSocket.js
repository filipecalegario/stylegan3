import { useState, useRef, useCallback } from 'react'

function useWebSocket() {
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const wsRef = useRef(null)

  const connect = useCallback(() => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close()
    }

    setProgress(null)
    setResult(null)
    setError(null)
    setIsGenerating(true)

    // In development, connect directly to backend; in production, use same host
    const isDev = import.meta.env.DEV
    const wsUrl = isDev
      ? 'ws://127.0.0.1:8000/ws/generate'
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/generate`

    console.log('Connecting to WebSocket:', wsUrl)
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'progress':
          setProgress({
            value: data.value,
            message: data.message,
          })
          break

        case 'complete':
          setResult({
            image_url: data.image_url,
            seed: data.seed,
            time: data.time,
            num_samples: data.num_samples,
            layout_mode: data.layout_mode,
          })
          setProgress(null)
          setIsGenerating(false)
          break

        case 'error':
          setError(data.message)
          setProgress(null)
          setIsGenerating(false)
          break

        default:
          console.warn('Unknown message type:', data.type)
      }
    }

    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      setError('WebSocket connection error')
      setIsGenerating(false)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
    }
  }, [])

  const send = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.error('WebSocket not connected')
      setError('WebSocket not connected')
    }
  }, [])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsGenerating(false)
  }, [])

  return {
    connect,
    send,
    disconnect,
    progress,
    result,
    error,
    isGenerating,
  }
}

export default useWebSocket
