import { useRef, useEffect, useCallback } from 'react'
import './WVectorCanvas.css'

export function WVectorCanvas({ wVector, onUpdateValues, onMouseUp }) {
  const canvasRef = useRef(null)
  const isPaintingRef = useRef(false)

  const WIDTH = 2048  // Canvas width (full width for wider bars)
  const HEIGHT = 350  // Canvas height
  const BAR_WIDTH = WIDTH / 512  // ~4px per bar
  const MID_Y = HEIGHT / 2

  // Draw the bars
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')

    // Clear background
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, WIDTH, HEIGHT)

    // Draw grid lines
    ctx.strokeStyle = '#252540'
    ctx.lineWidth = 1

    // Horizontal grid lines at 25%, 50%, 75%
    for (let y of [HEIGHT * 0.25, HEIGHT * 0.75]) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(WIDTH, y)
      ctx.stroke()
    }

    // Center line (zero)
    ctx.strokeStyle = '#444'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, MID_Y)
    ctx.lineTo(WIDTH, MID_Y)
    ctx.stroke()

    // Draw each bar
    for (let i = 0; i < 512; i++) {
      const value = wVector[i]
      const x = i * BAR_WIDTH

      // Bar height proportional to value
      const barHeight = (Math.abs(value) / 10) * (HEIGHT / 2 - 10)

      if (value > 0) {
        // Green bar going up
        const gradient = ctx.createLinearGradient(0, MID_Y - barHeight, 0, MID_Y)
        gradient.addColorStop(0, '#00ff88')
        gradient.addColorStop(1, '#00aa55')
        ctx.fillStyle = gradient
        ctx.fillRect(x, MID_Y - barHeight, BAR_WIDTH - 0.5, barHeight)
      } else if (value < 0) {
        // Red/orange bar going down
        const gradient = ctx.createLinearGradient(0, MID_Y, 0, MID_Y + barHeight)
        gradient.addColorStop(0, '#aa5500')
        gradient.addColorStop(1, '#ff4444')
        ctx.fillStyle = gradient
        ctx.fillRect(x, MID_Y, BAR_WIDTH - 0.5, barHeight)
      }
    }
  }, [wVector])

  // Calculate index and value from mouse position
  const getValueFromMouse = useCallback((e) => {
    const canvas = canvasRef.current
    if (!canvas) return { index: 0, value: 0 }

    const rect = canvas.getBoundingClientRect()
    const scaleX = WIDTH / rect.width
    const scaleY = HEIGHT / rect.height

    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const index = Math.floor(x / BAR_WIDTH)
    const clampedIndex = Math.max(0, Math.min(511, index))

    // Y to value: middle = 0, top = +10, bottom = -10
    const value = ((MID_Y - y) / (HEIGHT / 2)) * 10
    const clampedValue = Math.max(-10, Math.min(10, value))

    return { index: clampedIndex, value: clampedValue }
  }, [])

  const handleMouseDown = useCallback((e) => {
    isPaintingRef.current = true
    const { index, value } = getValueFromMouse(e)
    onUpdateValues([{ index, value }])
  }, [getValueFromMouse, onUpdateValues])

  const handleMouseMove = useCallback((e) => {
    if (!isPaintingRef.current) return
    const { index, value } = getValueFromMouse(e)
    onUpdateValues([{ index, value }])
  }, [getValueFromMouse, onUpdateValues])

  const handleMouseUp = useCallback(() => {
    if (isPaintingRef.current) {
      isPaintingRef.current = false
      onMouseUp()  // Trigger image generation
    }
  }, [onMouseUp])

  const handleMouseLeave = useCallback(() => {
    if (isPaintingRef.current) {
      isPaintingRef.current = false
      onMouseUp()
    }
  }, [onMouseUp])

  return (
    <div className="wvector-canvas-container">
      <div className="axis-labels">
        <span className="label-top">+10</span>
        <span className="label-mid">0</span>
        <span className="label-bottom">-10</span>
      </div>
      <canvas
        ref={canvasRef}
        width={WIDTH}
        height={HEIGHT}
        className="wvector-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      <div className="dimension-labels">
        <span>0</span>
        <span>128</span>
        <span>256</span>
        <span>384</span>
        <span>512</span>
      </div>
    </div>
  )
}
