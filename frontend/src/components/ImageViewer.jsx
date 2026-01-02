import { useState, useRef, useEffect } from 'react'

function ImageViewer({ imageUrl, onImageClick }) {
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const containerRef = useRef(null)
  const imageRef = useRef(null)

  // Reset zoom/pan when image changes
  useEffect(() => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }, [imageUrl])

  const handleWheel = (e) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    const newScale = Math.min(Math.max(scale * delta, 0.1), 10)

    // Zoom towards mouse position
    const rect = containerRef.current.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const newX = mouseX - (mouseX - position.x) * (newScale / scale)
    const newY = mouseY - (mouseY - position.y) * (newScale / scale)

    setScale(newScale)
    setPosition({ x: newX, y: newY })
  }

  const handleMouseDown = (e) => {
    if (e.button === 0) { // Left click
      setIsDragging(true)
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      })
    }
  }

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      })
    }
  }

  const handleMouseUp = (e) => {
    if (isDragging) {
      setIsDragging(false)
    }
  }

  const handleClick = (e) => {
    // Only trigger if not dragging
    if (!isDragging && onImageClick) {
      onImageClick(e, imageUrl)
    }
  }

  const handleZoomIn = () => {
    setScale(Math.min(scale * 1.25, 10))
  }

  const handleZoomOut = () => {
    setScale(Math.max(scale * 0.8, 0.1))
  }

  const handleReset = () => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }

  return (
    <div
      className="image-viewer"
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <img
        ref={imageRef}
        src={imageUrl}
        alt="Latent Space Map"
        onClick={handleClick}
        draggable={false}
        style={{
          transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
        }}
      />

      <div className="zoom-controls">
        <button className="zoom-btn" onClick={handleZoomIn} title="Zoom In">
          +
        </button>
        <button className="zoom-btn" onClick={handleZoomOut} title="Zoom Out">
          −
        </button>
        <button className="zoom-btn" onClick={handleReset} title="Reset">
          ⟲
        </button>
      </div>
    </div>
  )
}

export default ImageViewer
