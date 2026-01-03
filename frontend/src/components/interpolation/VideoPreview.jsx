import { useState, useEffect } from 'react';
import './VideoPreview.css';

export function VideoPreview({
  previewFrames,
  result,
  progress,
  isLoading
}) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play preview frames
  useEffect(() => {
    if (previewFrames.length > 0 && isPlaying) {
      const interval = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % previewFrames.length);
      }, 200);
      return () => clearInterval(interval);
    }
  }, [previewFrames, isPlaying]);

  // Reset frame when new preview loads
  useEffect(() => {
    setCurrentFrame(0);
    if (previewFrames.length > 0) {
      setIsPlaying(true);
    }
  }, [previewFrames]);

  if (isLoading && progress) {
    return (
      <div className="video-preview loading">
        <div className="progress-container">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress.value * 100}%` }}
            />
          </div>
          <p className="progress-message">{progress.message}</p>
        </div>
      </div>
    );
  }

  if (result) {
    return (
      <div className="video-preview has-result">
        <div className="result-container">
          <video
            src={result.video_url}
            controls
            autoPlay
            loop
            className="result-video"
          />

          <div className="result-info">
            <div className="info-row">
              <span className="label">Duration:</span>
              <span className="value">{result.duration}s</span>
            </div>
            <div className="info-row">
              <span className="label">Frames:</span>
              <span className="value">{result.total_frames}</span>
            </div>
            <div className="info-row">
              <span className="label">Keyframes:</span>
              <span className="value">{result.keyframes}</span>
            </div>
          </div>

          <div className="download-buttons">
            <a
              href={result.video_url}
              download
              className="download-btn primary"
            >
              Download MP4
            </a>
            <a
              href={result.gif_url}
              download
              className="download-btn secondary"
            >
              Download GIF
            </a>
          </div>
        </div>
      </div>
    );
  }

  if (previewFrames.length > 0) {
    return (
      <div className="video-preview has-preview">
        <div className="preview-container">
          <img
            src={previewFrames[currentFrame]}
            alt={`Preview frame ${currentFrame + 1}`}
            className="preview-image"
          />

          <div className="preview-controls">
            <button
              className="play-btn"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? '||' : '>'}
            </button>
            <div className="frame-indicator">
              Frame {currentFrame + 1} / {previewFrames.length}
            </div>
          </div>

          <div className="frame-dots">
            {previewFrames.map((_, idx) => (
              <button
                key={idx}
                className={`dot ${idx === currentFrame ? 'active' : ''}`}
                onClick={() => {
                  setCurrentFrame(idx);
                  setIsPlaying(false);
                }}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="video-preview empty">
      <div className="placeholder">
        <div className="placeholder-icon">~</div>
        <p>Upload DNA files and click Preview</p>
        <span>to see interpolation frames</span>
      </div>
    </div>
  );
}
