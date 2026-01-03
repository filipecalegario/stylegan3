import { useRef, useState } from 'react';
import './DNAUploader.css';

export function DNAUploader({
  dnaFiles,
  onAddFile,
  onRemoveFile,
  onReorder,
  onClear,
  isLoading
}) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [dragIndex, setDragIndex] = useState(null);

  const handleFileSelect = async (files) => {
    for (const file of files) {
      if (file.name.endsWith('.json')) {
        try {
          await onAddFile(file);
        } catch (err) {
          alert(`Error loading ${file.name}: ${err.message}`);
        }
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDragStart = (e, index) => {
    setDragIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOverItem = (e, index) => {
    e.preventDefault();
    if (dragIndex !== null && dragIndex !== index) {
      onReorder(dragIndex, index);
      setDragIndex(index);
    }
  };

  const handleDragEnd = () => {
    setDragIndex(null);
  };

  return (
    <div className="dna-uploader">
      <div className="uploader-header">
        <h3>DNA Sequence</h3>
        <span className="file-count">{dnaFiles.length} files</span>
      </div>

      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          multiple
          onChange={(e) => handleFileSelect(e.target.files)}
          style={{ display: 'none' }}
        />
        <div className="drop-icon">+</div>
        <p>Drop DNA files here or click to upload</p>
        <span className="hint">JSON files exported from Genetic Evolution</span>
      </div>

      {dnaFiles.length > 0 && (
        <div className="dna-list">
          {dnaFiles.map((dna, index) => (
            <div
              key={dna.id}
              className={`dna-item ${dragIndex === index ? 'dragging' : ''}`}
              draggable={!isLoading}
              onDragStart={(e) => handleDragStart(e, index)}
              onDragOver={(e) => handleDragOverItem(e, index)}
              onDragEnd={handleDragEnd}
            >
              <span className="dna-index">{index + 1}</span>
              <div className="dna-info">
                <span className="dna-filename">{dna.filename}</span>
                <span className="dna-meta">
                  {dna.generation !== undefined && `Gen ${dna.generation}`}
                  {dna.fitness !== undefined && ` | Fit: ${dna.fitness.toFixed(1)}`}
                </span>
              </div>
              <button
                className="remove-btn"
                onClick={() => onRemoveFile(dna.id)}
                disabled={isLoading}
                title="Remove"
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {dnaFiles.length > 0 && (
        <div className="uploader-actions">
          <button
            className="clear-btn"
            onClick={onClear}
            disabled={isLoading}
          >
            Clear All
          </button>
        </div>
      )}

      {dnaFiles.length > 0 && dnaFiles.length < 2 && (
        <div className="warning">
          Add at least 2 DNA files to create an interpolation
        </div>
      )}
    </div>
  );
}
