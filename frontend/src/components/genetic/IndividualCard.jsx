import { useState } from 'react';
import './IndividualCard.css';

export function IndividualCard({
  individual,
  imageUrl,
  fitness,
  onFitnessChange,
  onAddFavorite,
  isLoading
}) {
  const [isHovered, setIsHovered] = useState(false);

  const handleSliderChange = (e) => {
    const value = parseFloat(e.target.value);
    onFitnessChange(individual.id, value);
  };

  const handleFavoriteClick = () => {
    onAddFavorite(individual.id);
  };

  return (
    <div
      className={`individual-card ${isHovered ? 'hovered' : ''} ${isLoading ? 'loading' : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="individual-image-container">
        <img
          src={imageUrl}
          alt={`Individual ${individual.id}`}
          className="individual-image"
        />
        {isLoading && <div className="loading-overlay">Loading...</div>}
      </div>

      <div className="individual-controls">
        <div className="fitness-slider-container">
          <input
            type="range"
            min="0"
            max="10"
            step="0.5"
            value={fitness}
            onChange={handleSliderChange}
            className="fitness-slider"
            disabled={isLoading}
          />
          <span className="fitness-value">{fitness.toFixed(1)}</span>
        </div>

        <div className="individual-actions">
          <button
            className="favorite-button"
            onClick={handleFavoriteClick}
            title="Add to favorites"
            disabled={isLoading}
          >
            *
          </button>
        </div>
      </div>

      <div className="individual-info">
        <span className="individual-id">#{individual.id}</span>
        {individual.generation > 0 && (
          <span className="individual-gen">Gen {individual.generation}</span>
        )}
      </div>
    </div>
  );
}
