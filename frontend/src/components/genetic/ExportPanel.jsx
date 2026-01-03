import './ExportPanel.css';

const API_BASE = 'http://127.0.0.1:8000';

export function ExportPanel({
  favorites,
  onRemove,
  onDownloadJson,
  onDownloadImage
}) {
  if (favorites.length === 0) {
    return (
      <div className="export-panel empty">
        <h3>Favorites</h3>
        <p className="empty-message">
          Click the star button on images to save favorites
        </p>
      </div>
    );
  }

  return (
    <div className="export-panel">
      <h3>Favorites ({favorites.length})</h3>
      <div className="favorites-list">
        {favorites.map((fav) => (
          <div key={fav.id} className="favorite-item">
            <img
              src={`${API_BASE}/api/genetic/image/${fav.id}.png`}
              alt={`Favorite ${fav.id}`}
              className="favorite-thumb"
            />
            <div className="favorite-info">
              <span className="favorite-id">#{fav.id}</span>
              <span className="favorite-gen">Gen {fav.generation}</span>
            </div>
            <div className="favorite-actions">
              <button
                className="download-btn"
                onClick={() => onDownloadImage(fav)}
                title="Download PNG"
              >
                PNG
              </button>
              <button
                className="download-btn"
                onClick={() => onDownloadJson(fav)}
                title="Download DNA (JSON)"
              >
                DNA
              </button>
              <button
                className="remove-btn"
                onClick={() => onRemove(fav.id)}
                title="Remove from favorites"
              >
                X
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
