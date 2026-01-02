function ProgressBar({ value, message }) {
  const percentage = Math.round(value * 100)

  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="progress-message">
        {message} ({percentage}%)
      </div>
    </div>
  )
}

export default ProgressBar
