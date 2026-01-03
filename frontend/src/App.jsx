import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { LatentExplorer } from './pages/LatentExplorer'
import { GeneticEvolution } from './pages/GeneticEvolution'
import { LatentInterpolation } from './pages/LatentInterpolation'
import './App.css'

function Navigation() {
  const location = useLocation()

  return (
    <nav className="nav-bar">
      <Link
        to="/"
        className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
      >
        Latent Explorer
      </Link>
      <Link
        to="/genetic"
        className={`nav-link ${location.pathname === '/genetic' ? 'active' : ''}`}
      >
        Genetic Evolution
      </Link>
      <Link
        to="/interpolation"
        className={`nav-link ${location.pathname === '/interpolation' ? 'active' : ''}`}
      >
        Interpolation
      </Link>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <Routes>
          <Route path="/" element={<LatentExplorer />} />
          <Route path="/genetic" element={<GeneticEvolution />} />
          <Route path="/interpolation" element={<LatentInterpolation />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
