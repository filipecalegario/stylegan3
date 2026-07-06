import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { LatentExplorer } from './pages/LatentExplorer'
import { GeneticEvolution } from './pages/GeneticEvolution'
import { LatentInterpolation } from './pages/LatentInterpolation'
import { WVectorEditor } from './pages/WVectorEditor'
import { GenotypeStrip } from './components/GenotypeStrip'
import './App.css'

function Navigation() {
  const location = useLocation()

  return (
    <nav className="nav-bar">
      <GenotypeStrip count={220} seed={41} height={30} className="nav-ambient" />
      <div className="nav-brand">
        <GenotypeStrip count={14} seed={7} height={26} gap={2.5} className="brand-mark" />
        <span className="brand-name">Latent Studio</span>
        <span className="brand-sub">StyleGAN3</span>
      </div>
      <div className="nav-links">
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
        <Link
          to="/w-editor"
          className={`nav-link ${location.pathname === '/w-editor' ? 'active' : ''}`}
        >
          W Editor
        </Link>
      </div>
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
          <Route path="/w-editor" element={<WVectorEditor />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
