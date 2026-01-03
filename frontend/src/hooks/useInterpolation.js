import { useState, useCallback, useRef } from 'react';

const API_BASE = 'http://127.0.0.1:8000';
const WS_BASE = 'ws://127.0.0.1:8000';

export function useInterpolation() {
  const [dnaFiles, setDnaFiles] = useState([]);
  const [config, setConfig] = useState({
    fps: 30,
    frames_per_transition: 60,
    interpolation_kind: 'cubic',
    loop: true,
    image_size: 512
  });
  const [previewFrames, setPreviewFrames] = useState([]);
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);

  const addDnaFile = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);

          // Validate DNA structure
          if (!data.w_vector || !Array.isArray(data.w_vector)) {
            throw new Error('Invalid DNA file: missing w_vector');
          }

          if (data.w_vector.length !== 512) {
            throw new Error(`Invalid DNA file: w_vector should have 512 values, got ${data.w_vector.length}`);
          }

          const dnaEntry = {
            id: data.id || `dna_${Date.now()}`,
            filename: file.name,
            w_vector: data.w_vector,
            fitness: data.fitness,
            generation: data.generation
          };

          setDnaFiles(prev => [...prev, dnaEntry]);
          resolve(dnaEntry);
        } catch (err) {
          reject(err);
        }
      };

      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }, []);

  const removeDnaFile = useCallback((id) => {
    setDnaFiles(prev => prev.filter(f => f.id !== id));
  }, []);

  const reorderDnaFiles = useCallback((fromIndex, toIndex) => {
    setDnaFiles(prev => {
      const newFiles = [...prev];
      const [removed] = newFiles.splice(fromIndex, 1);
      newFiles.splice(toIndex, 0, removed);
      return newFiles;
    });
  }, []);

  const clearDnaFiles = useCallback(() => {
    setDnaFiles([]);
    setPreviewFrames([]);
    setResult(null);
  }, []);

  const updateConfig = useCallback((updates) => {
    setConfig(prev => ({ ...prev, ...updates }));
  }, []);

  const generatePreview = useCallback(async (model) => {
    if (dnaFiles.length < 2) {
      setError('Need at least 2 DNA files for interpolation');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/interpolation/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          w_vectors: dnaFiles.map(d => d.w_vector),
          num_frames: 10,
          interpolation_kind: config.interpolation_kind,
          loop: config.loop,
          image_size: 256
        })
      });

      if (!response.ok) {
        throw new Error(`Preview failed: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Convert relative URLs to absolute
      const frames = data.frames.map(url => `${API_BASE}${url}`);
      setPreviewFrames(frames);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [dnaFiles, config]);

  const generateVideo = useCallback((model) => {
    if (dnaFiles.length < 2) {
      setError('Need at least 2 DNA files for interpolation');
      return;
    }

    setIsLoading(true);
    setError(null);
    setProgress({ value: 0, message: 'Connecting...' });
    setResult(null);

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_BASE}/ws/interpolation`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        action: 'generate',
        params: {
          model,
          w_vectors: dnaFiles.map(d => d.w_vector),
          ...config
        }
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setProgress({
          value: data.value,
          message: data.message
        });
      } else if (data.type === 'complete') {
        setResult({
          ...data,
          video_url: `${API_BASE}${data.video_url}`,
          gif_url: `${API_BASE}${data.gif_url}`
        });
        setIsLoading(false);
        setProgress(null);
      } else if (data.type === 'error') {
        setError(data.message);
        setIsLoading(false);
        setProgress(null);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection failed');
      setIsLoading(false);
      setProgress(null);
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, [dnaFiles, config]);

  const cancelGeneration = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsLoading(false);
    setProgress(null);
  }, []);

  return {
    // State
    dnaFiles,
    config,
    previewFrames,
    result,
    progress,
    isLoading,
    error,

    // DNA file management
    addDnaFile,
    removeDnaFile,
    reorderDnaFiles,
    clearDnaFiles,

    // Config
    updateConfig,

    // Generation
    generatePreview,
    generateVideo,
    cancelGeneration
  };
}
