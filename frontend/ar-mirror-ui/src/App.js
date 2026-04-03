import React, { useState, useEffect } from 'react';
import './App.css';
import Header from './components/Header';
import FitEngineMode from './components/FitEngineMode';
import config from './config';

/**
 * AR Mirror - Apple-Inspired Body Tracking Interface
 *
 * Design Philosophy:
 * - Clean, minimal, elegant (Apple.com aesthetic)
 * - Generous whitespace
 * - Subtle animations
 * - Focus on clarity and typography
 */

function App() {
  // Backend API configuration
  const API_BASE = config.API_BASE_URL;

  // State management
  const [systemState, setSystemState] = useState({
    fps: 0,
    garment: '',
    measurements: null,
    connected: false,
  });

  /**
   * Fetch system state from backend
   * Polls every 100ms for smooth updates
   */
  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/state`);
        if (response.ok) {
          const data = await response.json();
          setSystemState(prev => ({
            ...data,
            connected: true,
          }));
        }
      } catch (error) {
        setSystemState(prev => ({ ...prev, connected: false }));
      }
    };

    // Initial fetch
    fetchState();

    // Poll for updates
    const interval = setInterval(fetchState, 100);

    return () => clearInterval(interval);
  }, [API_BASE]);

  return (
    <div className="app">
      <Header
        connected={systemState.connected}
        fps={systemState.fps}
        fitEngineEnabled={false}
        inFitEngineMode
        onToggleFitEngine={null}
      />

      <FitEngineMode
        apiBase={API_BASE}
        state={systemState}
        showExit={false}
      />
    </div>
  );
}

export default App;
