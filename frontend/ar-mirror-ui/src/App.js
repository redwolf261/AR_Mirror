import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import VideoFeed from './components/VideoFeed';
import MeasurementsPanel from './components/MeasurementsPanel';
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

  const [garments, setGarments] = useState([]);
  const [selectedGarment, setSelectedGarment] = useState('');

  // UI Controls
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showMeasurements, setShowMeasurements] = useState(true);
  const [inFitEngineMode, setInFitEngineMode] = useState(false);

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

  /**
   * Load available garments on mount
   */
  useEffect(() => {
    const fetchGarments = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/garments`);
        if (response.ok) {
          const data = await response.json();
          setGarments(data.garments || []);
          if (data.garments && data.garments.length > 0) {
            setSelectedGarment(data.garments[0]);
          }
        }
      } catch (error) {
        console.error('Failed to fetch garments:', error);
      }
    };

    fetchGarments();
  }, [API_BASE]);

  /**
   * Handle garment selection
   */
  const handleGarmentChange = useCallback(async (garmentName) => {
    try {
      const response = await fetch(`${API_BASE}/api/garment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: garmentName }),
      });

      if (response.ok) {
        setSelectedGarment(garmentName);
      }
    } catch (error) {
      console.error('Failed to change garment:', error);
    }
  }, [API_BASE]);

/**
   * Update backend parameters when UI controls change
   */
  const updateParams = useCallback(async (params) => {
    try {
      await fetch(`${API_BASE}/api/params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
    } catch (error) {
      console.error('Failed to update params:', error);
    }
  }, [API_BASE]);

  // Sync skeleton visibility with backend
  useEffect(() => {
    updateParams({ show_skeleton: showSkeleton });
  }, [showSkeleton, updateParams]);

  return (
    <div className="app">
      <Header
        connected={systemState.connected}
        fps={systemState.fps}
        fitEngineEnabled={config.FITENGINE_MODE_ENABLED}
        inFitEngineMode={inFitEngineMode}
        onToggleFitEngine={() => setInFitEngineMode(prev => !prev)}
      />

      {config.FITENGINE_MODE_ENABLED && inFitEngineMode ? (
        <FitEngineMode
          apiBase={API_BASE}
          state={systemState}
          onExit={() => setInFitEngineMode(false)}
        />
      ) : (
        <div className="main-container">
          {/* Main Video Section - Centered Focus */}
          <VideoFeed
            apiBase={API_BASE}
            showSkeleton={showSkeleton}
            showMeasurements={showMeasurements}
          />

          {/* Measurements Panel - Clean Side Panel */}
          <MeasurementsPanel
            measurements={systemState.measurements}
            fps={systemState.fps}
            garment={systemState.garment}
            garments={garments}
            selectedGarment={selectedGarment}
            onGarmentChange={handleGarmentChange}
            showSkeleton={showSkeleton}
            setShowSkeleton={setShowSkeleton}
            showMeasurements={showMeasurements}
            setShowMeasurements={setShowMeasurements}
            fitEngineEnabled={config.FITENGINE_MODE_ENABLED}
            onStartFitEngine={() => setInFitEngineMode(true)}
          />
        </div>
      )}
    </div>
  );
}

export default App;
