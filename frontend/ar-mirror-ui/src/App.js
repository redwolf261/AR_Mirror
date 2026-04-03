import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import Header from './components/Header';
import FitEngineMode from './components/FitEngineMode';
import GarmentOverlayMode from './components/GarmentOverlayMode';
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

  const [viewMode, setViewMode] = useState('fitengine');
  const [garments, setGarments] = useState([]);
  const [selectedGarment, setSelectedGarment] = useState('');

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

  useEffect(() => {
    const fetchGarments = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/garments`);
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        const items = data.garments || [];
        setGarments(items);
        if (!selectedGarment && items.length > 0) {
          setSelectedGarment(typeof items[0] === 'string' ? items[0] : (items[0].sku || items[0].name || items[0].file || ''));
        }
      } catch (_err) {
        // Non-blocking.
      }
    };

    fetchGarments();
  }, [API_BASE, selectedGarment]);

  const handleGarmentChange = useCallback(async (garmentId) => {
    if (!garmentId) return;
    try {
      const response = await fetch(`${API_BASE}/api/garment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: garmentId }),
      });
      if (response.ok) {
        setSelectedGarment(garmentId);
      }
    } catch (_err) {
      // Non-blocking.
    }
  }, [API_BASE]);

  useEffect(() => {
    const updateRenderMode = async () => {
      try {
        const overlayEnabled = viewMode === 'overlay';
        await fetch(`${API_BASE}/api/params`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            render_tryon_overlay: overlayEnabled,
            show_garment_box: false,
            show_skeleton: false,
          }),
        });
      } catch (_err) {
        // Non-blocking visual preference update.
      }
    };

    updateRenderMode();
  }, [API_BASE, viewMode]);

  return (
    <div className="app">
      <Header
        connected={systemState.connected}
        fps={systemState.fps}
        fitEngineEnabled={false}
        inFitEngineMode={viewMode === 'fitengine'}
        onToggleFitEngine={null}
      />

      {viewMode === 'fitengine' ? (
        <FitEngineMode
          apiBase={API_BASE}
          state={systemState}
          showExit={false}
          onProceedToOverlay={() => setViewMode('overlay')}
        />
      ) : (
        <GarmentOverlayMode
          apiBase={API_BASE}
          fps={systemState.fps}
          garments={garments}
          selectedGarment={selectedGarment}
          onGarmentChange={handleGarmentChange}
          onBack={() => setViewMode('fitengine')}
        />
      )}
    </div>
  );
}

export default App;
