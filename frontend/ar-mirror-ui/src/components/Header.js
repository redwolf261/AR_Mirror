import React from 'react';

/**
 * Header Component - Clean top navigation
 * Apple-style: Minimal, translucent background, subtle blur
 */
const Header = ({ connected, fps, fitEngineEnabled, inFitEngineMode, onToggleFitEngine }) => {
  return (
    <header className="header">
      <div className="header-title">
        <span>AR Mirror</span>
        {connected && <span className="status-dot" />}
      </div>

      <div className="header-status">
        {fitEngineEnabled && (
          <button
            type="button"
            onClick={onToggleFitEngine}
            style={{
              border: '1px solid rgba(120, 120, 128, 0.28)',
              borderRadius: 16,
              padding: '4px 10px',
              background: inFitEngineMode ? 'rgba(0, 113, 227, 0.1)' : '#fff',
              color: inFitEngineMode ? '#0071e3' : '#515154',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {inFitEngineMode ? 'AR Mode' : 'FitEngine'}
          </button>
        )}
        {connected ? (
          <>
            <span style={{ color: '#34c759'}}>● Live</span>
            <span style={{ color: '#86868b' }}>•</span>
            <span>{(fps ?? 0).toFixed(1)} FPS</span>
          </>
        ) : (
          <span style={{ color: '#ff3b30' }}>Disconnected</span>
        )}
      </div>
    </header>
  );
};

export default Header;
