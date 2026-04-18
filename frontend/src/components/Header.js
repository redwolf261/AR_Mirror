import React from 'react';

/**
 * Header Component - Clean top navigation
 * Apple-style: Minimal, translucent background, subtle blur
 */
const Header = ({ connected, fps }) => {
  return (
    <header className="header">
      <div className="header-title">
        <span>AR Mirror</span>
        {connected && <span className="status-dot" />}
      </div>

      <div className="header-status">
        {connected ? (
          <>
            <span style={{ color: '#34c759'}}>● Live</span>
            <span style={{ color: '#86868b' }}>•</span>
            <span>{fps.toFixed(1)} FPS</span>
          </>
        ) : (
          <span style={{ color: '#ff3b30' }}>Disconnected</span>
        )}
      </div>
    </header>
  );
};

export default Header;
