import React from 'react';
import VideoFeed from './VideoFeed';
import './FitEngineMode.css';

const getGarmentLabel = (garment, index) => {
  if (typeof garment === 'string') return garment;
  return garment?.name || garment?.sku || garment?.file || `Garment ${index + 1}`;
};

const getGarmentId = (garment, index) => {
  if (typeof garment === 'string') return garment;
  return garment?.sku || garment?.file || garment?.name || `Garment ${index + 1}`;
};

const GarmentOverlayMode = ({
  apiBase,
  fps,
  garments,
  selectedGarment,
  onGarmentChange,
  onBack,
}) => {
  return (
    <div className="fitengine-shell">
      <div className="fitengine-topbar">
        <div>
          <h2>Garment Overlay - Try Your Look</h2>
          <p>FitEngine sizing complete. Now preview garments with a clean live overlay.</p>
        </div>
        <button type="button" onClick={onBack} className="fitengine-ghost-btn">Back to FitEngine</button>
      </div>

      <div className="fitengine-height-row">
        <label>Live Try-On</label>
        <div className="fitengine-reference">Live: {(fps ?? 0).toFixed(1)} FPS</div>
      </div>

      <div className="fitengine-workflow">
        <div className="fitengine-workflow-main">
          <section className="fitengine-panel fitengine-panel-capture">
            <h4>Overlay Preview</h4>
            <p>Stand centered and choose a garment to see realistic drape and fit.</p>
            <VideoFeed apiBase={apiBase} showSkeleton={false} showMeasurements={false} />
          </section>
        </div>

        <aside className="fitengine-summary-column">
          <div className="fitengine-result-wrap fitengine-result-wrap-sticky">
            <div className="fitengine-result-card">
              <div className="fitengine-result-head">
                <div>
                  <h4>Garments</h4>
                  <div className="fitengine-result-subtitle">Select a garment to preview</div>
                </div>
              </div>

              <div className="fitengine-garment-grid">
                {(garments || []).slice(0, 12).map((garment, index) => {
                  const label = getGarmentLabel(garment, index);
                  const garmentId = getGarmentId(garment, index);
                  const isSelected = selectedGarment === garmentId;
                  return (
                    <button
                      key={`${garmentId}-${index}`}
                      type="button"
                      className={`fitengine-garment-btn ${isSelected ? 'fitengine-garment-btn-active' : ''}`}
                      onClick={() => onGarmentChange && onGarmentChange(garmentId)}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              <div className="fitengine-summary-note">
                Tip: For most realistic result, keep shoulders and torso fully in frame.
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default GarmentOverlayMode;
