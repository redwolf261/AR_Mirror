import React from 'react';
import MetricCard from './MetricCard';
import GarmentSelector from './GarmentSelector';
import Controls from './Controls';

/**
 * MeasurementsPanel Component - Right sidebar
 * Displays body measurements and controls in a clean, organized manner
 *
 * Design: Apple-style refinement - subtle separators, consistent spacing
 */
const MeasurementsPanel = ({
  measurements,
  fps,
  garment,
  garments,
  selectedGarment,
  onGarmentChange,
  showSkeleton,
  setShowSkeleton,
  showMeasurements,
  setShowMeasurements,
}) => {
  return (
    <div className="measurements-panel">
      {/* Measurements Section */}
      <div className="panel-section">
        <h3 className="panel-title">Body Measurements</h3>

        {measurements ? (
          <div className="metric-grid">
            <MetricCard
              label="Shoulder"
              value={measurements.shoulder_cm}
              unit="cm"
            />
            <MetricCard
              label="Chest"
              value={measurements.chest_cm}
              unit="cm"
            />
            <MetricCard
              label="Waist"
              value={measurements.waist_cm}
              unit="cm"
            />
            <MetricCard
              label="Torso"
              value={measurements.torso_cm}
              unit="cm"
            />
          </div>
        ) : (
          <div
            style={{
              padding: '32px',
              textAlign: 'center',
              color: '#86868b',
              fontSize: '13px',
            }}
          >
            Detecting body...
          </div>
        )}

        {/* fit size indicator */}
        {measurements && measurements.size && (
          <div
            style={{
              marginTop: '16px',
              padding: '12px 16px',
              background: 'rgba(0, 113, 227, 0.08)',
              borderRadius: '12px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#0071e3',
              textAlign: 'center',
            }}
          >
            Recommended Size: {measurements.size}
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="panel-section">
        <h3 className="panel-title">Performance</h3>

        <div className="fps-display">
          <span className="fps-label">Frame Rate</span>
          <span className="fps-value">{fps.toFixed(1)} FPS</span>
        </div>
      </div>

      {/* Garment Selection */}
      {garments && garments.length > 0 && (
        <div className="panel-section">
          <h3 className="panel-title">Garments</h3>
          <GarmentSelector
            garments={garments}
            selected={selectedGarment}
            onSelect={onGarmentChange}
          />
        </div>
      )}

      {/* Controls */}
      <div className="panel-section">
        <h3 className="panel-title">Display Options</h3>
        <Controls
          showSkeleton={showSkeleton}
          setShowSkeleton={setShowSkeleton}
          showMeasurements={showMeasurements}
          setShowMeasurements={setShowMeasurements}
        />
      </div>
    </div>
  );
};

export default MeasurementsPanel;
