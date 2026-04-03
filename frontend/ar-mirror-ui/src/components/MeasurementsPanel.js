import React from 'react';
import MetricCard from './MetricCard';
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
  fitEngineEnabled,
  onStartFitEngine,
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

        {/* Size Recommendation */}
        {measurements && measurements.size_recommendation && (
          <div style={{
            marginTop: '16px',
            padding: '16px',
            background: 'rgba(0, 113, 227, 0.08)',
            borderRadius: '12px',
            border: '1px solid rgba(0, 113, 227, 0.15)'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '8px'
            }}>
              <span style={{
                fontSize: '15px',
                fontWeight: 600,
                color: '#0071e3',
              }}>
                Size {measurements.size_recommendation}
              </span>
              <span style={{
                fontSize: '12px',
                color: '#86868b',
                fontWeight: 500
              }}>
                {measurements.size_confidence}% match
              </span>
            </div>

            {measurements.size_description && (
              <div style={{
                fontSize: '12px',
                color: '#515154',
                lineHeight: '1.4',
                whiteSpace: 'pre-line'
              }}>
                {measurements.size_description.split('\n')[1]} {/* Skip first line, show measurements */}
              </div>
            )}

            {/* Alternative sizes */}
            {measurements.size_alternatives && Object.keys(measurements.size_alternatives).length > 1 && (
              <div style={{ marginTop: '12px' }}>
                <div style={{
                  fontSize: '11px',
                  color: '#86868b',
                  marginBottom: '6px',
                  fontWeight: 500
                }}>
                  Other sizes:
                </div>
                <div style={{
                  display: 'flex',
                  gap: '6px',
                  flexWrap: 'wrap'
                }}>
                  {Object.entries(measurements.size_alternatives)
                    .filter(([size]) => size !== measurements.size_recommendation)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 3)
                    .map(([size, score]) => (
                      <span key={size} style={{
                        padding: '4px 8px',
                        background: score > 0.6 ? 'rgba(52, 199, 89, 0.1)' : 'rgba(142, 142, 147, 0.1)',
                        color: score > 0.6 ? '#34c759' : '#8e8e93',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: 500
                      }}>
                        {size} ({Math.round(score * 100)}%)
                      </span>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="panel-section">
        <h3 className="panel-title">Performance</h3>

        <div className="fps-display">
          <span className="fps-label">Frame Rate</span>
          <span className="fps-value">{(fps ?? 0).toFixed(1)} FPS</span>
        </div>
      </div>

      {/* Garment Selection - NOW ENABLED */}
      {garments && garments.length > 0 && (
        <div className="panel-section">
          <h3 className="panel-title">Garments</h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '8px',
            marginBottom: '12px'
          }}>
            {garments.slice(0, 6).map((garment, index) => (
              <button
                key={`garment-${index}`}
                onClick={() => onGarmentChange && onGarmentChange(typeof garment === 'string' ? garment : (garment.name || garment.file || ''))}
                style={{
                  padding: '12px',
                  border: selectedGarment === (typeof garment === 'string' ? garment : (garment.name || garment.file || ''))
                    ? '2px solid #0071e3'
                    : '1px solid rgba(134, 134, 139, 0.3)',
                  borderRadius: '8px',
                  background: selectedGarment === (typeof garment === 'string' ? garment : (garment.name || garment.file || ''))
                    ? 'rgba(0, 113, 227, 0.08)'
                    : '#f5f5f7',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: selectedGarment === (typeof garment === 'string' ? garment : (garment.name || garment.file || '')) ? 600 : 400,
                  color: selectedGarment === (typeof garment === 'string' ? garment : (garment.name || garment.file || '')) ? '#0071e3' : '#1d1d1f',
                  transition: 'all 0.2s ease',
                  textAlign: 'center'
                }}
                onMouseOver={(e) => {
                  if (selectedGarment !== (typeof garment === 'string' ? garment : (garment.name || garment.file || ''))) {
                    e.target.style.background = '#e8e8ed'
                    e.target.style.borderColor = 'rgba(134, 134, 139, 0.5)'
                  }
                }}
                onMouseOut={(e) => {
                  if (selectedGarment !== (typeof garment === 'string' ? garment : (garment.name || garment.file || ''))) {
                    e.target.style.background = '#f5f5f7'
                    e.target.style.borderColor = 'rgba(134, 134, 139, 0.3)'
                  }
                }}
              >
                {typeof garment === 'string' ? garment : (garment.name || garment.file || `Garment ${index + 1}`)}
              </button>
            ))}
          </div>

          {/* Quick try-on instructions */}
          <div style={{
            padding: '12px',
            background: 'rgba(142, 142, 147, 0.08)',
            borderRadius: '8px',
            fontSize: '11px',
            color: '#86868b',
            textAlign: 'center',
            lineHeight: '1.4'
          }}>
            📍 Stand 1.5-2m from camera for best fit
          </div>
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

      {fitEngineEnabled && (
        <div className="panel-section">
          <h3 className="panel-title">Guided Sizing</h3>
          <button
            type="button"
            onClick={onStartFitEngine}
            style={{
              width: '100%',
              border: 0,
              borderRadius: 10,
              background: '#0071e3',
              color: '#fff',
              padding: '12px 14px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Start 2-Step FitEngine Flow
          </button>
          <div style={{ marginTop: 8, fontSize: 11, color: '#86868b', lineHeight: 1.4 }}>
            Captures front and side views in sequence, then generates size guidance.
          </div>
        </div>
      )}
    </div>
  );
};

export default MeasurementsPanel;
