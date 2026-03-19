import React from 'react';

/**
 * Controls Component - Toggle switches for display options
 * Apple-style iOS toggle switches with smooth animations
 */
const Controls = ({
  showSkeleton,
  setShowSkeleton,
  showMeasurements,
  setShowMeasurements,
}) => {
  return (
    <div className="control-group">
      {/* Show Skeleton Toggle */}
      <div
        className="toggle"
        onClick={() => setShowSkeleton(!showSkeleton)}
      >
        <span className="toggle-label">Show Pose Skeleton</span>
        <div className={`toggle-switch ${showSkeleton ? 'active' : ''}`}>
          <div className="toggle-switch-handle" />
        </div>
      </div>

      {/* Show Measurements Toggle */}
      <div
        className="toggle"
        onClick={() => setShowMeasurements(!showMeasurements)}
        style={{ marginTop: '8px' }}
      >
        <span className="toggle-label">Show Measurements</span>
        <div className={`toggle-switch ${showMeasurements ? 'active' : ''}`}>
          <div className="toggle-switch-handle" />
        </div>
      </div>
    </div>
  );
};

export default Controls;
