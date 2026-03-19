import React from 'react';

/**
 * MetricCard Component - Individual measurement display
 * Apple-style: Clean card with hover effect, subtle transitions
 */
const MetricCard = ({ label, value, unit }) => {
  // Handle null/undefined values gracefully
  const displayValue = value != null ? value.toFixed(1) : '--';

  return (
    <div className="metric-card scale-in">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        <span>{displayValue}</span>
        {value != null && <span className="metric-unit">{unit}</span>}
      </div>
    </div>
  );
};

export default MetricCard;
