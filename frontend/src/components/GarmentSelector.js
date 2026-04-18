import React from 'react';

/**
 * GarmentSelector Component - Grid of garment thumbnails
 * Apple-style: Clean grid, subtle hover states, active border
 */
const GarmentSelector = ({ garments, selected, onSelect }) => {
  // Get thumbnail URL for a garment
  const getThumbnailUrl = (garmentName) => {
    return `/api/garment_image/${garmentName}`;
  };

  return (
    <div className="garment-selector">
      {garments.map((garment, index) => (
        <div
          key={index}
          className={`garment-item ${selected === garment ? 'active' : ''}`}
          onClick={() => onSelect(garment)}
        >
          <img
            src={getThumbnailUrl(garment)}
            alt={garment}
            className="garment-thumbnail"
            loading="lazy"
            onError={(e) => {
              // Fallback to placeholder on error
              e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="300"%3E%3Crect fill="%23f5f5f7" width="200" height="300"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2386868b" font-size="14" font-family="system-ui"%3ENo Image%3C/text%3E%3C/svg%3E';
            }}
          />
        </div>
      ))}
    </div>
  );
};

export default GarmentSelector;
