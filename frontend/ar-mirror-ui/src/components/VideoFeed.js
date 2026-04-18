import React, { useRef, useEffect, useState } from 'react';

/**
 * VideoFeed Component - Main viewport
 * Displays MJPEG stream from backend with optional overlays
 *
 * Design: Centered, rounded corners, subtle shadow, smooth hover effect
 */
const VideoFeed = ({ apiBase, showSkeleton, showMeasurements }) => {
  const imgRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (imgRef.current) {
      // Add timestamp to prevent caching
      const streamUrl = `${apiBase}/stream?t=${Date.now()}`;
      imgRef.current.src = streamUrl;

      imgRef.current.onload = () => {
        setIsLoading(false);
      };

      imgRef.current.onerror = () => {
        setIsLoading(true);
        // Retry after 2 seconds
        setTimeout(() => {
          if (imgRef.current) {
            imgRef.current.src = `${apiBase}/stream?t=${Date.now()}`;
          }
        }, 2000);
      };
    }
  }, [apiBase]);

  return (
    <div className="video-section">
      <div className="video-container">
        {isLoading && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: '#86868b',
              fontSize: '15px',
              fontWeight: 500,
            }}
          >
            Connecting to camera...
          </div>
        )}

        {/* MJPEG Stream */}
        <img
          ref={imgRef}
          className={`video-stream ${isLoading ? '' : 'fade-in'}`}
          alt="AR Mirror Stream"
          style={{ opacity: isLoading ? 0 : 1 }}
        />

        {/* Pose Overlay (if needed for additional UI elements) */}
        {!isLoading && (
          <div className="pose-overlay">
            {/* Additional overlay elements can go here */}
            {/* For now, skeleton is rendered on the backend */}
          </div>
        )}
      </div>

      {/* Subtle caption */}
      <div
        style={{
          marginTop: '24px',
          fontSize: '13px',
          color: '#86868b',
          textAlign: 'center',
        }}
      >
        {showSkeleton ? 'Pose tracking active' : 'Pose tracking hidden'}
      </div>
    </div>
  );
};

export default VideoFeed;
