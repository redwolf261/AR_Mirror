import { motion } from 'framer-motion';
import ConfidenceRing from './ConfidenceRing';
import GarmentCanvas from './GarmentCanvas';
import PoseCanvas from './PoseCanvas';
import styles from './CameraCard.module.css';

/**
 * CameraCard — renders live webcam + garment overlay + pose skeleton.
 * Z-index stack (bottom → top):
 *   0  videoBackdrop  (always below everything)
 *   1  <video> / <img> stream (the actual feed)
 *   2  PoseCanvas (skeleton SVG)
 *   3  GarmentCanvas (garment image drawn on canvas)
 *   4  bottomBar / badges (UI chrome)
 */
export default function CameraCard({
  label,
  posePoints,
  isActive,
  confidence,
  instruction,
  streamUrl,
  webcamRef,
  webcamReady,
  webcamError,
  selectedGarment,
}) {
  const garmentSrc = selectedGarment?.texture || null;
  const hasLiveFeed = webcamReady || !!streamUrl;

  return (
    <motion.article
      className={`${styles.card} ${isActive ? styles.active : ''}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      {/* z=0  dark gradient backdrop — always rendered first/lowest */}
      <div className={styles.videoBackdrop} />

      {/* z=1  Live camera feed */}
      {streamUrl ? (
        <img className={styles.stream} src={streamUrl} alt="AR stream" />
      ) : (
        <video
          ref={webcamRef}
          className={styles.stream}
          autoPlay
          playsInline
          muted
        />
      )}

      {/* Camera requesting overlay — only shows before permission granted */}
      {!hasLiveFeed && (
        <div className={styles.cameraOffOverlay}>
          <span>📷</span>
          <p>{webcamError || 'Allow camera access to start live try-on'}</p>
        </div>
      )}

      {/* z=2  Skeleton pose overlay (SVG) */}
      <PoseCanvas posePoints={posePoints} />

      {/* z=3  Garment AR overlay (canvas) */}
      {garmentSrc && (
        <GarmentCanvas
          posePoints={posePoints}
          garmentSrc={garmentSrc}
          videoRef={webcamRef}
          visible
        />
      )}

      {/* Badges */}
      <div className={styles.badge}>{label}</div>
      {selectedGarment && (
        <div className={styles.garmentBadge}>
          👗 {selectedGarment.name}
        </div>
      )}

      {/* Bottom HUD bar */}
      <div className={styles.bottomBar}>
        <motion.div
          className={styles.instructionPill}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          {instruction}
        </motion.div>
        <ConfidenceRing value={confidence} />
      </div>
    </motion.article>
  );
}
