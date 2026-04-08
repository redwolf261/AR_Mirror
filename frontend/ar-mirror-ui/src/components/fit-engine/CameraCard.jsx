import { motion } from 'framer-motion';
import ConfidenceRing from './ConfidenceRing';
import PoseCanvas from './PoseCanvas';
import styles from './CameraCard.module.css';

export default function CameraCard({
  label,
  posePoints,
  isActive,
  confidence,
  instruction,
}) {
  return (
    <motion.article
      className={`${styles.card} ${isActive ? styles.active : ''}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <div className={styles.badge}>{label}</div>
      <div className={styles.videoBackdrop} />
      <PoseCanvas posePoints={posePoints} />

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
