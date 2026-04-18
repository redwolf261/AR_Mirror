import { motion } from 'framer-motion';
import ConfidenceRing from './ConfidenceRing';
import styles from './SizeReveal.module.css';

export default function SizeReveal({ size, confidence, label }) {
  return (
    <div className={styles.wrapper}>
      <p className={styles.kicker}>Size</p>
      <div className={styles.core}>
        <motion.div
          className={styles.letter}
          initial={{ scale: 0.3, opacity: 0, rotateX: -30 }}
          animate={{ scale: 1, opacity: 1, rotateX: 0 }}
          transition={{ type: 'spring', stiffness: 200, damping: 18 }}
        >
          {size}
        </motion.div>
        <div className={styles.ringWrap}>
          <ConfidenceRing value={confidence} size={96} stroke={7} />
          <span>{label}</span>
        </div>
      </div>
    </div>
  );
}
