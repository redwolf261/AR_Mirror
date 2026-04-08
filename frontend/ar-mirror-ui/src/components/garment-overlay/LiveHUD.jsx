import { motion } from 'framer-motion';
import FPSCounter from './FPSCounter';
import GarmentSelector from './GarmentSelector';
import styles from './LiveHUD.module.css';

export default function LiveHUD({ fps, garments, selectedGarment, onSelect }) {
  return (
    <div className={styles.hud}>
      <motion.div className={styles.livePill} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}>
        <span className={styles.dot} /> Live Try-On <FPSCounter fps={fps} /> FPS
      </motion.div>

      <div className={styles.topRight}>
        <GarmentSelector garments={garments} selectedId={selectedGarment.id} onSelect={onSelect} />
      </div>

      <motion.div
        className={styles.bottomShelf}
        initial={{ y: 80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 190, damping: 21 }}
      >
        <div>
          <p className={styles.garmentName}>{selectedGarment.name}</p>
          <span className={styles.size}>Size {selectedGarment.size}</span>
        </div>

        <div className={styles.meter}>
          <p>Fit Confidence</p>
          <div className={styles.track}><div className={styles.fill} /></div>
        </div>

        <div className={styles.actions}>
          <button>Save Look</button>
          <button>Share</button>
          <button>Swap</button>
        </div>
      </motion.div>
    </div>
  );
}
