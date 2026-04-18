/**
 * GarmentPicker.jsx — compact thumbnail grid for picking the AR garment.
 * Shown in the Fit Engine side panel so user can switch mid-session.
 */
import { motion } from 'framer-motion';
import styles from './GarmentPicker.module.css';

export default function GarmentPicker({ garments, selected, onSelect }) {
  return (
    <div className={styles.wrap}>
      <p className={styles.label}>👗 Virtual Try-On</p>
      <div className={styles.grid}>
        {garments.map((g) => (
          <motion.button
            key={g.id}
            className={`${styles.tile} ${selected?.id === g.id ? styles.active : ''}`}
            onClick={() => onSelect(g)}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            title={g.name}
          >
            <img
              src={g.thumb}
              alt={g.name}
              className={styles.thumb}
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
            <span className={styles.name}>{g.name}</span>
            {selected?.id === g.id && (
              <span className={styles.activeBadge}>✓ On</span>
            )}
          </motion.button>
        ))}
      </div>
    </div>
  );
}
