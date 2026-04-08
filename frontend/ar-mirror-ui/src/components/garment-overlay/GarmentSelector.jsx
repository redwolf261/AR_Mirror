import { AnimatePresence, motion } from 'framer-motion';
import styles from './GarmentSelector.module.css';

export default function GarmentSelector({ garments, selectedId, onSelect }) {
  return (
    <AnimatePresence>
      <motion.div
        className={styles.panel}
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 16 }}
      >
        <h4>Garments</h4>
        <div className={styles.grid}>
          {garments.map((item) => (
            <button
              key={item.id}
              className={item.id === selectedId ? styles.active : ''}
              onClick={() => onSelect(item.id)}
            >
              <img src={item.thumb} alt={item.name} onError={(e) => { e.currentTarget.style.opacity = '0.35'; }} />
              <span>{item.name}</span>
            </button>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
