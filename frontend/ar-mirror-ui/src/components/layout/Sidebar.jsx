import { motion } from 'framer-motion';
import { useState } from 'react';
import styles from './Sidebar.module.css';

const items = [
  { id: 'scan', label: 'Scan' },
  { id: 'fit', label: 'Fit' },
  { id: 'overlay', label: 'Overlay' },
  { id: 'share', label: 'Share' },
];

export default function Sidebar({ activeId, onAction, disabledIds = [], busyActionId = null }) {
  const [open, setOpen] = useState(true);

  return (
    <motion.aside
      className={styles.sidebar}
      animate={{ width: open ? 78 : 56 }}
      transition={{ type: 'spring', stiffness: 220, damping: 22 }}
    >
      <button className={styles.toggle} onClick={() => setOpen((v) => !v)}>
        {open ? '◀' : '▶'}
      </button>

      <div className={styles.rail}>
        {items.map((item) => (
          <button
            key={item.id}
            className={`${styles.item} ${activeId === item.id ? styles.active : ''}`}
            title={item.label}
            onClick={() => onAction?.(item.id)}
            disabled={disabledIds.includes(item.id) || busyActionId === item.id}
          >
            <span className={styles.dot} />
            {open && <span>{item.label}</span>}
          </button>
        ))}
      </div>
    </motion.aside>
  );
}
