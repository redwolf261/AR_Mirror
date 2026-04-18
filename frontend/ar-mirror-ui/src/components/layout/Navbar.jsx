import { motion } from 'framer-motion';
import styles from './Navbar.module.css';

export default function Navbar({ connected, fps, onShare, sharing }) {
  return (
    <motion.header
      className={styles.navbar}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <div className={styles.brand}>
        <div className={styles.brandDot} />
        <div>
          <h1>Adaptive AR Try-On Engine</h1>
          <p>Real-Time Fit Intelligence</p>
        </div>
      </div>

      <div className={styles.telemetry}>
        <span className={connected ? styles.connected : styles.disconnected}>
          {connected ? 'WS LIVE' : 'WS OFF'}
        </span>
        <span className={styles.fps}>{fps} FPS</span>
        <motion.button
          className={styles.shareBtn}
          onClick={onShare}
          disabled={sharing}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          title="Share your try-on result"
        >
          {sharing ? (
            <span className={styles.shareSpinner}>⏳</span>
          ) : (
            <>📤 Share</>
          )}
        </motion.button>
      </div>
    </motion.header>
  );
}
