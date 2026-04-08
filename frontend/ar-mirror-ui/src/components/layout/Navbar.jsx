import { motion } from 'framer-motion';
import { useAppStore } from '../../store/appStore';
import styles from './Navbar.module.css';

export default function Navbar({ connected, fps }) {
  const { mode, setMode } = useAppStore();

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

      <nav className={styles.switcher}>
        <button
          className={mode === 'fit-engine' ? styles.active : ''}
          onClick={() => setMode('fit-engine')}
        >
          FitEngine
        </button>
        <button
          className={mode === 'garment-overlay' ? styles.active : ''}
          onClick={() => setMode('garment-overlay')}
        >
          Garment Overlay
        </button>
      </nav>

      <div className={styles.telemetry}>
        <span className={connected ? styles.connected : styles.disconnected}>
          {connected ? 'WS LIVE' : 'WS OFF'}
        </span>
        <span className={styles.fps}>{fps} FPS</span>
      </div>
    </motion.header>
  );
}
