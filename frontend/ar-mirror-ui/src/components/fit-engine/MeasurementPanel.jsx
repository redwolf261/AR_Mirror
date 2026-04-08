import { motion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import SizeReveal from './SizeReveal';
import styles from './MeasurementPanel.module.css';

function useAnimatedNumber(value, duration = 650) {
  const [display, setDisplay] = useState(value);

  useEffect(() => {
    const start = performance.now();
    const from = display;
    let raf;

    const step = (t) => {
      const p = Math.min(1, (t - start) / duration);
      setDisplay(from + (value - from) * p);
      if (p < 1) {
        raf = requestAnimationFrame(step);
      }
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return display;
}

export default function MeasurementPanel({ data }) {
  const metricItems = useMemo(
    () => [
      ['Chest', data.metrics.chest],
      ['Waist', data.metrics.waist],
      ['Shoulder', data.metrics.shoulder],
      ['Collar', data.metrics.collar],
      ['Trouser Waist', data.metrics.trouserWaist],
      ['Torso', data.metrics.torso],
    ],
    [data]
  );

  return (
    <motion.aside
      className={styles.panel}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <SizeReveal size={data.size} confidence={data.confidence} label={data.confidenceLabel} />

      <motion.div className={styles.grid} variants={{ hidden: {}, show: { transition: { staggerChildren: 0.07 } } }} initial="hidden" animate="show">
        {metricItems.map(([label, metric]) => {
          return (
            <MetricCell key={label} label={label} metric={metric} />
          );
        })}
      </motion.div>

      <div className={styles.scoreRow}>
        {Object.entries(data.scores).map(([k, v]) => (
          <motion.div key={k} layout className={styles.scorePill}>
            <div className={styles.scoreLabel}>{k}</div>
            <div className={styles.track}><motion.div className={styles.fill} initial={{ width: 0 }} animate={{ width: `${v}%` }} /></div>
          </motion.div>
        ))}
      </div>

      <div className={styles.tips}>
        {data.tips.map((tip, idx) => (
          <motion.p key={tip} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.18 }}>
            {`> ${tip}`}
          </motion.p>
        ))}
      </div>
    </motion.aside>
  );
}

function MetricCell({ label, metric }) {
  const v = useAnimatedNumber(metric.value);

  return (
    <motion.div className={styles.metric} variants={{ hidden: { opacity: 0, x: -12 }, show: { opacity: 1, x: 0 } }}>
      <span className={styles.icon}>◉</span>
      <div>
        <p className={styles.name}>{label}</p>
        <p className={styles.value}>{v.toFixed(1)} {metric.unit}</p>
      </div>
    </motion.div>
  );
}
