import { motion } from 'framer-motion';
import { useMemo } from 'react';
import SizeReveal from './SizeReveal';
import styles from './SizeSummaryOverlay.module.css';

export default function SizeSummaryOverlay({ data }) {
  const stats = useMemo(() => {
    const metrics = data?.metrics || {};
    const scores = data?.scores || {};
    const reasons = Array.isArray(data?.tips) ? data.tips : [];

    return {
      size: data?.size || '--',
      confidence: Number.isFinite(data?.confidence) ? data.confidence : 0,
      label: data?.confidenceLabel || 'Calibrating',
      chest: metrics.chest,
      waist: metrics.waist,
      shoulder: metrics.shoulder,
      torso: metrics.torso,
      scores,
      reasons,
    };
  }, [data]);

  return (
    <motion.div
      className={styles.overlay}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className={styles.glow} />
      <div className={styles.shell}>
        <div className={styles.header}>
          <p>Auto-captured fit result</p>
          <span>Preparing garment overlay</span>
        </div>

        <div className={styles.hero}>
          <SizeReveal size={stats.size} confidence={stats.confidence} label={stats.label} />
          <div className={styles.subtitle}>
            {data?.calibratedEnough
              ? 'Size prediction complete. The garment overlay will open automatically.'
              : 'Measurements are calibrating. Keep full torso in frame and hold still.'}
          </div>
        </div>

        <div className={styles.statsGrid}>
          {[
            ['Chest', stats.chest],
            ['Waist', stats.waist],
            ['Shoulder', stats.shoulder],
            ['Torso', stats.torso],
          ].map(([label, metric]) => (
            <div key={label} className={styles.statCard}>
              <span>{label}</span>
              <strong>{Number.isFinite(metric?.value) ? `${metric.value.toFixed(1)} ${metric.unit}` : '--'}</strong>
            </div>
          ))}
        </div>

        <div className={styles.scoreRow}>
          {Object.entries(stats.scores).map(([key, value]) => (
            <div key={key} className={styles.scoreCard}>
              <span>{key}</span>
              <strong>{value}/100</strong>
            </div>
          ))}
        </div>

        <div className={styles.reasonList}>
          {stats.reasons.slice(0, 4).map((reason) => (
            <p key={reason}>{reason}</p>
          ))}
        </div>

      </div>
    </motion.div>
  );
}