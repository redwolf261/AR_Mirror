import styles from './ConfidenceRing.module.css';

export default function ConfidenceRing({ value = 33, size = 48, stroke = 5 }) {
  const radius = (size - stroke) / 2;
  const c = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, value));
  const dash = c - (pct / 100) * c;

  return (
    <svg className={styles.ring} width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle className={styles.track} cx={size / 2} cy={size / 2} r={radius} strokeWidth={stroke} />
      <circle
        className={styles.progress}
        cx={size / 2}
        cy={size / 2}
        r={radius}
        strokeWidth={stroke}
        strokeDasharray={c}
        strokeDashoffset={dash}
      />
    </svg>
  );
}
