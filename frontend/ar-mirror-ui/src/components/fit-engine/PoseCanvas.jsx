import { motion } from 'framer-motion';
import { POSE_BONES } from '../../data/mockData';
import styles from './PoseCanvas.module.css';

export default function PoseCanvas({ posePoints = [], width = 640, height = 480 }) {
  const map = new Map(posePoints.map((p) => [p.id, p]));

  return (
    <svg className={styles.canvas} viewBox={`0 0 ${width} ${height}`}>
      {POSE_BONES.map(([a, b]) => {
        const p1 = map.get(a);
        const p2 = map.get(b);
        if (!p1 || !p2) {
          return null;
        }
        return <line key={`${a}-${b}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} className={styles.bone} />;
      })}

      {posePoints.map((p, i) => (
        <motion.circle
          key={p.id}
          cx={p.x}
          cy={p.y}
          r={p.id.includes('shoulder') ? 5 : 3}
          className={p.id.includes('shoulder') ? styles.activeJoint : styles.joint}
          animate={{ scale: [1, 1.28, 1] }}
          transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.05 }}
          style={{ transformOrigin: `${p.x}px ${p.y}px` }}
        />
      ))}
    </svg>
  );
}
