import { motion } from 'framer-motion';
import { useAppStore } from '../../store/appStore';
import { usePoseMock } from '../../hooks/usePoseMock';
import CameraCard from './CameraCard';
import MeasurementPanel from './MeasurementPanel';
import styles from './FitEngineView.module.css';

export default function FitEngineView() {
  const { measurements, setMode, setSessionActive, setCaptureState } = useAppStore();
  const frontPose = usePoseMock({ phaseOffset: 0 });
  const sidePose = usePoseMock({ phaseOffset: 0.8 });

  return (
    <motion.section
      className={styles.view}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <div className={styles.meshBackground} />

      <div className={styles.grid}>
        <CameraCard
          label="Front Capture"
          posePoints={frontPose}
          isActive
          confidence={89}
          instruction="Keep shoulders centered and chin level"
        />

        <CameraCard
          label="Side Capture"
          posePoints={sidePose}
          isActive={false}
          confidence={84}
          instruction="Rotate 90 degrees and keep posture neutral"
        />

        <MeasurementPanel data={measurements} />
      </div>

      <div className={styles.actionBarWrap}>
        <div className={styles.actionBar}>
          <button onClick={() => setSessionActive(true)}>Start Session</button>
          <button onClick={() => setCaptureState((s) => ({ ...s, front: true }))}>Capture Front</button>
          <button onClick={() => setCaptureState((s) => ({ ...s, side: true }))}>Capture Side</button>
          <motion.button
            className={styles.cta}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setMode('garment-overlay')}
          >
            Get My Size ✦
          </motion.button>
          <button onClick={() => setCaptureState({ front: false, side: false })}>Reset</button>
        </div>
      </div>
    </motion.section>
  );
}
