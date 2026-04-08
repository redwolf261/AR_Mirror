import { motion } from 'framer-motion';
import { usePoseMock } from '../../hooks/usePoseMock';
import ARScene from '../three/ARScene';
import LiveHUD from './LiveHUD';
import styles from './GarmentView.module.css';

export default function GarmentView({ garments, selectedGarment, onSelect, fps }) {
  const pose = usePoseMock({ phaseOffset: 0.45 });

  return (
    <motion.section
      className={styles.view}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <ARScene posePoints={pose} garment={selectedGarment} />
      <LiveHUD
        garments={garments}
        selectedGarment={selectedGarment}
        onSelect={onSelect}
        fps={fps}
      />
    </motion.section>
  );
}
