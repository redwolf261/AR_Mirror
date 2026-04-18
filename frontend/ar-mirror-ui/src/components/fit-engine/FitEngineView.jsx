import { motion } from 'framer-motion';
import { usePoseMock } from '../../hooks/usePoseMock';
import { useFrontendPose } from '../../hooks/useFrontendPose';
import CameraCard from './CameraCard';
import GarmentPicker from './GarmentPicker';
import MeasurementPanel from './MeasurementPanel';
import { normalizeFitEngineMeasurements, normalizePosePoints } from '../../lib/backendTransforms';
import styles from './FitEngineView.module.css';

export default function FitEngineView({
  livePosePoints = [],
  fitEngineData,
  backendState,
  streamUrl = null,
  autoCaptureEnabled = false,
  autoCueMessage = '',
  webcamRef,
  webcamReady,
  webcamError,
  selectedGarment,
  garments = [],
  onSelectGarment,
}) {
  const fallbackMockPose = usePoseMock({ phaseOffset: 0 });
  const frontendPoseData = useFrontendPose(webcamRef?.current ? webcamRef : null);
  const liveFrontPose = normalizePosePoints(backendState?.pose2D || []);
  
  // Prefer Backend > Frontend Mediapipe > Mock Generator
  const fallbackFrontPose = frontendPoseData.keypoints.length > 0 ? frontendPoseData.keypoints : fallbackMockPose;
  const frontPose = livePosePoints.length > 0 ? livePosePoints : liveFrontPose.length > 0 ? liveFrontPose : fallbackFrontPose;
  
  const readiness = backendState?.fit_engine?.readiness || fitEngineData?.fitEngine?.readiness || {};
  const session = backendState?.fit_engine?.session || fitEngineData?.fitEngine?.session || {};
  const calibratedEnough = Boolean(fitEngineData?.calibratedEnough);

  // If frontend Mediapipe is actively tracking, override the panel data
  const finalMeasurementData = frontendPoseData.metrics.active ? {
    ...fitEngineData,
    size: frontendPoseData.metrics.size,
    metrics: {
      chest: { value: frontendPoseData.metrics.chest, unit: 'cm' },
      waist: { value: frontendPoseData.metrics.waist, unit: 'cm' },
      shoulder: { value: frontendPoseData.metrics.shoulder, unit: 'cm' },
    }
  } : (!frontendPoseData.metrics.active && frontendPoseData.keypoints.length === 0) ? {
    // Zero out if strictly no person found
    ...fitEngineData,
    size: '-',
    metrics: { chest: {value: 0}, waist: {value: 0}, shoulder: {value: 0} }
  } : fitEngineData;

  const instructionByStep = {
    front: calibratedEnough
      ? 'Face front, keep shoulders centered, hold still.'
      : 'Calibrating scale: keep your full torso visible and stay still.',
    side: calibratedEnough
      ? 'Turn to your side and hold still for side capture.'
      : 'Calibrating scale: keep your full torso visible and stay still.',
    review: 'Analyzing measurements and preparing try-on.',
  };
  const currentInstruction = instructionByStep[session.step] || 'Align body to continue';

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
          label="Live Capture"
          posePoints={frontPose}
          isActive
          confidence={Math.round(finalMeasurementData?.confidence || 89)}
          instruction={currentInstruction}
          streamUrl={streamUrl}
          webcamRef={webcamRef}
          webcamReady={webcamReady}
          webcamError={webcamError}
          selectedGarment={selectedGarment}
        />

        <div className={styles.sideColumn}>
          <MeasurementPanel data={finalMeasurementData || normalizeFitEngineMeasurements(backendState || {})} />

          {/* Garment picker */}
          {garments.length > 0 && onSelectGarment && (
            <GarmentPicker
              garments={garments}
              selected={selectedGarment}
              onSelect={onSelectGarment}
            />
          )}

          <div className={styles.autoStatus}>
            Auto Capture: {autoCaptureEnabled ? 'ON' : 'OFF'}
          </div>
        </div>
      </div>

      {autoCaptureEnabled && autoCueMessage ? <div className={styles.autoCue}>{autoCueMessage}</div> : null}

      <div className={styles.actionBarWrap}>
        <div className={styles.actionBar}>
          <span className={styles.readinessText}>
            {!calibratedEnough
              ? 'Calibrating measurements'
              : session.step === 'side'
                ? 'Rotate to side view for next capture'
                : readiness.measurement_ready
                  ? 'Measurements ready'
                  : 'Align body to continue'}
          </span>
        </div>
      </div>
    </motion.section>
  );
}
