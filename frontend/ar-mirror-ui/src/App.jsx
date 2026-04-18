import { AnimatePresence, motion } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';
import FitEngineView from './components/fit-engine/FitEngineView';
import SizeSummaryOverlay from './components/fit-engine/SizeSummaryOverlay';
import GarmentView from './components/garment-overlay/GarmentView';
import Navbar from './components/layout/Navbar';
import ShareModal from './components/share/ShareModal';
import { useBackendBridge } from './hooks/useBackendBridge';
import { useFrameTime } from './hooks/useFrameTime';
import { useWebcam } from './hooks/useWebcam';
import { useWebSocket } from './hooks/useWebSocket';
import { generateShare } from './lib/backendApi';
import { normalizeFitEngineMeasurements, normalizePosePoints } from './lib/backendTransforms';
import { MOCK_GARMENTS } from './data/mockData';
import { AppStoreProvider, useAppStore } from './store/appStore';
import styles from './App.module.css';

function deriveAccuracyMetricsFromState(state) {
  if (!state) {
    return null;
  }

  const meas = state.measurements || {};
  const toScore = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      return null;
    }
    return n <= 1 ? Math.max(0, Math.min(100, n * 100)) : Math.max(0, Math.min(100, n));
  };

  const poseConf = toScore(state.pose_confidence) ?? 0;
  const stability = toScore(meas.size_confidence);
  const finalConfidence = toScore(state?.fit_engine?.session?.result?.confidence) ?? stability;
  const fps = Number(state.fps);
  const fpsValue = Number.isFinite(fps) && fps > 0 ? fps : null;
  const latency = fpsValue ? (1000 / fpsValue) : null;

  const shoulderConf = toScore(meas?.measurement_confidence?.shoulder);
  const chestConf = toScore(meas?.measurement_confidence?.chest);
  const torsoConf = toScore(meas?.measurement_confidence?.torso);
  const confVals = [shoulderConf, chestConf, torsoConf].filter((v) => typeof v === 'number');
  const inferredQuality = confVals.length > 0
    ? confVals.reduce((acc, v) => acc + v, 0) / confVals.length
    : poseConf;

  return {
    pose_quality_score: Number(((poseConf * 0.6) + (inferredQuality * 0.4)).toFixed(1)),
    measurement: {
      stability_score: stability,
    },
    confidence: {
      final: finalConfidence,
    },
    rolling_performance: {
      fps: { p50: fpsValue },
      latency_ms: { p90: latency ? Number(latency.toFixed(1)) : null },
    },
  };
}

function AppContent() {
  const socket = useWebSocket();
  const { fps } = useFrameTime();
  const { setMode } = useAppStore();
  const { state, garments: backendGarments, accuracyMetrics, connected, error, actions } = useBackendBridge();

  // Browser webcam — always on, used as the live mirror feed
  const { videoRef: webcamRef, ready: webcamReady, captureSnapshot } = useWebcam({ enabled: true });

  const livePosePoints = normalizePosePoints(state?.pose2D || []);
  const fitEngineData = normalizeFitEngineMeasurements(state || {});
  const effectiveAccuracyMetrics = accuracyMetrics || deriveAccuracyMetricsFromState(state);
  const garments = backendGarments.length > 0 ? backendGarments : MOCK_GARMENTS.map((item) => ({
    ...item,
    id: String(item.id),
    backendKey: String(item.id),
  }));
  const [selectedGarmentId, setSelectedGarmentId] = useState(null);

  // Derived: full garment object from ID (defaults to first garment)
  const selectedGarment = garments.find((g) => g.id === selectedGarmentId) || garments[0] || null;
  const setSelectedGarment = useCallback((garment) => setSelectedGarmentId(garment?.id ?? null), []);

  // ── Share feature state ────────────────────────────────────────────────────
  const [shareData, setShareData] = useState(null);
  const [sharing, setSharing] = useState(false);
  const [shareError, setShareError] = useState(null);
  const sharingRef = useRef(false);

  const handleShare = useCallback(async () => {
    if (sharingRef.current) return;
    sharingRef.current = true;
    setSharing(true);
    setShareError(null);

    // Capture current webcam frame as JPEG base64
    const previewB64 = captureSnapshot ? captureSnapshot(0.88) : '';

    // Use real hostname (works over local WiFi, not just localhost)
    const origin = window.location.origin;   // e.g. http://192.168.1.x:3002

    try {
      const data = await generateShare(previewB64 || '');
      if (data?.ok) {
        if (!data.preview_b64 && previewB64) data.preview_b64 = previewB64;
        setShareData(data);
      } else {
        throw new Error('offline');
      }
    } catch {
      // Offline fallback: Use the origin URL. 
      // NOTE: If testing from mobile, you must use your machine's network IP (e.g. 192.168.1.5) instead of localhost
      const fallbackUrl = `${origin}/share/demo`;

      setShareData({
        ok:          true,
        share_id:    'demo',
        share_url:   fallbackUrl,
        nfc_payload: fallbackUrl,
        qr_b64:      '',
        qr_content:  fallbackUrl, // Fix: Use URL for QR code to avoid blowing up the QR string size!
        preview_b64: previewB64 || '',
        expires_at:  Date.now() / 1000 + 900,
        _offline:    true,
      });
      setShareError(previewB64
        ? 'Demo mode — QR encodes your live snapshot. Scan to view the photo!'
        : 'Backend offline. Start app.py for full features.');
    } finally {
      sharingRef.current = false;
      setSharing(false);
    }
  }, [captureSnapshot]);

  const backendReady = connected && !error;
  const [autoCaptureEnabled] = useState(true);
  const [autoCueMessage, setAutoCueMessage] = useState('');
  const [flowState, setFlowState] = useState('idle');
  const [frozenSummaryData, setFrozenSummaryData] = useState(null);
  const autoPilotRef = useRef({ inFlight: false, lastActionAt: 0 });
  const autoCueTimerRef = useRef(null);
  const stabilityGateRef = useRef({ step: null, readySince: 0 });
  const hasStartedSessionRef = useRef(false);
  const summaryHandledRef = useRef(null);
  const summaryTimerRef = useRef(null);
  const reviewFallbackTimerRef = useRef(null);

  const clearAutoCueTimer = useCallback(() => {
    if (autoCueTimerRef.current) {
      window.clearInterval(autoCueTimerRef.current);
      autoCueTimerRef.current = null;
    }
  }, []);

  const speakCue = useCallback((message) => {
    if (typeof window === 'undefined' || !window.speechSynthesis || !message) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.02;
    utterance.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }, []);

  useEffect(() => {
    if (!backendReady) {
      hasStartedSessionRef.current = false;
      summaryHandledRef.current = null;
      // If offline, default to idle so FitEngineView displays the camera, garment overlay, and measurements together.
      setFlowState('idle');
      setFrozenSummaryData(null);
      return;
    }
    if (state?.fit_engine?.session?.active || hasStartedSessionRef.current) {
      return;
    }
    hasStartedSessionRef.current = true;
    setFlowState('auto-capture');
    setMode('fit-engine');
    void actions.startSession(170).catch(() => {
      hasStartedSessionRef.current = false;
    });
  }, [backendReady, state, actions]);

  useEffect(() => {
    return () => {
      clearAutoCueTimer();
      if (summaryTimerRef.current) {
        window.clearTimeout(summaryTimerRef.current);
      }
      if (reviewFallbackTimerRef.current) {
        window.clearTimeout(reviewFallbackTimerRef.current);
      }
    };
  }, [clearAutoCueTimer]);

  useEffect(() => {
    if (!autoCaptureEnabled) {
      clearAutoCueTimer();
      setAutoCueMessage('');
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    }
  }, [autoCaptureEnabled, clearAutoCueTimer]);

  useEffect(() => {
    if (!garments.length) {
      return;
    }
    if (!selectedGarmentId || !garments.some((item) => item.id === selectedGarmentId)) {
      setSelectedGarmentId(garments[0].id);
    }
  }, [garments, selectedGarmentId]);


  useEffect(() => {
    const result = state?.fit_engine?.session?.result;
    const sessionId = state?.fit_engine?.session?.session_id;
    const summaryKey = result?.generated_at ? `${sessionId}:${result.generated_at}` : null;

    if (!result || flowState !== 'auto-capture') {
      if (!result) {
        summaryHandledRef.current = null;
      }
      return;
    }

    if (summaryHandledRef.current === summaryKey) {
      return;
    }

    summaryHandledRef.current = summaryKey;
    setFrozenSummaryData(fitEngineData);
    setFlowState('measure-display');
    setMode('fit-engine');
  }, [flowState, state, setMode, fitEngineData]);

  useEffect(() => {
    if (flowState !== 'measure-display') {
      return;
    }

    if (summaryTimerRef.current) {
      window.clearTimeout(summaryTimerRef.current);
    }

    summaryTimerRef.current = window.setTimeout(() => {
      setFlowState('tryon');
      setMode('garment-overlay');
    }, 3000);

    return () => {
      if (summaryTimerRef.current) {
        window.clearTimeout(summaryTimerRef.current);
      }
    };
  }, [flowState, setMode]);

  useEffect(() => {
    if (!autoCaptureEnabled || flowState !== 'auto-capture' || !backendReady) {
      return;
    }

    const fitEngine = state?.fit_engine;
    const session = fitEngine?.session;
    const readiness = fitEngine?.readiness;
    if (!session?.active || !readiness) {
      return;
    }
    const calibratedEnough = Boolean(fitEngineData?.calibratedEnough);

    if (!calibratedEnough) {
      setAutoCueMessage('Calibrating body scale. Keep full torso visible and hold still.');
      return;
    }

    const now = Date.now();
    if (autoPilotRef.current.inFlight || now - autoPilotRef.current.lastActionAt < 900) {
      return;
    }

    let actionPlan = null;
    if (session.step === 'front' && readiness.measurement_ready) {
      actionPlan = {
        action: actions.captureFront,
        countLabel: 'Front capture in',
        progressLabel: 'Capturing front view',
        spokenHint: 'Hold still. Capturing front now.',
        countdown: true,
      };
    } else if (
      session.step === 'side' &&
      readiness.measurement_ready
    ) {
      actionPlan = {
        action: actions.captureSide,
        countLabel: 'Side capture in',
        progressLabel: 'Capturing side view',
        spokenHint: 'Turn sideways. Capturing side now.',
        countdown: true,
      };
    } else if (session.step === 'review') {
      actionPlan = {
        action: actions.finalizeSession,
        progressLabel: 'Analyzing measurements',
        spokenHint: 'Analyzing your measurements now.',
        countdown: false,
      };
    }

    if (!actionPlan) {
      return;
    }

    const executeAction = () => {
      setAutoCueMessage(`${actionPlan.progressLabel}...`);
      autoPilotRef.current.inFlight = true;
      autoPilotRef.current.lastActionAt = Date.now();
      speakCue(actionPlan.spokenHint);
      Promise.resolve(actionPlan.action())
        .catch(() => {
          // Keep auto mode running; transient frame/pose jitter can fail single requests.
        })
        .finally(() => {
          autoPilotRef.current.inFlight = false;
          window.setTimeout(() => setAutoCueMessage(''), 900);
        });
    };

    if (actionPlan.countdown) {
      clearAutoCueTimer();
      autoPilotRef.current.inFlight = true;
      autoPilotRef.current.lastActionAt = now;
      let ticks = 2;
      setAutoCueMessage(`${actionPlan.countLabel} ${ticks}`);
      speakCue(actionPlan.spokenHint);
      autoCueTimerRef.current = window.setInterval(() => {
        ticks -= 1;
        if (ticks > 0) {
          setAutoCueMessage(`${actionPlan.countLabel} ${ticks}`);
          return;
        }

        clearAutoCueTimer();
        Promise.resolve(actionPlan.action())
          .catch(() => {
            // Keep auto mode running; transient frame/pose jitter can fail single requests.
          })
          .finally(() => {
            autoPilotRef.current.inFlight = false;
            autoPilotRef.current.lastActionAt = Date.now();
            setAutoCueMessage(`${actionPlan.progressLabel}...`);
            window.setTimeout(() => setAutoCueMessage(''), 900);
          });
      }, 1000);
      return;
    }

    autoPilotRef.current.lastActionAt = now;
    if (!autoPilotRef.current.inFlight) {
      executeAction();
    }
  }, [
    autoCaptureEnabled,
    flowState,
    backendReady,
    state,
    actions.captureFront,
    actions.captureSide,
    actions.finalizeSession,
    clearAutoCueTimer,
    speakCue,
    fitEngineData,
  ]);

  useEffect(() => {
    if (flowState !== 'auto-capture') {
      if (reviewFallbackTimerRef.current) {
        window.clearTimeout(reviewFallbackTimerRef.current);
        reviewFallbackTimerRef.current = null;
      }
      return;
    }

    const step = state?.fit_engine?.session?.step;
    const hasResult = Boolean(state?.fit_engine?.session?.result);
    if (step !== 'review' || hasResult) {
      if (reviewFallbackTimerRef.current) {
        window.clearTimeout(reviewFallbackTimerRef.current);
        reviewFallbackTimerRef.current = null;
      }
      return;
    }

    if (!reviewFallbackTimerRef.current) {
      reviewFallbackTimerRef.current = window.setTimeout(() => {
        setFlowState('tryon');
        setMode('garment-overlay');
      }, 1600);
    }
  }, [flowState, state, setMode]);

  return (
    <div className={styles.app}>
      <Navbar
        connected={backendReady && socket.connected}
        fps={fps}
        onShare={handleShare}
        sharing={sharing}
      />

      <AnimatePresence mode="wait">
        <motion.div
          key={flowState}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 0.35 }}
        >
          {flowState === 'tryon' ? (
            <GarmentView
              garments={garments}
              selectedGarment={selectedGarment}
              onSelect={async (id) => {
                const nextGarment = garments.find((item) => item.id === id);
                if (!nextGarment) return;
                setSelectedGarmentId(nextGarment.id);
                await actions.selectGarment(nextGarment.backendKey || nextGarment.id);
              }}
              fps={fps}
              streamUrl={connected ? 'http://localhost:5051/stream' : null}
              webcamRef={webcamRef}
              webcamReady={webcamReady}
              livePosePoints={livePosePoints}
              backendState={state}
            />
          ) : (
            <FitEngineView
              livePosePoints={livePosePoints}
              fitEngineData={fitEngineData}
              backendState={state}
              autoCaptureEnabled={autoCaptureEnabled}
              autoCueMessage={autoCueMessage}
              webcamRef={webcamRef}
              webcamReady={webcamReady}
              selectedGarment={selectedGarment}
              garments={garments}
              onSelectGarment={setSelectedGarment}
            />
          )}
        </motion.div>
      </AnimatePresence>

      <AnimatePresence>
        {flowState === 'measure-display' ? (
          <SizeSummaryOverlay
            key="size-summary"
            data={frozenSummaryData || fitEngineData}
          />
        ) : null}
      </AnimatePresence>

      {/* ── Share Modal ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {shareData ? (
          <ShareModal
            key="share-modal"
            shareData={shareData}
            offlineError={shareError}
            onClose={() => { setShareData(null); setShareError(null); }}
          />
        ) : null}
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  return (
    <AppStoreProvider>
      <AppContent />
    </AppStoreProvider>
  );
}
