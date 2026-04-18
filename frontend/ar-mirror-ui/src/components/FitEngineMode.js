import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import './FitEngineMode.css';
import {
  buildFrontMetrics,
  buildSideMetrics,
  fuseTruth,
} from '../utils/fitengineTruthMath';

const FitEngineMode = ({ apiBase, state, onExit, showExit = true, onProceedToOverlay }) => {
  const [heightCm, setHeightCm] = useState('170');
  const [sessionStatus, setSessionStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [isCaptured, setIsCaptured] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [capturedFrames, setCapturedFrames] = useState({ front: '', side: '' });
  const [capturedFrame, setCapturedFrame] = useState(null);
  const [worldLandmarks, setWorldLandmarks] = useState({ front: null, side: null });
  const [truthMetrics, setTruthMetrics] = useState(null);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [detectorReady, setDetectorReady] = useState(false);

  const session = sessionStatus?.session || {};
  const readiness = sessionStatus?.readiness || {};

  const pollStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/fitengine/session/status`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      setSessionStatus(data);
    } catch (_err) {
      // Non-blocking polling; leave last known status.
    }
  }, [apiBase]);

  useEffect(() => {
    pollStatus();
    const interval = setInterval(pollStatus, 500);
    return () => clearInterval(interval);
  }, [pollStatus]);

  useEffect(() => {
    let cancelled = false;

    const initDetector = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
          },
          runningMode: 'IMAGE',
          numPoses: 1,
        });
        if (!cancelled) {
          setPoseLandmarker(landmarker);
          setDetectorReady(true);
        }
      } catch (_err) {
        if (!cancelled) {
          setDetectorReady(false);
        }
      }
    };

    initDetector();
    return () => {
      cancelled = true;
      if (poseLandmarker) {
        poseLandmarker.close();
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const postAction = useCallback(async (path, payload = null) => {
    setBusy(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload ? JSON.stringify(payload) : undefined,
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || 'Action failed');
      }
      await pollStatus();
    } catch (err) {
      setError(err.message || 'Action failed');
    } finally {
      setBusy(false);
    }
  }, [apiBase, pollStatus]);

  const setGarmentOverlayVisible = useCallback(async (visible) => {
    try {
      await fetch(`${apiBase}/api/params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          show_garment_box: Boolean(visible),
          render_tryon_overlay: Boolean(visible),
        }),
      });
    } catch (_err) {
      // Non-blocking visual preference update.
    }
  }, [apiBase]);

  useEffect(() => {
    setGarmentOverlayVisible(false);
  }, [setGarmentOverlayVisible]);

  const fetchSnapshotFrame = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/snapshot`);
      if (!response.ok) {
        return '';
      }
      const data = await response.json();
      const b64 = data?.frame_jpeg_b64;
      if (!b64) {
        return '';
      }
      return `data:image/jpeg;base64,${b64}`;
    } catch (_err) {
      return '';
    }
  }, [apiBase]);

  const processCapturedFrame = useCallback(async (view, frameUrl) => {
    if (!poseLandmarker || !frameUrl) {
      return null;
    }
    try {
      const image = new Image();
      image.src = frameUrl;
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
      });

      const result = poseLandmarker.detect(image);
      const wl = result?.worldLandmarks?.[0] || null;
      if (!wl) {
        return null;
      }

      setWorldLandmarks((prev) => ({ ...prev, [view]: wl }));
      setCapturedFrame({ view, frameUrl, ts: Date.now() });
      return wl;
    } catch (_err) {
      return null;
    }
  }, [poseLandmarker]);

  const submitTruthMetrics = useCallback(async (metrics) => {
    if (!metrics) return;
    try {
      await fetch(`${apiBase}/api/fitengine/session/truth`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics }),
      });
    } catch (_err) {
      // Non-blocking submission.
    }
  }, [apiBase]);

  useEffect(() => {
    if (!worldLandmarks.front || !worldLandmarks.side) {
      return;
    }
    const height = Number(heightCm || 170);
    const frontMetrics = buildFrontMetrics(worldLandmarks.front, height);
    const sideMetrics = buildSideMetrics(worldLandmarks.side, height);
    const fused = fuseTruth(frontMetrics, sideMetrics);
    if (fused) {
      setTruthMetrics(fused);
      submitTruthMetrics(fused);
    }
  }, [heightCm, submitTruthMetrics, worldLandmarks.front, worldLandmarks.side]);

  const handleStartSession = useCallback(async () => {
    setCapturedFrames({ front: '', side: '' });
    setIsCaptured(false);
    setIsVerifying(false);
    setCapturedFrame(null);
    setWorldLandmarks({ front: null, side: null });
    setTruthMetrics(null);
    await setGarmentOverlayVisible(false);
    await postAction('/api/fitengine/session/start', { height_cm: Number(heightCm || 170) });
  }, [heightCm, postAction, setGarmentOverlayVisible]);

  const handleCaptureFront = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      const freezeFrame = await fetchSnapshotFrame();
      const response = await fetch(`${apiBase}/api/fitengine/session/capture-front`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || 'Capture front failed');
      }

      setIsCaptured(true);
      setIsVerifying(true);
      if (freezeFrame) {
        setCapturedFrames((prev) => ({ ...prev, front: freezeFrame }));
        await processCapturedFrame('front', freezeFrame);
      }
      await setGarmentOverlayVisible(false);
      await pollStatus();
      setIsVerifying(false);
    } catch (err) {
      setError(err.message || 'Capture front failed');
      setIsVerifying(false);
    } finally {
      setBusy(false);
    }
  }, [apiBase, fetchSnapshotFrame, pollStatus, processCapturedFrame, setGarmentOverlayVisible]);

  const handleCaptureSide = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      const freezeFrame = await fetchSnapshotFrame();
      const response = await fetch(`${apiBase}/api/fitengine/session/capture-side`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || 'Capture side failed');
      }

      setIsCaptured(true);
      setIsVerifying(true);
      if (freezeFrame) {
        setCapturedFrames((prev) => ({ ...prev, side: freezeFrame }));
        await processCapturedFrame('side', freezeFrame);
      }
      await setGarmentOverlayVisible(false);
      await pollStatus();
      setIsVerifying(false);
    } catch (err) {
      setError(err.message || 'Capture side failed');
      setIsVerifying(false);
    } finally {
      setBusy(false);
    }
  }, [apiBase, fetchSnapshotFrame, pollStatus, processCapturedFrame, setGarmentOverlayVisible]);

  const handleReset = useCallback(async () => {
    setCapturedFrames({ front: '', side: '' });
    setIsCaptured(false);
    setIsVerifying(false);
    setCapturedFrame(null);
    setWorldLandmarks({ front: null, side: null });
    setTruthMetrics(null);
    await setGarmentOverlayVisible(false);
    await postAction('/api/fitengine/session/reset');
  }, [postAction, setGarmentOverlayVisible]);

  const statusLabel = useMemo(() => {
    if (isVerifying) {
      return 'Frame Captured - Verifying Accuracy';
    }
    if (isCaptured) {
      return 'Measurements Locked';
    }
    if (!session.active) {
      return 'Align yourself in frame';
    }
    if (session.step === 'front') {
      return 'Step 1: Front View';
    }
    if (session.step === 'side') {
      return 'Step 2: Side View';
    }
    if (session.step === 'review') {
      return 'Review';
    }
    if (session.step === 'complete') {
      return 'Complete';
    }
    return session.step || 'In Progress';
  }, [isCaptured, isVerifying, session.active, session.step]);

  const streamUrl = `${apiBase}/stream?t=${Date.now()}`;
  const frontFrameSrc = capturedFrames.front || streamUrl;
  const sideFrameSrc = capturedFrames.side || streamUrl;
  const displayedResult = useMemo(() => {
    const baseResult = session?.result || {};
    if (!truthMetrics) return baseResult;
    return {
      ...baseResult,
      chest_cm: truthMetrics.chest_circumference_cm,
      waist_cm: truthMetrics.waist_circumference_cm,
      shoulder_cm: truthMetrics.shoulder_cm,
      torso_cm: truthMetrics.torso_cm,
      confidence: truthMetrics.truth_confidence,
      confidence_level: truthMetrics.truth_confidence >= 80 ? 'High' : truthMetrics.truth_confidence >= 60 ? 'Medium' : 'Low',
      summary: 'Measurements locked from deterministic single-frame world-landmark inference.',
    };
  }, [session?.result, truthMetrics]);
  const truthDiagnostics = displayedResult.truth_diagnostics || truthMetrics?.diagnostics || null;
  const gateBadges = useMemo(() => {
    const badges = [];
    badges.push({
      label: detectorReady ? 'Detector Ready' : 'Detector Loading',
      state: detectorReady ? 'ok' : 'warn',
    });
    badges.push({
      label: displayedResult.truth_source || truthMetrics ? 'Truth Layer Active' : 'Truth Layer Pending',
      state: displayedResult.truth_source || truthMetrics ? 'ok' : 'warn',
    });
    if (truthDiagnostics) {
      badges.push({
        label: truthDiagnostics.has_chest_ellipse ? 'Chest Ellipse OK' : 'Chest Ellipse Missing',
        state: truthDiagnostics.has_chest_ellipse ? 'ok' : 'warn',
      });
      badges.push({
        label: truthDiagnostics.has_waist_ellipse ? 'Waist Ellipse OK' : 'Waist Ellipse Missing',
        state: truthDiagnostics.has_waist_ellipse ? 'ok' : 'warn',
      });
    }
    return badges;
  }, [detectorReady, displayedResult.truth_source, truthDiagnostics, truthMetrics]);
  const formatCm = (value) => (typeof value === 'number' ? `${value.toFixed(1)} cm` : '--');
  const formatInches = (value) => (typeof value === 'number' ? `${value}"` : '--');

  return (
    <div className="fitengine-shell">
      <div className="fitengine-topbar">
        <div>
          <h2>FitEngine - Find Your Size</h2>
          <p>Takes about 20 seconds. Two photos, one result.</p>
        </div>
        {showExit && onExit && (
          <button type="button" onClick={onExit} className="fitengine-ghost-btn">Back to AR Mode</button>
        )}
      </div>

      <div className="fitengine-height-row">
        <label htmlFor="fitengine-height">Your height (cm or ft’in)</label>
        <input
          id="fitengine-height"
          type="number"
          min="130"
          max="240"
          value={heightCm}
          onChange={(e) => setHeightCm(e.target.value)}
        />
        <div className="fitengine-reference">Status: {statusLabel} • Live: {(state?.fps ?? 0).toFixed(1)} FPS</div>
      </div>

      {!detectorReady && (
        <div className="fitengine-warning-strip">
          World-landmark detector not initialized yet. Capture is available, but 3D truth metrics will populate once detector loads.
        </div>
      )}

      <div className="fitengine-workflow">
        <div className="fitengine-workflow-main">
          <div className="fitengine-grid fitengine-grid-main">
            <section className="fitengine-panel fitengine-panel-capture">
              <h4>Step 1 - Front View</h4>
              <p>Stand straight, arms slightly away from body.</p>
              <img src={frontFrameSrc} alt="Front view stream" className="fitengine-frame" />
              <div className={`fitengine-check-note fitengine-check-note-inline ${readiness.can_capture_front ? 'fitengine-check-inline-ok' : 'fitengine-check-inline-warn'}`}>
                {readiness.can_capture_front ? 'Front photo looks good' : 'Adjust posture and hold steady'}
              </div>
            </section>

            <section className="fitengine-panel fitengine-panel-capture">
              <h4>Step 2 - Side View (right side)</h4>
              <p>Turn 90° to your right and keep full upper body visible.</p>
              <img src={sideFrameSrc} alt="Side view stream" className="fitengine-frame" />
              <div className={`fitengine-check-note fitengine-check-note-inline ${readiness.can_capture_side ? 'fitengine-check-inline-ok' : 'fitengine-check-inline-warn'}`}>
                {readiness.can_capture_side ? 'Side photo looks good' : 'Turn fully to side and hold still'}
              </div>
            </section>
          </div>

          {isCaptured && (
            <div className="fitengine-warning-strip">Frame captured. Live preview paused for verification; use Reset to resume live stream.</div>
          )}

          {(sessionStatus?.warnings || []).length > 0 && (
            <div className="fitengine-warning-strip">
              {(sessionStatus.warnings || []).map((w) => (
                <div key={w}>⚠ {w}</div>
              ))}
            </div>
          )}

          {error && <div className="fitengine-error-strip">{error}</div>}

          <div className="fitengine-actions">
            <button type="button" disabled={busy} onClick={handleStartSession}>Start Session</button>
            <button type="button" disabled={busy || !readiness.can_capture_front} onClick={handleCaptureFront}>Capture Front</button>
            <button type="button" disabled={busy || !readiness.can_capture_side} onClick={handleCaptureSide}>Capture Side</button>
            <button type="button" disabled={busy || !readiness.can_finalize} onClick={() => postAction('/api/fitengine/session/finalize')}>Get My Size</button>
            <button type="button" disabled={busy} onClick={handleReset}>Reset</button>
            {onProceedToOverlay && (
              <button
                type="button"
                className="fitengine-proceed-btn"
                disabled={busy || !displayedResult.recommended_size}
                onClick={() => onProceedToOverlay(displayedResult)}
              >
                Continue to Garment Overlay
              </button>
            )}
          </div>
        </div>

        <aside className="fitengine-summary-column">
          <div className="fitengine-result-wrap fitengine-result-wrap-sticky">
            <div className="fitengine-result-card">
              <div className="fitengine-result-head">
                <div>
                  <h4>Your Size Recommendation</h4>
                  <div className="fitengine-result-subtitle">Fast summary from the captured front and side views</div>
                </div>
                <div className="fitengine-confidence-pill">{displayedResult.confidence != null ? `${Math.round(displayedResult.confidence)}%` : '--'}</div>
              </div>

              <div className="fitengine-confidence-line">
                Confidence Level: <strong>{displayedResult.confidence_level ?? '--'}</strong>
              </div>

              {capturedFrame && (
                <div className="fitengine-result-subtitle">Last capture: {capturedFrame.view} • Deterministic frame locked</div>
              )}

              <div className="fitengine-size-hero">Size {displayedResult.recommended_size ?? '--'}</div>

              <div className="fitengine-mini-grid">
                <div className="fitengine-metric-row"><span>Collar</span><strong>{formatInches(displayedResult.collar_in)}</strong></div>
                <div className="fitengine-metric-row"><span>Trouser Waist</span><strong>{formatInches(displayedResult.trouser_waist_in)}</strong></div>
                <div className="fitengine-metric-row"><span>Chest</span><strong>{formatCm(displayedResult.chest_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Waist</span><strong>{formatCm(displayedResult.waist_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Shoulder</span><strong>{formatCm(displayedResult.shoulder_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Torso</span><strong>{formatCm(displayedResult.torso_cm)}</strong></div>
              </div>

              <div className="fitengine-summary-note">
                {displayedResult.summary || 'Capture both views, then generate the final size guidance.'}
              </div>

              <div className="fitengine-gate-badges">
                {gateBadges.map((badge) => (
                  <span key={badge.label} className={`fitengine-gate-badge fitengine-gate-${badge.state}`}>{badge.label}</span>
                ))}
              </div>

              {truthDiagnostics && (
                <div className="fitengine-diagnostics-grid">
                  <div className="fitengine-breakdown-item"><span>Scale Drift</span><strong>{truthDiagnostics.scale_drift_pct ?? '--'}%</strong></div>
                  <div className="fitengine-breakdown-item"><span>Chest W/D</span><strong>{truthDiagnostics.chest_width_cm ? `${truthDiagnostics.chest_width_cm.toFixed(1)}/${truthDiagnostics.chest_depth_cm?.toFixed(1) ?? '--'} cm` : '--'}</strong></div>
                  <div className="fitengine-breakdown-item"><span>Waist W/D</span><strong>{truthDiagnostics.waist_width_cm ? `${truthDiagnostics.waist_width_cm.toFixed(1)}/${truthDiagnostics.waist_depth_cm?.toFixed(1) ?? '--'} cm` : '--'}</strong></div>
                </div>
              )}

              {displayedResult.confidence_breakdown && (
                <div className="fitengine-breakdown-grid">
                  <div className="fitengine-breakdown-item"><span>Base</span><strong>{displayedResult.confidence_breakdown.base ?? '--'}%</strong></div>
                  <div className="fitengine-breakdown-item"><span>Stability</span><strong>{displayedResult.confidence_breakdown.stability ?? '--'}%</strong></div>
                  <div className="fitengine-breakdown-item"><span>Coverage</span><strong>{displayedResult.confidence_breakdown.coverage ?? '--'}%</strong></div>
                </div>
              )}

              {Array.isArray(displayedResult.reasons) && displayedResult.reasons.length > 0 && (
                <div className="fitengine-reasons">
                  {displayedResult.reasons.map((reason, idx) => (
                    <div key={`${idx}-${reason}`}>• {reason}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default FitEngineMode;
