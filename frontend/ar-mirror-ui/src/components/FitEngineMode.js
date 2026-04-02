import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './FitEngineMode.css';

const FitEngineMode = ({ apiBase, state, onExit }) => {
  const [heightCm, setHeightCm] = useState('170');
  const [sessionStatus, setSessionStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

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

  const statusLabel = useMemo(() => {
    if (!session.active) {
      return 'Idle';
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
  }, [session.active, session.step]);

  const streamUrl = `${apiBase}/stream?t=${Date.now()}`;
  const result = session?.result || {};
  const formatCm = (value) => (typeof value === 'number' ? `${value.toFixed(1)} cm` : '--');
  const formatInches = (value) => (typeof value === 'number' ? `${value}"` : '--');

  return (
    <div className="fitengine-shell">
      <div className="fitengine-topbar">
        <div>
          <h2>FitEngine - Find Your Size</h2>
          <p>Takes about 20 seconds. Two photos, one result. Existing AR mode remains preserved.</p>
        </div>
        <button type="button" onClick={onExit} className="fitengine-ghost-btn">Back to AR Mode</button>
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

      <div className="fitengine-workflow">
        <div className="fitengine-workflow-main">
          <div className="fitengine-grid fitengine-grid-main">
            <section className="fitengine-panel fitengine-panel-capture">
              <h4>Step 1 - Front View</h4>
              <p>Stand straight, arms slightly away from body.</p>
              <img src={streamUrl} alt="Front view stream" className="fitengine-frame" />
              <div className={`fitengine-check-note fitengine-check-note-inline ${readiness.can_capture_front ? 'fitengine-check-inline-ok' : 'fitengine-check-inline-warn'}`}>
                {readiness.can_capture_front ? 'Front photo looks good' : 'Adjust posture and hold steady'}
              </div>
            </section>

            <section className="fitengine-panel fitengine-panel-capture">
              <h4>Step 2 - Side View (right side)</h4>
              <p>Turn 90° to your right and keep full upper body visible.</p>
              <img src={streamUrl} alt="Side view stream" className="fitengine-frame" />
              <div className={`fitengine-check-note fitengine-check-note-inline ${readiness.can_capture_side ? 'fitengine-check-inline-ok' : 'fitengine-check-inline-warn'}`}>
                {readiness.can_capture_side ? 'Side photo looks good' : 'Turn fully to side and hold still'}
              </div>
            </section>
          </div>

          {(sessionStatus?.warnings || []).length > 0 && (
            <div className="fitengine-warning-strip">
              {(sessionStatus.warnings || []).map((w) => (
                <div key={w}>⚠ {w}</div>
              ))}
            </div>
          )}

          {error && <div className="fitengine-error-strip">{error}</div>}

          <div className="fitengine-actions">
            <button type="button" disabled={busy} onClick={() => postAction('/api/fitengine/session/start', { height_cm: Number(heightCm || 170) })}>Start Session</button>
            <button type="button" disabled={busy || !readiness.can_capture_front} onClick={() => postAction('/api/fitengine/session/capture-front')}>Capture Front</button>
            <button type="button" disabled={busy || !readiness.can_capture_side} onClick={() => postAction('/api/fitengine/session/capture-side')}>Capture Side</button>
            <button type="button" disabled={busy || !readiness.can_finalize} onClick={() => postAction('/api/fitengine/session/finalize')}>Get My Size</button>
            <button type="button" disabled={busy} onClick={() => postAction('/api/fitengine/session/reset')}>Reset</button>
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
                <div className="fitengine-confidence-pill">{result.confidence != null ? `${Math.round(result.confidence)}%` : '--'}</div>
              </div>

              <div className="fitengine-size-hero">Size {result.recommended_size ?? '--'}</div>

              <div className="fitengine-mini-grid">
                <div className="fitengine-metric-row"><span>Collar</span><strong>{formatInches(result.collar_in)}</strong></div>
                <div className="fitengine-metric-row"><span>Trouser Waist</span><strong>{formatInches(result.trouser_waist_in)}</strong></div>
                <div className="fitengine-metric-row"><span>Chest</span><strong>{formatCm(result.chest_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Waist</span><strong>{formatCm(result.waist_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Shoulder</span><strong>{formatCm(result.shoulder_cm)}</strong></div>
                <div className="fitengine-metric-row"><span>Torso</span><strong>{formatCm(result.torso_cm)}</strong></div>
              </div>

              <div className="fitengine-summary-note">
                {result.summary || 'Capture both views, then generate the final size guidance.'}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default FitEngineMode;
