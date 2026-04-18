import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  captureFitEngineFront,
  captureFitEngineSide,
  finalizeFitEngineSession,
  getAccuracyMetrics,
  getBackendGarments,
  getBackendState,
  resetFitEngineSession,
  setBackendGarment,
  startFitEngineSession,
} from '../lib/backendApi';
import { normalizeGarments } from '../lib/backendTransforms';

function describeError(err) {
  if (!err) {
    return null;
  }
  return err instanceof Error ? err.message : String(err);
}

export function useBackendBridge() {
  const [state, setState] = useState(null);
  const [garments, setGarments] = useState([]);
  const [accuracyMetrics, setAccuracyMetrics] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const [snapshot, garmentPayload, metricsPayload] = await Promise.all([
        getBackendState(),
        getBackendGarments(),
        getAccuracyMetrics().catch(() => null),
      ]);
      setState(snapshot);
      setGarments(normalizeGarments(garmentPayload?.garments || []));
      setAccuracyMetrics(metricsPayload);
      setConnected(true);
      setError(null);
      setLastUpdatedAt(Date.now());
      return snapshot;
    } catch (err) {
      setConnected(false);
      setError(describeError(err));
      return null;
    }
  }, []);

  useEffect(() => {
    let active = true;
    let timer = null;

    const loop = async () => {
      await refresh();
      if (active) {
        timer = window.setTimeout(loop, 650);
      }
    };

    loop();

    return () => {
      active = false;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [refresh]);

  const actions = useMemo(() => ({
    startSession: async (height_cm = 170) => {
      await startFitEngineSession(height_cm);
      await refresh();
    },
    captureFront: async () => {
      await captureFitEngineFront();
      await refresh();
    },
    captureSide: async () => {
      await captureFitEngineSide();
      await refresh();
    },
    finalizeSession: async () => {
      await finalizeFitEngineSession();
      await refresh();
    },
    resetSession: async () => {
      await resetFitEngineSession();
      await refresh();
    },
    selectGarment: async (name) => {
      await setBackendGarment(name);
      await refresh();
    },
    refresh,
  }), [refresh]);

  return {
    state,
    garments,
    accuracyMetrics,
    connected,
    error,
    lastUpdatedAt,
    actions,
  };
}
