import { useEffect, useRef, useState } from 'react';

const DEFAULT_WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8765';

export function useWebSocket() {
  const [state, setState] = useState({
    connected: false,
    latency: null,
    streamQuality: 'OFF',
    backendTick: 0,
  });
  const retryTimerRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    let active = true;

    const clearRetry = () => {
      if (retryTimerRef.current) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    };

    const connect = () => {
      if (!active) {
        return;
      }

      clearRetry();

      try {
        const socket = new WebSocket(DEFAULT_WS_URL);
        socketRef.current = socket;

        socket.onopen = () => {
          if (!active) {
            socket.close();
            return;
          }
          setState((prev) => ({
            ...prev,
            connected: true,
            streamQuality: 'LIVE',
            latency: prev.latency ?? 0,
          }));
        };

        socket.onmessage = (event) => {
          if (!active) {
            return;
          }

          let payload = null;
          try {
            payload = JSON.parse(event.data);
          } catch {
            payload = null;
          }

          const timestamp = Number(payload?.timestamp ?? payload?.ts_ms ?? payload?.timestampMs ?? 0);
          const latency = Number.isFinite(timestamp) && timestamp > 0
            ? Math.max(0, Math.round(Date.now() - timestamp))
            : null;

          setState((prev) => ({
            ...prev,
            connected: true,
            latency,
            streamQuality: payload?.pose2D?.length ? 'LIVE' : 'CONNECTED',
            backendTick: prev.backendTick + 1,
          }));
        };

        socket.onerror = () => {
          if (!active) {
            return;
          }
          setState((prev) => ({
            ...prev,
            connected: false,
            streamQuality: 'OFF',
          }));
        };

        socket.onclose = () => {
          if (!active) {
            return;
          }

          setState((prev) => ({
            ...prev,
            connected: false,
            streamQuality: 'OFF',
          }));

          retryTimerRef.current = window.setTimeout(connect, 1200);
        };
      } catch {
        setState((prev) => ({
          ...prev,
          connected: false,
          streamQuality: 'OFF',
        }));
        retryTimerRef.current = window.setTimeout(connect, 1500);
      }
    };

    connect();

    return () => {
      active = false;
      clearRetry();
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  return state;
}
