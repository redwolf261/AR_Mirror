import { useEffect, useState } from 'react';

export function useWebSocket() {
  const [state, setState] = useState({
    connected: true,
    latency: 22,
    streamQuality: 'HD',
    backendTick: 0,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setState((prev) => ({
        ...prev,
        latency: 18 + Math.round(Math.random() * 16),
        backendTick: prev.backendTick + 1,
      }));
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return state;
}
