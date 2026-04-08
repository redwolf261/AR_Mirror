import { useEffect, useRef, useState } from 'react';

export function useFrameTime(alpha = 0.1) {
  const [fps, setFps] = useState(30);
  const [frameMs, setFrameMs] = useState(33.3);
  const emaRef = useRef(33.3);
  const lastTimeRef = useRef(performance.now());
  const logCounterRef = useRef(0);

  useEffect(() => {
    let raf;

    const tick = () => {
      const now = performance.now();
      const delta = now - lastTimeRef.current;
      lastTimeRef.current = now;
      emaRef.current = alpha * delta + (1 - alpha) * emaRef.current;
      setFrameMs(emaRef.current);
      setFps(Math.max(1, Math.round(1000 / emaRef.current)));

      logCounterRef.current += 1;
      if (logCounterRef.current % 60 === 0) {
        console.log('[FrameTime]', {
          rawMs: Number(delta.toFixed(2)),
          emaMs: Number(emaRef.current.toFixed(2)),
          fps: Math.max(1, Math.round(1000 / emaRef.current)),
        });
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [alpha]);

  return { fps, frameMs, emaMs: emaRef.current };
}
