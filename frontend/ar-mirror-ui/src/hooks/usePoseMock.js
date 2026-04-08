import { useEffect, useMemo, useState } from 'react';
import { INITIAL_POSE } from '../data/mockData';

export function usePoseMock({ width = 640, height = 480, phaseOffset = 0 } = {}) {
  const base = useMemo(() => {
    const sx = width / 640;
    const sy = height / 480;
    return INITIAL_POSE.map((kp) => ({
      ...kp,
      x: kp.x * sx,
      y: kp.y * sy,
      confidence: 0.9,
    }));
  }, [width, height]);

  const [keypoints, setKeypoints] = useState(base);

  useEffect(() => {
    let t = 0;
    const id = setInterval(() => {
      t += 0.08;
      setKeypoints(
        base.map((kp, i) => ({
          ...kp,
          x: kp.x + Math.sin(t + i * 0.35 + phaseOffset) * (2 + (i % 3)),
          y: kp.y + Math.cos(t * 0.9 + i * 0.27 + phaseOffset) * (1.8 + (i % 2)),
          confidence: 0.85 + Math.random() * 0.15,
        }))
      );
    }, 100);

    return () => clearInterval(id);
  }, [base, phaseOffset]);

  return keypoints;
}
