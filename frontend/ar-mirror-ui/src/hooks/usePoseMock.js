import { useEffect, useMemo, useState } from 'react';
import { INITIAL_POSE } from '../data/mockData';

export function usePoseMock({ width = 640, height = 480, phaseOffset = 0 } = {}) {
  const base = useMemo(() => {
    return INITIAL_POSE.map((kp) => ({
      ...kp,
      x: kp.x / 640,
      y: kp.y / 480,
      confidence: 0.9,
    }));
  }, []);

  const [keypoints, setKeypoints] = useState(base);

  useEffect(() => {
    let t = 0;
    const id = setInterval(() => {
      t += 0.08;
      setKeypoints(
        base.map((kp, i) => ({
          ...kp,
          x: kp.x + (Math.sin(t + i * 0.35 + phaseOffset) * (2 + (i % 3))) / 640,
          y: kp.y + (Math.cos(t * 0.9 + i * 0.27 + phaseOffset) * (1.8 + (i % 2))) / 480,
          confidence: 0.85 + Math.random() * 0.15,
        }))
      );
    }, 100);

    return () => clearInterval(id);
  }, [base, phaseOffset]);

  return keypoints;
}
