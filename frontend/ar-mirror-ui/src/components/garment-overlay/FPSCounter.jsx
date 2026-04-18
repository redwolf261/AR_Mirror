import { useEffect, useState } from 'react';

export default function FPSCounter({ fps }) {
  const [smooth, setSmooth] = useState(fps);

  useEffect(() => {
    const id = setInterval(() => {
      setSmooth((v) => v + (fps - v) * 0.28);
    }, 120);
    return () => clearInterval(id);
  }, [fps]);

  return <>{smooth.toFixed(1)}</>;
}
