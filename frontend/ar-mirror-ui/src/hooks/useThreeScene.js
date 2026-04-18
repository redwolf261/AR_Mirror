import { useEffect } from 'react';

export function useThreeScene(initScene, deps = []) {
  useEffect(() => {
    const cleanup = initScene?.();
    return () => {
      if (typeof cleanup === 'function') {
        cleanup();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
