/**
 * useWebcam.js
 * Browser WebRTC camera feed — works without Python backend.
 *
 * Bug fix: video element may not be mounted when getUserMedia resolves.
 * Solution: poll until videoRef.current exists before assigning srcObject.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export function useWebcam({ enabled = true } = {}) {
  const videoRef  = useRef(null);
  const streamRef = useRef(null);
  const [ready, setReady]   = useState(false);
  const [error, setError]   = useState(null);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width:      { ideal: 1280 },
            height:     { ideal: 720  },
            facingMode: 'user',
            frameRate:  { ideal: 30  },
          },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        // Poll until the <video> element is actually mounted in the DOM
        const attach = () => {
          if (cancelled) return;
          const video = videoRef.current;
          if (!video) {
            setTimeout(attach, 50);
            return;
          }
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play()
              .then(() => { if (!cancelled) setReady(true); })
              .catch((e) => { console.warn('[useWebcam] play():', e); });
          };
          // Fallback: some browsers fire canplay instead
          video.oncanplay = () => {
            if (!cancelled && !ready) setReady(true);
          };
        };

        attach();
      } catch (err) {
        if (!cancelled) {
          console.error('[useWebcam] getUserMedia failed:', err);
          setError(err.message || 'Camera access denied');
        }
      }
    }

    start();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setReady(false);
    };
  }, [enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Capture the current frame as a base64 JPEG string (no data: prefix).
   * Mirrors horizontally to match the selfie-mode display.
   */
  const captureSnapshot = useCallback(
    (quality = 0.88) => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return null;

      const w = video.videoWidth  || 1280;
      const h = video.videoHeight || 720;

      const canvas = document.createElement('canvas');
      canvas.width  = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');

      // Mirror so captured image matches what user sees
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0);

      return canvas.toDataURL('image/jpeg', quality).split(',')[1];
    },
    [],
  );

  return { videoRef, ready, error, captureSnapshot };
}
