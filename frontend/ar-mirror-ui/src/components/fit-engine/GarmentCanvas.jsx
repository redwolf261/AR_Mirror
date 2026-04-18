/**
 * GarmentCanvas.jsx
 * Renders the selected garment image on a canvas overlaid on the webcam feed.
 * Uses pose skeleton points (normalized 0–1) to anchor and scale the garment
 * to the detected shoulder/hip region in real time.
 *
 * Works in offline mode (no Python backend) using the mock pose animation.
 */
import { useEffect, useRef } from 'react';

const POSE_KEYS = [
  'nose',           // 0
  'left_eye',       // 1
  'right_eye',      // 2
  'left_ear',       // 3
  'right_ear',      // 4
  'left_shoulder',  // 5
  'right_shoulder', // 6
  'left_elbow',     // 7
  'right_elbow',    // 8
  'left_wrist',     // 9
  'right_wrist',    // 10
  'left_hip',       // 11
  'right_hip',      // 12
];

/**
 * Given the posePoints array (normalized {x,y} objects), look up each
 * named keypoint. Handles both named objects (from mock) and indexed arrays.
 */
function getPt(posePoints, name) {
  if (!posePoints?.length) return null;
  // Named objects (mock data uses id strings)
  const byId = posePoints.find((p) => p.id === name);
  if (byId) return byId;
  // Indexed fallback (backend-normalized)
  const idx = POSE_KEYS.indexOf(name);
  return idx >= 0 ? posePoints[idx] : null;
}

export default function GarmentCanvas({ posePoints, garmentSrc, videoRef, visible }) {
  const canvasRef  = useRef(null);
  const imgRef     = useRef(null);
  const rafRef     = useRef(null);

  // Load garment image whenever it changes
  useEffect(() => {
    imgRef.current = null;
    if (!garmentSrc) return;
    const img = new Image();
    img.src = garmentSrc;
    img.onload = () => { imgRef.current = img; };
  }, [garmentSrc]);

  // The animation loop — draw garment every frame
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const frame = () => {
      rafRef.current = requestAnimationFrame(frame);

      const video = videoRef?.current;
      const vw = video?.videoWidth  || canvas.offsetWidth  || 640;
      const vh = video?.videoHeight || canvas.offsetHeight || 480;

      // Resize canvas to match video dimensions
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width  = vw;
        canvas.height = vh;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const img = imgRef.current;
      if (!img || !visible || !posePoints?.length) return;

      const W = canvas.width;
      const H = canvas.height;

      // Retrieve key landmarks (normalized coords from mock/backend)
      const ls = getPt(posePoints, 'left_shoulder');
      const rs = getPt(posePoints, 'right_shoulder');
      const lh = getPt(posePoints, 'left_hip');
      const rh = getPt(posePoints, 'right_hip');

      if (!ls || !rs) return;

      // Convert normalized → pixel coords
      // Canvas is already CSS-mirrored (scaleX(-1)), so draw at normal x and CSS flips it!
      const px   = (nx) => nx * W;
      const py   = (ny) => ny * H;

      const lsX = px(ls.x); const lsY = py(ls.y);
      const rsX = px(rs.x); const rsY = py(rs.y);
      const lhX = lh ? px(lh.x) : lsX - 10;
      const lhY = lh ? py(lh.y) : lsY + W * 0.45;
      const rhX = rh ? px(rh.x) : rsX + 10;
      const rhY = rh ? py(rh.y) : rsY + W * 0.45;

      // Garment bounding box
      const shoulderMidX = (lsX + rsX) / 2;
      const shoulderMidY = (lsY + rsY) / 2;
      const hipMidY      = (lhY + rhY) / 2;

      const shoulderSpan  = Math.abs(rsX - lsX);
      const garmentWidth  = shoulderSpan * 2.0;   // padding around shoulders
      const garmentHeight = (hipMidY - shoulderMidY) * 1.55; // extend below hips

      const drawX = shoulderMidX - garmentWidth / 2;
      const drawY = shoulderMidY - garmentHeight * 0.12; // slight upward anchor

      // Draw garment with semi-transparency so body is still visible
      ctx.save();
      ctx.globalAlpha = 0.82;
      ctx.drawImage(img, drawX, drawY, garmentWidth, Math.max(garmentHeight, garmentWidth * 1.1));
      ctx.restore();
    };

    rafRef.current = requestAnimationFrame(frame);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [posePoints, videoRef, visible]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        zIndex: 3,
        pointerEvents: 'none',
        // Mirror to match the CSS-flipped video
        transform: 'scaleX(-1)',
      }}
    />
  );
}
