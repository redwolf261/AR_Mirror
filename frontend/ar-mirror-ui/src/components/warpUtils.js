export const LANDMARK_INDEX = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
};

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

const pointFromPacket = (packet, idx) => {
  const arr = packet?.pose2D;
  if (!Array.isArray(arr)) return null;
  const hit = arr.find((pt) => Array.isArray(pt) && pt[0] === idx);
  if (!hit) return null;
  const [, x, y, vis] = hit;
  if (typeof x !== 'number' || typeof y !== 'number') return null;
  return { x, y, vis: typeof vis === 'number' ? vis : 0 };
};

export const extractControlPoints = (packet) => {
  const ls = pointFromPacket(packet, LANDMARK_INDEX.LEFT_SHOULDER);
  const rs = pointFromPacket(packet, LANDMARK_INDEX.RIGHT_SHOULDER);
  const lh = pointFromPacket(packet, LANDMARK_INDEX.LEFT_HIP);
  const rh = pointFromPacket(packet, LANDMARK_INDEX.RIGHT_HIP);

  if (!ls || !rs || !lh || !rh) {
    return null;
  }

  const shoulderCenter = { x: (ls.x + rs.x) * 0.5, y: (ls.y + rs.y) * 0.5 };
  const hipCenter = { x: (lh.x + rh.x) * 0.5, y: (lh.y + rh.y) * 0.5 };
  const torsoCenter = { x: (shoulderCenter.x + hipCenter.x) * 0.5, y: (shoulderCenter.y + hipCenter.y) * 0.5 };

  const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  const hipWidth = Math.hypot(lh.x - rh.x, lh.y - rh.y);
  const torsoHeight = Math.max(0.001, Math.hypot(shoulderCenter.x - hipCenter.x, shoulderCenter.y - hipCenter.y));

  const vis = [ls.vis, rs.vis, lh.vis, rh.vis].filter((v) => Number.isFinite(v));
  const confidence = vis.length ? vis.reduce((a, b) => a + b, 0) / vis.length : 0;

  return {
    ls,
    rs,
    lh,
    rh,
    shoulderCenter,
    hipCenter,
    torsoCenter,
    shoulderWidth,
    hipWidth,
    torsoHeight,
    confidence: clamp(confidence, 0, 1),
  };
};

export const computeGarmentPose = (packet, prevPose) => {
  const cp = extractControlPoints(packet);
  if (!cp) {
    return prevPose || null;
  }

  const backendAnchor = packet?.anchor;
  const backendScale = typeof packet?.scale === 'number' ? packet.scale : null;

  const targetAnchor = {
    x: typeof backendAnchor?.x === 'number' ? backendAnchor.x : cp.torsoCenter.x,
    y: typeof backendAnchor?.y === 'number' ? backendAnchor.y : cp.torsoCenter.y,
  };
  const targetScale = backendScale ?? (cp.shoulderWidth * 2.2);

  const alpha = 0.22;
  const anchor = prevPose
    ? {
        x: prevPose.anchor.x + (targetAnchor.x - prevPose.anchor.x) * alpha,
        y: prevPose.anchor.y + (targetAnchor.y - prevPose.anchor.y) * alpha,
      }
    : targetAnchor;
  const scale = prevPose ? prevPose.scale + (targetScale - prevPose.scale) * alpha : targetScale;

  return {
    anchor,
    scale: clamp(scale, 0.12, 1.6),
    cp,
  };
};

export const drawWarpedGarment = (ctx, garmentImg, pose, width, height) => {
  if (!ctx || !garmentImg || !pose) return;

  const { cp, anchor, scale } = pose;
  const conf = cp?.confidence ?? 0;

  const centerX = anchor.x * width;
  const centerY = anchor.y * height;
  const shoulderPx = cp.shoulderWidth * width;
  const hipPx = cp.hipWidth * width;
  const torsoPx = cp.torsoHeight * height;

  const garmentH = clamp(torsoPx * 2.1 * scale, 120, height * 0.78);
  const topW = clamp(shoulderPx * 2.2 * scale, 80, width * 0.8);
  const midW = clamp(((shoulderPx + hipPx) * 0.5) * 2.3 * scale, 90, width * 0.84);
  const botW = clamp(hipPx * 2.5 * scale, 90, width * 0.86);

  const topY = centerY - garmentH * 0.42;
  const skew = (cp.ls.y - cp.rs.y) * height * 0.35;

  if (conf < 0.45) {
    const affineW = midW;
    ctx.save();
    ctx.globalAlpha = 0.78;
    ctx.drawImage(garmentImg, centerX - affineW * 0.5, topY, affineW, garmentH);
    ctx.restore();
    return;
  }

  // Piecewise warp approximation with 3 horizontal bands for real-time speed.
  const bands = [
    { sy: 0.0, sh: 0.34, w: topW, yShift: -skew * 0.35 },
    { sy: 0.34, sh: 0.33, w: midW, yShift: 0 },
    { sy: 0.67, sh: 0.33, w: botW, yShift: skew * 0.25 },
  ];

  ctx.save();
  ctx.globalAlpha = 0.84;
  bands.forEach((b) => {
    const srcY = garmentImg.height * b.sy;
    const srcH = garmentImg.height * b.sh;
    const destY = topY + garmentH * b.sy + b.yShift;
    const destH = garmentH * b.sh;
    ctx.drawImage(
      garmentImg,
      0,
      srcY,
      garmentImg.width,
      srcH,
      centerX - b.w * 0.5,
      destY,
      b.w,
      destH,
    );
  });
  ctx.restore();
};

export const buildForegroundMask = (ctx, packet, width, height) => {
  if (!ctx) return;

  const seg = packet?.segmentation;
  const segPoly = seg?.maskPolygon;
  if (seg?.maskAvailable && Array.isArray(segPoly) && segPoly.length >= 3) {
    ctx.beginPath();
    segPoly.forEach((pt, i) => {
      if (!Array.isArray(pt) || pt.length < 2) return;
      const x = pt[0] * width;
      const y = pt[1] * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
    return;
  }

  const cp = extractControlPoints(packet);
  if (!cp) return;

  const nose = pointFromPacket(packet, LANDMARK_INDEX.NOSE);
  const lw = pointFromPacket(packet, LANDMARK_INDEX.LEFT_WRIST);
  const rw = pointFromPacket(packet, LANDMARK_INDEX.RIGHT_WRIST);

  const shoulderL = { x: cp.ls.x * width, y: cp.ls.y * height };
  const shoulderR = { x: cp.rs.x * width, y: cp.rs.y * height };
  const hipL = { x: cp.lh.x * width, y: cp.lh.y * height };
  const hipR = { x: cp.rh.x * width, y: cp.rh.y * height };

  ctx.beginPath();
  ctx.moveTo(shoulderL.x - 18, shoulderL.y - 6);
  ctx.lineTo(shoulderR.x + 18, shoulderR.y - 6);
  ctx.lineTo(hipR.x + 14, hipR.y + 12);
  ctx.lineTo(hipL.x - 14, hipL.y + 12);
  ctx.closePath();
  ctx.fill();

  if (nose) {
    const hx = nose.x * width;
    const hy = nose.y * height;
    const hr = clamp(cp.shoulderWidth * width * 0.55, 24, 110);
    ctx.beginPath();
    ctx.arc(hx, hy, hr, 0, Math.PI * 2);
    ctx.fill();
  }

  [lw, rw].forEach((wpt) => {
    if (!wpt || (wpt.vis ?? 0) < 0.3) return;
    ctx.beginPath();
    ctx.arc(wpt.x * width, wpt.y * height, 26, 0, Math.PI * 2);
    ctx.fill();
  });
};

export const drawDebug = (ctx, packet, width, height, options = {}) => {
  const { showKeypoints = false, showWarpGrid = false, showMask = false } = options;
  const cp = extractControlPoints(packet);
  if (!cp) return;

  if (showWarpGrid) {
    ctx.save();
    ctx.strokeStyle = 'rgba(85, 183, 255, 0.85)';
    ctx.lineWidth = 1;
    [cp.ls, cp.rs, cp.lh, cp.rh].forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, 4, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.beginPath();
    ctx.moveTo(cp.ls.x * width, cp.ls.y * height);
    ctx.lineTo(cp.rs.x * width, cp.rs.y * height);
    ctx.lineTo(cp.rh.x * width, cp.rh.y * height);
    ctx.lineTo(cp.lh.x * width, cp.lh.y * height);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  if (showKeypoints && Array.isArray(packet?.pose2D)) {
    ctx.save();
    ctx.fillStyle = 'rgba(255, 233, 120, 0.85)';
    packet.pose2D.forEach((pt) => {
      if (!Array.isArray(pt) || pt.length < 4) return;
      const [, x, y, vis] = pt;
      if ((vis ?? 0) < 0.25) return;
      ctx.beginPath();
      ctx.arc(x * width, y * height, 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  if (showMask) {
    ctx.save();
    ctx.fillStyle = 'rgba(0, 255, 170, 0.18)';
    buildForegroundMask(ctx, packet, width, height);
    ctx.restore();
  }
};
