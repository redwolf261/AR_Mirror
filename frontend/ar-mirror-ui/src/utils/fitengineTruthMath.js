export const POSE_INDEX = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
};

export function distance3D(a, b) {
  if (!a || !b) return null;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function midpoint(a, b) {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
  };
}

export function calibrateScaleFactor(worldLandmarks, userHeightCm) {
  if (!Array.isArray(worldLandmarks) || worldLandmarks.length < 31) return null;
  const head = worldLandmarks[POSE_INDEX.NOSE];
  const leftHeel = worldLandmarks[POSE_INDEX.LEFT_HEEL];
  const rightHeel = worldLandmarks[POSE_INDEX.RIGHT_HEEL];
  const heelMid = midpoint(leftHeel, rightHeel);
  const modelHeight = distance3D(head, heelMid);
  if (!modelHeight || modelHeight <= 1e-6) return null;
  return userHeightCm / modelHeight;
}

export function getMetricDistance(p1, p2, scaleFactor) {
  const raw = distance3D(p1, p2);
  if (!raw || !scaleFactor) return null;
  return raw * scaleFactor;
}

export function ellipseCircumference(widthCm, depthCm) {
  if (!widthCm || !depthCm || widthCm <= 0 || depthCm <= 0) return null;
  const a = widthCm / 2;
  const b = depthCm / 2;
  return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
}

export function buildFrontMetrics(worldLandmarks, userHeightCm) {
  const scaleFactor = calibrateScaleFactor(worldLandmarks, userHeightCm);
  if (!scaleFactor) return null;

  const ls = worldLandmarks[POSE_INDEX.LEFT_SHOULDER];
  const rs = worldLandmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lh = worldLandmarks[POSE_INDEX.LEFT_HIP];
  const rh = worldLandmarks[POSE_INDEX.RIGHT_HIP];
  const shoulderMid = midpoint(ls, rs);
  const hipMid = midpoint(lh, rh);

  const shoulderCm = getMetricDistance(ls, rs, scaleFactor);
  const torsoCm = getMetricDistance(shoulderMid, hipMid, scaleFactor);
  const chestWidthCm = shoulderCm ? shoulderCm * 0.92 : null;
  const waistWidthCm = getMetricDistance(lh, rh, scaleFactor);

  return {
    scaleFactor,
    shoulderCm,
    torsoCm,
    chestWidthCm,
    waistWidthCm,
  };
}

export function buildSideMetrics(worldLandmarks, userHeightCm) {
  const scaleFactor = calibrateScaleFactor(worldLandmarks, userHeightCm);
  if (!scaleFactor) return null;

  const ls = worldLandmarks[POSE_INDEX.LEFT_SHOULDER];
  const rs = worldLandmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lh = worldLandmarks[POSE_INDEX.LEFT_HIP];
  const rh = worldLandmarks[POSE_INDEX.RIGHT_HIP];

  const shoulderDepthRaw = ls && rs ? Math.abs(ls.z - rs.z) : null;
  const hipDepthRaw = lh && rh ? Math.abs(lh.z - rh.z) : null;

  return {
    scaleFactor,
    chestDepthCm: shoulderDepthRaw ? shoulderDepthRaw * scaleFactor : null,
    waistDepthCm: hipDepthRaw ? hipDepthRaw * scaleFactor : null,
  };
}

export function fuseTruth(frontMetrics, sideMetrics) {
  if (!frontMetrics || !sideMetrics) return null;

  const chestCirc = ellipseCircumference(frontMetrics.chestWidthCm, sideMetrics.chestDepthCm);
  const waistCirc = ellipseCircumference(frontMetrics.waistWidthCm, sideMetrics.waistDepthCm);

  const scaleDrift = frontMetrics.scaleFactor && sideMetrics.scaleFactor
    ? Math.abs(frontMetrics.scaleFactor - sideMetrics.scaleFactor) / Math.max(frontMetrics.scaleFactor, sideMetrics.scaleFactor)
    : 0.25;

  const chestWidthCm = frontMetrics.chestWidthCm ?? null;
  const chestDepthCm = sideMetrics.chestDepthCm ?? null;
  const waistWidthCm = frontMetrics.waistWidthCm ?? null;
  const waistDepthCm = sideMetrics.waistDepthCm ?? null;

  const hasChestEllipse = Boolean(chestWidthCm && chestDepthCm && chestWidthCm > 0 && chestDepthCm > 0);
  const hasWaistEllipse = Boolean(waistWidthCm && waistDepthCm && waistWidthCm > 0 && waistDepthCm > 0);

  let gatePenalty = 0;
  if (!hasChestEllipse) gatePenalty += 12;
  if (!hasWaistEllipse) gatePenalty += 12;
  if (scaleDrift > 0.22) gatePenalty += 14;

  const truthConfidence = Math.max(18, Math.min(98, Math.round(((1 - scaleDrift) * 100) - gatePenalty)));

  return {
    shoulder_cm: frontMetrics.shoulderCm,
    torso_cm: frontMetrics.torsoCm,
    chest_circumference_cm: chestCirc,
    waist_circumference_cm: waistCirc,
    truth_confidence: truthConfidence,
    scale_factor_front: frontMetrics.scaleFactor,
    scale_factor_side: sideMetrics.scaleFactor,
    diagnostics: {
      scale_drift_pct: Math.round(scaleDrift * 1000) / 10,
      chest_width_cm: chestWidthCm,
      chest_depth_cm: chestDepthCm,
      waist_width_cm: waistWidthCm,
      waist_depth_cm: waistDepthCm,
      has_chest_ellipse: hasChestEllipse,
      has_waist_ellipse: hasWaistEllipse,
      deterministic_single_frame: true,
    },
  };
}
